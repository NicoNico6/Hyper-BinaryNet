import os
import math
import random
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from models.wide_resnet_binary import *
from models.alexnet_binary import *
from models.resnet_binary import *
from models.vgg_binary import *
from models.modules import *
from models.cyclic_lr import *
from models.vgg_small import *
from models.vgg_variant import *
from models.resnet_imagenet import *

import pdb
import gc 

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, 12 + device_id, True, True, set_affinity = True)
        self.input = ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=1, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        self.iteration = torch.zeros(1).cuda()
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB)
            self.res = ops.Resize(resize_x=crop, resize_y=crop)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512)
            self.res = ops.RandomResizedCrop(device=dali_device, size =(crop, crop))

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        
        return [output, self.labels.gpu()]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, 12 + device_id, True, True, set_affinity = True)
        self.iteration = torch.zeros(1).cuda()
        self.input = ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=1, random_shuffle=False)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        self.iteration += 1
        if self.iteration % 200 == 0:
          del images, self.jpegs
        return [output, self.labels]

    def iter_setup(self):
        gc.collect()
        

if __name__ == "__main__":
  
   parser = argparse.ArgumentParser(description='PyTorch BH-Net Training')
   parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
   parser.add_argument('--train_batch_size', type = int, default = 256, help = 'batch_size of train_dataloader')
   parser.add_argument('--test_batch_size', type = int, default = 256, help = 'batch_size of test_dataloader')
   parser.add_argument('--num_workers', type = int, default = 8, help = "dataloader workers")
   parser.add_argument('--model', type = str, default = 'ResNet18', help = 'models to be trained')
   parser.add_argument('--root_dir', type = str, default = 'BH_Logs', help = 'dir of saving logs')
   parser.add_argument('--dataset', type = str, default = 'CIFAR10', help = 'dataset to be used')
   parser.add_argument('--set_cuda_device', type = int, default = 0, help = 'use which gpu to train BNN')
   parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'training optimizer')
   parser.add_argument('--z_dim', type = int, default = 9, help = 'layer discription')
   parser.add_argument('--skip_activation_binarize', action = 'store_true', help = 'activation binarizing scheme of convolution in skip connection ')
   parser.add_argument('--skip_weight_binarize', action = 'store_true', help = 'weight binarizing scheme of convolution in skip connection ')
   parser.add_argument('--skip_kernel_size', type = int, default = 3, help = 'kernel size of convolution in skip connection ')
   parser.add_argument('--full_binarize', action = 'store_true', help = 'full binarizing or weight only binarizing')
   parser.add_argument('--ste', type = str, default = 'clipped_elu', help = 'full binarizing or weight only binarizing')
   parser.add_argument('--weight_binarize', action = 'store_true', help = 'weight binarizing or not')
   parser.add_argument('--hyper_accumulation', action = 'store_true', help = 'hyper_accumulation or not')
   parser.add_argument('--depth', type = int, default = 1, help = 'depth of hypernetwork')
   parser.add_argument('--criterion', type = str, default = 'CrossEntropy', help = 'depth of hypernetwork')
   parser.add_argument('--multi_gpu', action = 'store_true', help = 'multi gpu training or not')
   parser.add_argument('--dali_cpu', action = 'store_true', help = 'loading data using cpu')
   parser.add_argument('--reuse', action = 'store_true', help = 'traing with pretrained model')
   opt = parser.parse_args()
   
   valdir = '/media/data/ILSVRC2012/ILSVRC2012_img_val'
    
   crop_size = 224
   val_size = 256
   cudnn.benchmark = True   
   
   #pipe = HybridValPipe(batch_size=opt.test_batch_size, num_threads=opt.num_workers, device_id=0, data_dir=valdir, crop=crop_size, size=val_size)
   #pipe.build()
   #testloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
   
   net = ResNet_ImageNet(num_classes = 1000, inflate = 4, opt = opt, z_dim = opt.z_dim)
   name = net.name
   max_epochs = net.optim_params['Max_Epochs']
   
   #state_dict = torch.load("results/ImageNet/Model/resnet_imagenet/BH_resnet_imagenet_9.pth.zip")
   #print(state_dict['net'])
   #net.load_state_dict(state_dict['net'])
   net = torch.load('checkpoints/ImageNet/resnet_imagenet/BH_resnet_imagenet_9_Best.pth.zip')['net'].cpu()
   input = torch.randn(1,3,224, 224)
   output = net(input)
   print(output)