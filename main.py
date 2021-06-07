import os
import math
import random
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from models.wide_resnet_binary import *
from models.alexnet_binary import *
from models.resnet_temperature import *
from models.vgg_binary import *
from models.modules import *
from models.cyclic_lr import *
from models.vgg_small import *
from models.vgg_variant import *
from models.googlenet_binary import *
from models.adam import *

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=True):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id, set_affinity = True)
        self.input = ops.FileReader(file_root=data_dir, num_shards = device_id+1, shard_id=device_id, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        device_memory_padding=211025920 if decoder_device == 'mixed' else 0
        host_memory_padding=140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device = decoder_device, output_type = types.RGB,
                                                device_memory_padding = device_memory_padding,
                                                host_memory_padding = host_memory_padding,
                                                random_aspect_ratio = [0.8, 1.25],
                                                random_area = [0.08, 1.0],
                                                num_attempts = 100)
        self.res = ops.Resize(device = dali_device, resize_x=crop, resize_y=crop, interp_type = types.INTERP_TRIANGULAR)
        
        ''' 
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB,
                                                    #random_aspect_ratio=[0.8, 1.25],
                                                    #random_area=[0.1, 1.0],
                                                    #num_attempts=100
                                                    )
                                                    
            self.res = ops.RandomResizedCrop(device=dali_device, size =(crop, crop))
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512)
                                                    
            self.res = ops.RandomResizedCrop(device=dali_device, size =(crop, crop))
        '''
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        
        return [output, self.labels]
        
        
class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id, set_affinity = True)
        
        self.input = ops.FileReader(file_root=data_dir, num_shards = device_id+1, shard_id=device_id,  random_shuffle=False)
        self.decode = ops.ImageDecoder(device = 'mixed', output_type = types.RGB)
        #self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device = 'gpu', resize_shorter = size, interp_type = types.INTERP_TRIANGULAR)
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
        #self.iteration += 1
        #if self.iteration % 200 == 0:
        #  del images, self.jpegs
        return [output, self.labels]

    def iter_setup(self):
        gc.collect()

def reuse(net, opt):
    if opt.resume:
      ckpt = torch.load('checkpoints/{}/BH_{}.pth.zip'.format(net.name, net.name))
      net.load_state_dict(ckpt['net'])
    
    return net
    
    
def get_data(opt, net):


    if not opt.dataset == 'ImageNet':
      transform_train = net.input_transforms['{}'.format(opt.dataset)]['train']
      transform_test = net.input_transforms['{}'.format(opt.dataset)]['eval']
    
      
      if opt.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='data/MNIST/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='data/MNIST/', train=False, download=True, transform=transform_test)
        
      elif opt.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='data/CIFAR10/', train=False, download=True, transform=transform_test)
    
      elif opt.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/CIFAR100/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='data/CIFAR100/', train=False, download=True, transform=transform_test)
      
      elif opt.dataset == 'ImageNet':
        traindir = '/media/opt48/data/gnh/dataset/ILSVRC2012/train'
        valdir = '/media/opt48/data/gnh/dataset/ILSVRC2012/val'
        trainset = torchvision.datasets.ImageFolder(root=traindir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=valdir, transform=transform_test)
      
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
      testloader = torch.utils.data.DataLoader(testset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    else:
      traindir = '/media/opt48/data/gnh/dataset/ILSVRC2012/train'
      valdir = '/media/opt48/data/gnh/dataset/ILSVRC2012/val'
    
      crop_size = 224
      val_size = 256
      
      pipe = HybridTrainPipe(batch_size=opt.train_batch_size, num_threads=opt.num_workers, device_id=int(opt.set_cuda_device), data_dir=traindir, crop=crop_size, dali_cpu=opt.dali_cpu)
      pipe.build()
      trainloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1), auto_reset = True)
      #trainloader = DALIClassificationIterator(pipe, size=1000)

      pipe = HybridValPipe(batch_size=opt.test_batch_size, num_threads=opt.num_workers, device_id=int(opt.set_cuda_device), data_dir=valdir, crop=crop_size, size=val_size)
      pipe.build()
      testloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1), auto_reset = True)
      #testloader = DALIClassificationIterator(pipe, size=1000)
    return trainloader, testloader
   

def get_writer(opt, net):
    root_dir = opt.root_dir
    net_dir = net.name
    
    log_dir = os.path.join(root_dir, net_dir)
    
    writer = SummaryWriter(log_dir)
    
    return writer


def get_model(opt, num_classes, inflate):
    
    if opt.model == 'ResNet':
      net = resnet(num_classes = num_classes, inflate = inflate, opt = opt, z_dim = opt.z_dim)
      
    elif opt.model == 'ResNet18':
      net = resnet18(num_classes = num_classes, inflate = inflate, opt = opt, z_dim = opt.z_dim)
    
    elif opt.model == 'ResNet20':
      net = resnet20(num_classes = num_classes, inflate = 1, opt = opt, z_dim = opt.z_dim)
        
    elif opt.model == 'ResNet34':
      net = resnet34(num_classes = num_classes, inflate = inflate, opt = opt, z_dim = opt.z_dim)
       
    elif opt.model == 'ResNet50':
      net = resnet50(num_classes = num_classes, inflate = inflate, opt = opt, z_dim = opt.z_dim)
    
    elif opt.model == 'ResNet56':
      net = resnet56(num_classes = num_classes, inflate = 1, opt = opt, z_dim = opt.z_dim)
       
    elif opt.model == 'ResNet101':
      net = resnet101(num_classes = num_classes, inflate = inflate, opt = opt, z_dim = opt.z_dim)
      
    elif opt.model == 'ResNet152':
      net = resnet152(num_classes = num_classes, inflate = inflate, opt = opt, z_dim = opt.z_dim)
      
    elif opt.model == 'WRN22':
      net = wrn22(num_classes = num_classes, opt = opt)
    
    elif opt.model == 'WRN40':
      net = wrn40(num_classes = num_classes, opt = opt)
    
    elif opt.model == 'vgg11_bn':
      net = vgg11_bn(opt = opt, num_classes = num_classes)
      
    elif opt.model == 'vgg13_bn':
      net = vgg13_bn(opt = opt, num_classes = num_classes)
      
    elif opt.model == 'vgg16_bn':
      net = vgg16_bn(opt = opt, num_classes = num_classes)
      
    elif opt.model == 'vgg19_bn':
      net = vgg19_bn(opt = opt, num_classes = num_classes)
      
    elif opt.model == 'vgg_small':
      net = vgg_small(num_classes = num_classes, opt = opt)
      
    elif opt.model == 'vgg_variant':
      net = vgg_variant(num_classes = num_classes, opt = opt)   
    
    elif opt.model == 'AlexNet':
      net = alexnet(num_classes = num_classes)
    
    elif opt.model == 'ShuffleNet':
      net = shufflenet(num_classes = num_classes, opt = opt)
    
    elif opt.model == 'MobileNet':
      net = mobilenet(num_classes = num_classes, opt = opt)
    
    elif opt.model == 'GoogleNet':
      net = googlenet(num_classes = num_classes, opt = opt)
   
        
    return net

def split_optim(net):
    for name, param in net.named_parameters():
        if name in ['random_z']:
          yield param
    
    
def get_optimizer(opt, net):
    
    optim_params = net.optim_params
    
    if opt.optimizer == 'Adam':
      optimizer = optim.Adam(
                        net.parameters(),
                        optim_params['Adam'][opt.dataset]['Init_lr'],
                        weight_decay = optim_params['Adam'][opt.dataset]['Weight_decay'],
                        betas = optim_params['Adam'][opt.dataset]['Betas'],
                        amsgrad = False 
                        )
    
    elif opt.optimizer == 'SGD':
      optimizer = optim.SGD(
                        net.parameters(), 
                        optim_params['SGD'][opt.dataset]['Init_lr'], 
                        momentum=optim_params['SGD'][opt.dataset]['Weight_momentum'], 
                        weight_decay=optim_params['SGD'][opt.dataset]['Weight_decay'],
                        nesterov  = False
                        )
                       
    else:
      raise ValueError('Only support Adam based optimization now ! Please identify coresponding optim_params !')
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=optim_params[opt.optimizer]['MultiStepLR']['step'], gamma=optim_params[opt.optimizer]['MultiStepLR']['ratio'])
    criterion = optim_params['Criterion']
        
    return optimizer, lr_scheduler,  criterion

        
def get_weight(net, opt):
    weight_l2 = torch.zeros(1)
    weight_mean = torch.zeros(1)
    num = torch.zeros(1)
    size = torch.zeros(1)
  
    for module in (net.module.modules() if opt.multi_gpu else net.modules()):
        if isinstance(module, BinarizedHypernetwork_Parrallel):
            num += 1
            weight_l2 += torch.sqrt((module.weight.data.cpu())**2 + 1e-8).sum() 
            weight_mean += abs(module.weight.data.cpu()).mean()
            size += module.weight.data.cpu().contiguous().view(-1).size(0)
        elif isinstance(module, BinarizeConv2d):
            num += 1
            weight_l2 += torch.sqrt((module.weight.data.cpu())**2 + 1e-8).sum() 
            weight_mean += abs(module.weight.data.cpu()).mean()
            size += module.weight.data.cpu().contiguous().view(-1).size(0)
               
    weight_mean = weight_mean/num
    weight_l2 = weight_l2/size                     
    return weight_l2, weight_mean

    
def adjust_momentum(net, epochs, opt):
    init_momentum = net.module.optim_params['BN_Momentum_Init']  if opt.multi_gpu else net.optim_params['BN_Momentum_Init'] 
    max_epochs =  net.module.optim_params['Max_Epochs']  if opt.multi_gpu else net.optim_params['Max_Epochs']
    for module in net.module.modules() if opt.multi_gpu else net.modules():
        if isinstance(module, (nn.BatchNorm2d)):
            momentum = max(init_momentum, init_momentum - (float(epochs))/max_epochs)
            module.momentum = momentum
     
    return momentum
    
    
def updatekernels(net):
    for module in net.modules():
      #print(module)
      if isinstance(module, BinaryKernels):
        
        module.update()
    
def reconstruction_loss(net):
    reconstruction_loss = 0.0
    for module in net.modules():
        if isinstance(module, BinarizedHypernetwork_Parrallel):
            reconstruction_loss += module.Loss
     
    return reconstruction_loss

              
def clip_grad(net):
    for module in net.modules():
        if isinstance(module, selu):
          torch.nn.utils.clip_grad_norm_(module.parameters(), 10, 2)
        else:
          torch.nn.utils.clip_grad_norm_(module.parameters(), 10, 2)
          
                    
def checkpoints(optimizer, opt, net, epochs, changes, running_train_loss, running_test_loss, running_train_prec1, running_train_prec5, running_test_prec1, running_test_prec5, running_weight_mean, running_best_prec1):
    
    if not os.path.exists('checkpoints'):
      os.mkdir('checkpoints')
    
    if opt.multi_gpu:
      name = net.module.name
    else:
      name = net.name
      
    if not os.path.exists('checkpoints/{}/'.format(opt.dataset)):
      os.mkdir('checkpoints/{}/'.format(opt.dataset))
    if not os.path.exists('checkpoints/{}/{}/'.format(opt.dataset, name)):
      os.mkdir('checkpoints/{}/{}/'.format(opt.dataset, name))
    
    state = {
        'net': net.state_dict(),
        'optimizer':optimizer.state_dict(), 
        'train_prec1': running_train_prec1,
        'train_prec5': running_train_prec5,
        'test_prec1': running_test_prec1,
        'test_prec5': running_test_prec5,
        'train_loss': running_train_loss,
        'test_loss': running_test_loss,
        'weight_mean': running_weight_mean,
        'best_acc': running_best_prec1,   
        'epochs': epochs,
        'changes': changes, 
      }
      
    #if opt.multi_gpu:
    #    nn.DataParallel(net, [0, 1, 2, 3])
    net = net.cuda()
      
    torch.save(state, 'checkpoints/{}/{}/BH_{}_{}_Current.zip'.format(opt.dataset, name, name, opt.note))  
    if running_test_prec1[-1] > running_best_prec1[-1]:
      
      running_best_prec1[-1] = running_test_prec1[-1]
      
      torch.save(state, 'checkpoints/{}/{}/BH_{}_{}_Best.pth.zip'.format(opt.dataset, name, name, opt.note))
    
      print("Saving model.....")
    
    return running_best_prec1[-1]
              
def accuracy(output, target, topk = (1,)):
    #maxk = max(topk)
    maxk = 5
    
    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    #print(pred.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum()
      res.append(correct_k)
    
    return res
    

def statistic(net):
    changes = torch.zeros(1)
    size = torch.zeros(1)
    for module in (net.module.modules() if opt.multi_gpu else net.modules()):
      if isinstance(module, BinarizeHyperConv2d):
        changes += (module.Weight.data.cpu().clone().sign() != module.init.cpu().clone().sign()).sum()
        module.init = module.Weight.clone()
        size += module.Weight.view(-1).size(0)  
      elif isinstance(module, BinarizeConv2d):
        changes += (module.Weight.data.cpu().clone().sign() != module.init.data.cpu().clone().sign()).sum()
        module.init = module.Weight.clone()
        size += module.Weight.view(-1).size(0)  
    return changes.float()/size*100


def adjust_temperoture(Init_T, epochs, iteration, opt):
    Temperature = 1*((Init_T *0.95**(epochs))) if not opt.dataset == 'ImageNet' else 1*((Init_T *0.985**(max(0, (max(0, iteration - 2*5000*5))//1000))))
    #T = Init_T*(0.5**(epochs//30))  * (math.cos(2*math.pi*float(epochs)/30) + 1)/2
    #T = 200 if not opt.dataset == 'ImageNet' else 70
    #Temperature = Init_T * 0.5*(1 + math.cos(float(epochs)*math.pi/T))
    Temperature_W = torch.zeros(1)
    Temperature_A = torch.zeros(1)
    Num_W = torch.zeros(1)
    Num_A = torch.zeros(1)
    for i, module in enumerate(net.module.modules() if opt.multi_gpu else net.modules()):
        if isinstance(module, (Softsign_W)):
          Num_W += 1
          module.Temperature = Temperature
          Temperature_W += module.Temperature
        
        if isinstance(module, (Softsign_A)):
          Num_A += 1
          module.Temperature = Temperature
          Temperature_A += module.Temperature
    
    
    return Temperature_W/Num_W, Temperature_A/Num_A, Temperature_A/Num_A

def adjust_architecture(net):
    for i, module in enumerate(net.modules()):
      if isinstance(module, BasicBlock):
        module.bn3 = nn.BatchNorm2d(module.out_chs, affine = True, momentum = 0.5, track_running_stats = True) 
      '''
      if isinstance(module, ResNet):
        module.conv1 = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias = False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
                            nn.BatchNorm2d(64),
                            )
      '''                                          
def train(net, train_dataloader, test_dataloader, optimizer, lr_scheduler, criterion, writer, opt, name, optim_params):
    
    best_accuracy = 0.
    print_freq = (10*2)
    total_iter = torch.zeros(1).cuda()
    epochs = 0.
    momentum = 0.5
    weight_L2, weight_mean, activation_mean, signs = torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)
    init_temperoture = opt.temperoture
    temperoture = init_temperoture

    max_epochs = net.optim_params['Max_Epochs'] if not opt.multi_gpu else net.module.optim_params['Max_Epochs'] 
    
    if state:
      running_train_prec1 = state['train_prec1']
      running_train_prec5 = state['train_prec5']
      running_test_prec1 = state['test_prec1']
      running_test_prec5 = state['test_prec5']
      running_best_prec1 = state['best_acc']
      running_train_loss = state['train_loss']
      running_test_loss = state['test_loss']
      current_epochs = state['epochs'] + 1
      running_weight_mean = []
      epochs = current_epochs 
      changes = state['changes']
      for k in range(int(current_epochs)):
        lr_scheduler.step()
      best_accuracy = state['best_acc'][-1]
    else:
      running_train_loss = []
      running_test_loss = []
      
      running_train_prec1 = []
      running_train_prec5 = []
      
      
      running_test_prec1 = []
      running_test_prec5 = []
      
      running_weight_mean = []
      running_best_prec1 = []
    
      current_epochs = epochs 
    
    
      changes = []
    
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize,  opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Loss',          {'train_loss':0, 'test_loss':0}, 0)
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Accuracy',      {'train_prec1':0, 'train_prec5':0, 'test_prec1':0, 'test_prec5':0}, 0)
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Best_Accuracy', {'best_accuracy':0}, 0) 
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Weight_L2',     {'L2':0}, 0)
    
    for n in range(int(current_epochs), max_epochs):
      
      running_loss = torch.zeros(1)
      
      train_total = 0.
      train_correct = 0
      avg_train_loss = torch.zeros(1)
      avg_train_prec1 = torch.zeros(1)
      avg_train_prec5 = torch.zeros(1)
      
      avg_test_loss = torch.zeros(1)
      avg_test_prec1 = torch.zeros(1)
      avg_test_prec5 = torch.zeros(1)
      
      #if n % 5 == 0:
      lr_scheduler.step()
      
      
      #temperoture = adjust_temperoture(Init_T = init_temperoture, epochs = n)
      #avg_temperoture_w, temperoture, avg_temperoture_a = adjust_temperoture(Init_T = init_temperoture, epochs = n, iteration = n)
      #net = net.cuda()
      
      
      signs = statistic(net) 
      
      #momentum = adjust_momentum(net = net, epochs = n, opt = opt)
      net.train()
      
        
      for i, data in enumerate(train_dataloader):
      
        train_total+=1.0
        if opt.dataset == 'ImageNet':
          avg_temperoture_w, temperoture, avg_temperoture_a = adjust_temperoture(Init_T = init_temperoture, epochs = n, iteration = (i + n*(train_dataloader._size / opt.train_batch_size)), opt = opt)
        
          inputs = Variable(data[0]['data'])
          labels = Variable(data[0]['label'].squeeze().long().cuda())
        else:
          avg_temperoture_w, temperoture, avg_temperoture_a = adjust_temperoture(Init_T = init_temperoture, epochs = n, iteration = n, opt = opt)
        
          inputs, labels = data
        
          inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        
        #target = labels
      
        
        
        # Forward + Backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #weight_L2, weight_mean, activation_mean = get_weight(net)
        
        #loss_sum = loss
        prec1, prec5 = accuracy(outputs.data, labels, (1,5))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        if i%2==0:
        	optimizer.step()
        	optimizer.zero_grad()
        
        
        #Top1 and Top2 Accuracy
        
        running_loss.add_(loss)
         
        avg_train_loss.add_(loss)
        avg_train_prec1.add_(prec1)
        avg_train_prec5.add_(prec5)
        
        #total_iter = i
        
        if i % print_freq == 0 and i != 0:
            print("[Epoch %d, Total Iterations %6d] Loss: %.4f  L2_weight: %.4f Weight_Mean: %.4f signs: %.2f  A_Mean: %3f Lr: %.8f Momentum: %.3f weight: %.3f, best test accuracy: %.4f T: %.6f, T_A: %.3f" % (epochs + 1, (i + 1), running_loss.item()/print_freq, weight_L2.data.item(), weight_mean.data.item(), signs.data.item(), activation_mean.data.item(), optimizer.param_groups[0]['lr'], momentum, 1, best_accuracy, avg_temperoture_w, avg_temperoture_a))
            running_loss.zero_()
            

        
      
      test_total = 0
      test_correct = 0
      
      with torch.no_grad():
        for tdata in test_dataloader:
          test_total += 1
          if opt.dataset == 'ImageNet':
            timages = Variable(tdata[0]['data'])
            tlabels = Variable(tdata[0]['label'].squeeze().cuda().long())
          else:
            
            timages, tlabels = tdata
            timages, tlabels = Variable(timages.cuda()), Variable(tlabels.cuda())
            
          target = tlabels
          
          if opt.criterion == 'HingleLoss':
            target=tlabels.unsqueeze(1)
            target_onehot = torch.cuda.FloatTensor(target.size(0), 10)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, target, 1)
            target=target.squeeze()
            tlabels = Variable(target_onehot)
               
          net.eval()
          
          toutputs = net(Variable(timages.cuda()))
          test_prec1, test_prec5 = accuracy(toutputs.data, target, (1,5))
          
          avg_test_prec1 += test_prec1
          avg_test_prec5 += test_prec5

          avg_test_loss += criterion(toutputs, tlabels).data.cpu()
      
      #print(test_total)
      '''
      avg_test_loss = avg_test_loss/test_total
      avg_test_prec1 = (100 * avg_test_prec1).float()/test_dataloader._size
      avg_test_prec5 = (100 * avg_test_prec5).float()/test_dataloader._size
      
      avg_train_loss = avg_train_loss/train_total
      avg_train_prec1 = (100 * avg_train_prec1)/train_dataloader._size
      avg_train_prec5 = (100 * avg_train_prec5)/train_dataloader._size
      
      '''
      avg_test_loss = avg_test_loss/(len(test_dataloader) if opt.dataset != 'ImageNet' else (test_dataloader._size / opt.test_batch_size))
      avg_test_prec1 = (100 * avg_test_prec1).float()/(len(test_dataloader.dataset) if opt.dataset != 'ImageNet' else test_dataloader._size)
      avg_test_prec5 = (100 * avg_test_prec5).float()/(len(test_dataloader.dataset) if opt.dataset != 'ImageNet' else test_dataloader._size)
      
      avg_train_loss = avg_train_loss/(len(train_dataloader) if opt.dataset != 'ImageNet' else train_dataloader._size / opt.train_batch_size)
      avg_train_prec1 = (100 * avg_train_prec1)/(len(train_dataloader.dataset) if opt.dataset != 'ImageNet' else train_dataloader._size)
      avg_train_prec5 = (100 * avg_train_prec5)/(len(train_dataloader.dataset) if opt.dataset != 'ImageNet' else train_dataloader._size)
     
      
      torch.cuda.empty_cache()
      
      print('After epoch %d, train_prec1: %.4f, test_prec1: %.4f, train_prec5: %.4f, test_prec5: %.4f, train loss: %4f, test loss: %4f signs: %.2f' % (epochs + 1, avg_train_prec1, avg_test_prec1, avg_train_prec5, avg_test_prec5, avg_train_loss, avg_test_loss, signs))
      
      running_train_loss.append(avg_train_loss.data.cpu().item())
      running_test_loss.append(avg_test_loss.data.cpu().item())
      
      running_train_prec1.append(avg_train_prec1.data.cpu().item())
      running_train_prec5.append(avg_train_prec5.data.cpu().item())
      
      running_test_prec1.append(avg_test_prec1.data.cpu().item())
      running_test_prec5.append(avg_test_prec5.data.cpu().item())
      
      running_weight_mean.append(weight_mean.data.cpu().item())
      running_best_prec1.append(best_accuracy)
      
      changes.append(signs.data.cpu().item())
      
      best_accuracy = checkpoints(optimizer, opt, net, epochs, changes, running_train_loss, running_test_loss, running_train_prec1, running_train_prec5, running_test_prec1, running_test_prec5, running_weight_mean, running_best_prec1)
      
      epochs += 1
      
      
      dataframe = pd.DataFrame({'epochs':n, 'changes': changes, 'running_train_loss':running_train_loss,'running_test_loss':running_test_loss, 'running_train_prec1':running_train_prec1, 'running_train_prec5':running_train_prec5, 'running_test_prec1':running_test_prec1, 'running_test_prec5':running_test_prec5})
      dataframe.to_csv("Logs/{}_hyper_{}_STE_{}_WB_{}_AB_{}_{}.csv".format(name, opt.hyper_accumulation, opt.ste, opt.weight_binarize, opt.full_binarize, opt.note),index=False) 
      
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize,  opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Loss',          {'train_loss':avg_train_loss, 'test_loss':avg_test_loss}, epochs)
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Accuracy',      {'train_prec1':avg_train_prec1, 'train_prec5':avg_train_prec5, 'test_prec1':avg_test_prec1, 'test_prec5':avg_test_prec5}, epochs)
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Best_Accuracy', {'best_accuracy':best_accuracy}, epochs) 
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Weight_L2',     {'L2':weight_L2}, epochs)
      
      torch.cuda.empty_cache()
      
      #if opt.dataset == 'ImageNet':
      #train_dataloader.reset()
      #test_dataloader.reset()
      
      
    writer.close()
    print('Finished Training')
    

          





if __name__ == "__main__":
   
   #######################################################################################
   parser = argparse.ArgumentParser(description='PyTorch BH-Net Training')
   parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
   parser.add_argument('--train_batch_size', type = int, default = 128, help = 'batch_size of train_dataloader')
   parser.add_argument('--test_batch_size', type = int, default = 128, help = 'batch_size of test_dataloader')
   parser.add_argument('--num_workers', type = int, default = 4, help = "dataloader workers")
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
   parser.add_argument('--retrain', action = 'store_true', help = 'training from scratch')
   parser.add_argument('--temperoture', type = float, default = 1, help = 'depth of hypernetwork')
   parser.add_argument('--note', type = str, default = 'Original', help = 'notes used for experiments classification')
   opt = parser.parse_args()
   
   #######################################################################################
   os.environ["CUDA_VISIABLE_DEVICE"] = "0,1,2,3"
   
   cuda_device = [0, 1, 2, 3]
   torch.manual_seed(1)
   
   torch.backends.cudnn.benchmark = True
   
   inflate = 4
   
   current_epoch = 0
   
   if opt.dataset == 'CIFAR10' or opt.dataset == 'MNIST':
      num_classes = 10
      
   elif opt.dataset == 'CIFAR100':
      num_classes = 100
      
   elif opt.dataset == 'ImageNet':
      num_classes = 1000
         
   
   
   
   if not opt.multi_gpu:
      torch.cuda.set_device(opt.set_cuda_device)
      net = get_model(opt, num_classes, inflate).cuda()
      name = net.name
      #print(net)
      input_transform = net.input_transforms
      optim_params = net.optim_params
      name = net.name
      max_epochs = net.optim_params['Max_Epochs']
      writer = get_writer(opt, net)
  
      train_dataloader, test_dataloader = get_data(opt, net)
   
      optimizer, lr_scheduler, criterion = get_optimizer(opt, net)
      
      adjust_architecture(net)
      net.cuda()
   else:
      net = get_model(opt, num_classes, inflate).cuda()
      name = net.name
      
      input_transform = net.input_transforms
      optim_params = net.optim_params
      name = net.name
      max_epochs = net.optim_params['Max_Epochs']
      writer = get_writer(opt, net)
  
      train_dataloader, test_dataloader = get_data(opt, net)
   
      optimizer, lr_scheduler, criterion = get_optimizer(opt, net)
      if opt.model == 'ResNet18':
        net.load_state_dict(models.resnet18(pretrained = True).state_dict())
        adjust_architecture(net)
      elif opt.model == 'ResNet34':
        net.load_state_dict(models.resnet34(pretrained = True).state_dict())
      	adjust_architecture(net)
      elif opt.model == 'ResNet50':
        net.load_state_dict(models.resnet50(pretrained = True).state_dict())
      net = nn.DataParallel(net.cuda(), cuda_device)
      
      
   if opt.reuse:
      #del net
      state = torch.load('checkpoints/{}/{}/BH_{}_{}_Current.zip'.format(opt.dataset, name, name, opt.note))
      net.load_state_dict(state['net'])
      
      if state['optimizer'] is not None:
        
        optimizer.load_state_dict(state['optimizer'])
        
   else:
      state = None
         
   criterion = criterion.cuda()    
   
   
   train(net, train_dataloader, test_dataloader, optimizer, lr_scheduler,  criterion, writer, opt, name, optim_params)
    

