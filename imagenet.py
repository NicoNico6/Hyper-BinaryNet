import os
import math
import random
import pandas as pd

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
#from models.shufflenet_binary import *
#from models.mobilenet_binary import *

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
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=True):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, 12 + device_id, exec_pipelined=True)
        self.input = ops.FileReader(file_root=data_dir, num_shards = device_id+1, shard_id=device_id,  random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        self.iteration = torch.zeros(1).cuda()
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    #random_aspect_ratio=[0.8, 1.25],
                                                    #random_area=[0.1, 1.0],
                                                    #num_attempts=100
                                                    )
                                                    
            self.res = ops.Resize(resize_x=crop, resize_y=crop)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100)
                                                    
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
    
    #def iter_setup(self):
    #    gc.collect()
        #torch.cuda.empty_cache()
        
class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, 12 + device_id, exec_pipelined=True)
        self.iteration = torch.zeros(1).cuda()
        self.input = ops.FileReader(file_root=data_dir, num_shards = device_id+1, shard_id=device_id,  random_shuffle=False)
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
    
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
      testloader = torch.utils.data.DataLoader(testset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    else:
      traindir = '/media/opt48/data/gnh/dataset/ILSVRC2012/train'
      valdir = '/media/opt48/data/gnh/dataset/ILSVRC2012/val'
    
      crop_size = 224
      val_size = 256
      
      pipe = HybridTrainPipe(batch_size=opt.train_batch_size, num_threads=opt.num_workers, device_id=int(opt.set_cuda_device), data_dir=traindir, crop=crop_size, dali_cpu=opt.dali_cpu)
      pipe.build()
      trainloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1))
      #trainloader = DALIClassificationIterator(pipe, size=1000)

      pipe = HybridValPipe(batch_size=opt.test_batch_size, num_threads=opt.num_workers, device_id=int(opt.set_cuda_device), data_dir=valdir, crop=crop_size, size=val_size)
      pipe.build()
      testloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1))
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
      net = vgg11_bn(z_dim = opt.z_dim, num_classes = num_classes)
      
    elif opt.model == 'vgg13_bn':
      net = vgg13_bn(z_dim = opt.z_dim, num_classes = num_classes)
      
    elif opt.model == 'vgg16_bn':
      net = vgg16_bn(z_dim = opt.z_dim, num_classes = num_classes)
      
    elif opt.model == 'vgg19_bn':
      net = vgg19_bn(z_dim = opt.z_dim, num_classes = num_classes)
      
    elif opt.model == 'vgg_small':
      net = vgg_small(num_classes = num_classes, opt = opt)
      
    elif opt.model == 'vgg_variant':
      net = vgg_variant(num_classes = num_classes, opt = opt)   
    
    elif opt.model == 'AlexNet':
      net = alexnet(num_classes = num_classes)
    '''
    elif opt.model == 'ShuffleNet':
       net = shufflenet(num_classes = num_classes, opt = opt)
    
    elif opt.model == 'MobileNet':
       net = mobilenet(num_classes = num_classes, opt = opt)
    '''    
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
                        nesterov  = True
                        )
                       
    else:
      raise ValueError('Only support Adam based optimization now ! Please identify coresponding optim_params !')
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=optim_params[opt.optimizer]['MultiStepLR']['step'], gamma=optim_params[opt.optimizer]['MultiStepLR']['ratio'])
    criterion = optim_params['Criterion']
        
    return optimizer, lr_scheduler,  criterion


def get_weight(net, opt):
    weight_l2 = 0.0
    weight_mean = 0.0
    num = 0.0
    size = 0.0
  
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
          
                    
def checkpoints(optimizer, opt, net, epochs, running_train_loss, running_test_loss, running_train_prec1, running_train_prec5, running_test_prec1, running_test_prec5, running_weight_mean, running_best_prec1):
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
        'net': net.module.cpu().state_dict() if opt.multi_gpu else net.cpu().state_dict(),
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
      }
      
    if opt.multi_gpu:
        nn.DataParallel(net, [0, 1, 2, 3])
    net = net.cuda()
      
    torch.save(state, 'checkpoints/{}/{}/BH_{}_{}_Current.zip'.format(opt.dataset, name, name, opt.z_dim))  
    if running_test_prec1[-1] > running_best_prec1[-1]:
      
      running_best_prec1[-1] = running_test_prec1[-1]
      
      torch.save(state, 'checkpoints/{}/{}/BH_{}_{}_Best.pth.zip'.format(opt.dataset, name, name, opt.z_dim))
    
    print("Saving model.....")
    
    return running_best_prec1[-1]
              
def accuracy(output, target, topk = (1,)):
    maxk = max(topk)
    batch_size = output.size(0)
    
    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    #print(pred.size())
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
      correct_k = correct[:k].contiguous().view(-1).float().sum()
      res.append(correct_k)
    
    return res
    

def statistic(net):
    changes = 0
    size = 0
    for module in net.modules():
      if isinstance(module, BinarizeHyperConv2d):
        changes += (module.Weight.data.cpu().clone().sign() != module.init.cpu().clone().sign()).sum()
        module.init = module.Weight.clone()
        size += module.Weight.view(-1).size(0)  
      elif isinstance(module, BinarizeConv2d):
        changes += (module.Weight.data.cpu().clone().sign() != module.init.data.cpu().clone().sign()).sum()
        module.init = module.Weight.clone()
        size += module.Weight.view(-1).size(0)  
    return changes

def adjust_temperoture(Init_T, epochs, iteration):
    T = 1*((Init_T *0.95**(epochs)))
    #T = (torch.ones(1)*Init_T  * (math.cos(2*math.pi*float(epochs)/600) + 1)/2).cuda()
    Temperature_W = 0
    Temperature_A = 0
    Num_W = 0
    Num_A = 0
    for i, module in enumerate(net.module.modules() if opt.multi_gpu else net.modules()):
        if isinstance(module, (Softsign_W)):
          Num_W += 1
          module.Temperature = T
          Temperature_W += module.Temperature
        
        if isinstance(module, (Softsign_A)):
          Num_A += 1
          module.Temperature = T
          Temperature_A += module.Temperature
    
    
    return Temperature_W/Num_W, Temperature_A/Num_A, Temperature_A/Num_A

                                  
def train(net, train_dataloader, test_dataloader, optimizer, lr_scheduler, criterion, writer, opt, name, optim_params, max_epochs, state):
    
    best_accuracy = 0.
    print_freq = 10
    total_iter = 0.
    momentum = 0.5
    epochs = 0
    init_temperoture = opt.temperoture
    temperoture = init_temperoture
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
      
      for k in range(int(current_epochs)):
        lr_scheduler.step()
     
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
    
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize,  opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Loss',          {'train_loss':0, 'test_loss':0}, 0)
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Accuracy',      {'train_prec1':0, 'train_prec5':0, 'test_prec1':0, 'test_prec5':0}, 0)
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Best_Accuracy', {'best_accuracy':0}, 0) 
    writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Weight_L2',     {'L2':0}, 0)
     
    for n in range(int(current_epochs), max_epochs):
      
      running_loss = 0.0
      
      train_total = 0.0
      avg_grad = 0.0
      train_correct = 0
      
      avg_train_loss = 0.0
      avg_train_prec1 = 0.0
      avg_train_prec5 = 0.0
      
      avg_test_loss = 0.0
      avg_test_prec1 = 0.0
      avg_test_prec5 = 0.0
      
      loss_weight = torch.zeros(1)
      weight_mean = torch.zeros(1)
      
      lr_scheduler.step()
      signs = statistic(net)
      if not opt.dataset == "ImageNet":
        
        momentum = adjust_momentum(net = net, epochs = n, opt = opt)
        
      for i, data in enumerate(train_dataloader):
        
        #pdb.set_trace() 
        avg_temperoture_w, temperoture, avg_temperoture_a = adjust_temperoture(Init_T = init_temperoture, epochs = (n*5000+i)//1000, iteration = (n*5000+i)//1000, opt = opt)
        #torch.cuda.empty_cache()
        if opt.dataset == "ImageNet":
          inputs = Variable((data[0]["data"]))
          labels = Variable((data[0]["label"].squeeze().long()))
        
        else:
          
          inputs, labels = data
          inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        #print(inputs)
        #inputs, labels = Variable(inputs.cuda(non_blocking=True)), Variable(labels.cuda(non_blocking=True))
        
        target = labels
        
        if opt.criterion == 'HingleLoss':
          target=labels.unsqueeze(1)
          target_onehot = torch.cuda.FloatTensor(target.size(0), 10)
          target_onehot.fill_(-1)
          target_onehot.scatter_(1, target, 1)
          target=target.squeeze()
          labels = Variable(target_onehot)   
        
        
        
        net.train()
        
        #pdb.set_trace() 
        # Forward + Backward
        outputs = net(inputs)
        #pdb.set_trace() 
        loss_sum = criterion(outputs, labels)
        #loss_weight, weight_mean = get_weight(net, opt)
        #print(outputs.cpu())
        optimizer.zero_grad()
        loss_sum.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        
        #torch.cuda.synchronize()
        
        #Top1 and Top2 Accuracy
        prec1, prec5 = accuracy(outputs.data, target, (1,5))
        running_loss += loss_sum.mean()
         
        avg_train_loss += loss_sum.sum()
        avg_train_prec1 += prec1
        avg_train_prec5 += prec5
        
       # del prec1, prec5, loss_sum, outputs, data, inputs, labels
        #gc.collect()
        if i % print_freq == 0 and i != 0:
            print("[Epoch %d, Total Iterations %6d] Loss: %.4f  L2_weight: %.4f Weight_Mean: %.4f Signs: %.d Lr: %.8f Momentum: %.3f weight: %.3f, best test accuracy: %.4f" % (epochs + 1, total_iter + 1, running_loss/print_freq, loss_weight.data.item(), weight_mean.data.item(), signs.data.item(), optimizer.param_groups[0]['lr'], momentum, 1, best_accuracy))
            running_loss = 0.0
            #torch.cuda.empty_cache()
            #gc.collect()
            
        train_total += 1.0
        total_iter += 1.0
        
      test_total = 0
      test_correct = 0
      
      with torch.no_grad():
        for tdata in test_dataloader:
          test_total += 1
          
          if opt.dataset == "ImageNet":
            timages = Variable((tdata[0]["data"]))
            tlabels = Variable((tdata[0]["label"].squeeze().long().cuda()))
        
          else:
          
            timages, tlabels = tdata
            timages, tlabels = Variable(timages.cuda()), Variable(tlabels.cuda())
          
          if opt.criterion == 'HingleLoss':
            target=tlabels.unsqueeze(1)
            target_onehot = torch.cuda.FloatTensor(target.size(0), 10)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, target, 1)
            target=target.squeeze()
            tlabels = Variable(target_onehot)
               
          net.eval()
          
          toutputs = net((timages))
          test_prec1, test_prec5 = accuracy(toutputs.data, tlabels, (1,5))
          
          avg_test_prec1 += test_prec1
          avg_test_prec5 += test_prec5

          avg_test_loss += criterion(toutputs, tlabels)
      print("train size :{}".format(train_total*opt.train_batch_size))
      print("test size :{}".format(test_total*opt.test_batch_size))
      avg_test_loss = avg_test_loss/test_total
      avg_test_prec1 = (100 * avg_test_prec1).float()/test_total/opt.test_batch_size
      avg_test_prec5 = (100 * avg_test_prec5).float()/test_total/opt.test_batch_size
      
      avg_train_loss = avg_train_loss/train_total
      avg_train_prec1 = (100 * avg_train_prec1)/train_total/opt.train_batch_size
      avg_train_prec5 = (100 * avg_train_prec5)/train_total/opt.train_batch_size
    
      
      torch.cuda.empty_cache()
      
      print('After epoch %d, train_prec1: %.4f, test_prec1: %.4f, train_prec5: %.4f, test_prec5: %.4f, train loss: %4f, test loss: %4f' % (epochs + 1, avg_train_prec1, avg_test_prec1, avg_train_prec5, avg_test_prec5, avg_train_loss, avg_test_loss))
      
      running_train_loss.append(avg_train_loss)
      running_test_loss.append(avg_test_loss)
      
      running_train_prec1.append(avg_train_prec1)
      running_train_prec5.append(avg_train_prec5)
      
      running_test_prec1.append(avg_test_prec1)
      running_test_prec5.append(avg_test_prec5)
      
      running_weight_mean.append(weight_mean)
      running_best_prec1.append(best_accuracy)
      
      best_accuracy = checkpoints(optimizer, opt, net, epochs, running_train_loss, running_test_loss, running_train_prec1, running_train_prec5, running_test_prec1, running_test_prec5, running_weight_mean, running_best_prec1)
      
      epochs += 1
      
      #lr_scheduler.step(avg_test_loss) 
      dataframe = pd.DataFrame({'epochs':n, 'running_train_loss':running_train_loss,'running_test_loss':running_test_loss, 'running_train_prec1':running_train_prec1, 'running_train_prec5':running_train_prec5, 'running_test_prec1':running_test_prec1, 'running_test_prec5':running_test_prec5})
      dataframe.to_csv("Logs/{}_hyper_{}_STE_{}_WB_{}_AB_{}.csv".format(name, opt.hyper_accumulation, opt.ste, opt.weight_binarize, opt.full_binarize),index=False) 
     
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize,  opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Loss',          {'train_loss':avg_train_loss, 'test_loss':avg_test_loss}, epochs)
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Accuracy',      {'train_prec1':avg_train_prec1, 'train_prec5':avg_train_prec5, 'test_prec1':avg_test_prec1, 'test_prec5':avg_test_prec5}, epochs)
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Best_Accuracy', {'best_accuracy':best_accuracy}, epochs) 
      writer.add_scalars(name + '_{}/hyper_{}_depth_{}_ste_{}/w_{}_A_{}/{}/skip_W_{}_A_{}_k_{}/loss_{}'.format(opt.dataset, opt.hyper_accumulation, opt.depth, opt.ste,  opt.weight_binarize, opt.full_binarize, opt.z_dim, opt.skip_weight_binarize, opt.skip_activation_binarize, opt.skip_kernel_size, opt.criterion) + '/Weight_L2',     {'L2':loss_weight}, epochs)
      
      torch.cuda.empty_cache()
      #gc.collect()
      
      if opt.dataset == "ImageNet":
        train_dataloader.reset()
        test_dataloader.reset()
      
      
    writer.close()
    print('Finished Training')
    
 
          





if __name__ == "__main__":
   
   #######################################################################################
   parser = argparse.ArgumentParser(description='PyTorch BH-Net Training')
   parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
   parser.add_argument('--train_batch_size', type = int, default = 128, help = 'batch_size of train_dataloader')
   parser.add_argument('--test_batch_size', type = int, default = 128, help = 'batch_size of test_dataloader')
   parser.add_argument('--num_workers', type = int, default = 12, help = "dataloader workers")
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
   parser.add_argument('--temperoture', type = float, default = 1, help = 'depth of hypernetwork')
   opt = parser.parse_args()
   
   #######################################################################################
   os.environ["CUDA_VISIABLE_DEVICE"] = "0,1,2,3"
   
   cuda_device = [0, 1, 2, 3]
   torch.manual_seed(1)
   
   torch.cuda.set_device(opt.set_cuda_device)
   
   #cudnn.benchmark = True
   #torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = True
   #torch.backends.cudnn.enabled = False
   inflate = 4
   
   current_epoch = 0
   
   if opt.dataset == 'CIFAR10' or opt.dataset == 'MNIST':
      num_classes = 10
      
   elif opt.dataset == 'CIFAR100':
      num_classes = 100
      
   elif opt.dataset == 'ImageNet':
      num_classes = 1000
         
   net = get_model(opt, num_classes, inflate).cuda()
   name = net.name
      
   input_transform = net.input_transforms
   optim_params = net.optim_params
   name = net.name
   max_epochs = net.optim_params['Max_Epochs']
   writer = get_writer(opt, net)
  
   train_dataloader, test_dataloader = get_data(opt, net)
   
   optimizer, lr_scheduler, criterion = get_optimizer(opt, net)
   
   #net = torchvision.models.resnet18()
   
   if opt.reuse:
      #del net
      state = torch.load('checkpoints/{}/{}/BH_{}_{}_Current.zip'.format(opt.dataset, name, name, opt.z_dim))
      net.load_state_dict(state['net'])
      if state['optimizer'] is not None:
        #state['optimizer'].values = state['optimizer'].values.cpu()
        #print(state['optimizer']['state'])
        optimizer.load_state_dict(state['optimizer'])
        #optimizer.state_dict()['state'] = state['optimizer']['state']
        #optimizer.state_dict()['param'] = state['optimizer']['param']
        #print(state['optimizer'])
   else:
      state = None   
      
   if not opt.multi_gpu:
      torch.cuda.set_device(opt.set_cuda_device)
   else:
      net = nn.DataParallel(net, cuda_device)
      
   net = net.cuda()
   train(net, train_dataloader, test_dataloader, optimizer, lr_scheduler, criterion, writer, opt, name, optim_params, max_epochs, state)
    


   
    


