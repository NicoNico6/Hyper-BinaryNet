import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from hyper_modules import *

import sys
import numpy as np

def init_model(model):
    for m in model.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
      	m.bias.data.zero_()
        
class BinaryHyperConv3x3(nn.Module):
    def __init__(self, 
        in_chs = 16, 
        out_chs = 16, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        bias = True, 
        activation_binarize = True,
        weight_binarize = True):
        super(BinaryHyperConv3x3, self).__init__()
        
        
        self.BinaryConv3x3 = BinarizeConv2d(in_chs, out_chs, kernel_size, stride, padding, bias, weight_binarize, activation_binarize)
          
          
    def forward(self, input):
        output = self.BinaryConv3x3(input)
        return output
        

class BinaryHyperConv1x1(nn.Module):
    def __init__(self, 
        in_chs = 16, 
        out_chs = 16, 
        kernel_size = 1, 
        stride = 1, 
        padding = 1, 
        bias = True, 
        activation_binarize = True,
        weight_binarize = True):
        super(BinaryHyperConv3x3, self).__init__()
        self.BinaryConv1x1 = BinarizeConv2d(in_chs, out_chs, kernel_size, stride, padding, bias, weight_binarize, activation_binarize)
          
    def forward(self, input):
        output = self.BinaryConv1x1(input)
        return output
                
            
class PABasicBlock(nn.Module):
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, activation_binarize = False, drop = False, weight_binarize = True):
        super(PABasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_chs, affine = True, momentum = 0.5, track_running_stats = True)
        if stride == 1:
          self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, activation_binarize = activation_binarize, weight_binarize = weight_binarize)
        else:
          self.conv1 = nn.Sequential(
                       nn.AvgPool2d(kernel_size = 2, padding = 0, stride = 2),
                       BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = 1, activation_binarize = activation_binarize, weight_binarize = weight_binarize),              
                       )
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, stride = 1, activation_binarize = activation_binarize, weight_binarize = weight_binarize)
        self.equalInOut = (in_chs == out_chs)
        self.downsample = downsample
        
        self.stride = stride
          
    def forward(self, input):
        if not self.equalInOut:
          input = self.bn1(input)
        else:
          out = self.bn1(input)
        residual = input
        if self.downsample is not None:
          residual = self.downsample(((residual)))
        
        out = self.bn2(self.conv1(out if self.equalInOut else input))
        out = (self.conv2(out))
        out = out.add(residual)
        
        return out       

        
class BasicBlock(nn.Module):
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, activation_binarize = False, drop = False, weight_binarize = True):
        super(BasicBlock, self).__init__()
        if stride == 1:
          self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth)
        else:
          self.conv1 = nn.Sequential(
                       nn.AvgPool2d(kernel_size = 3, padding = 1, stride = 2),
                       BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = 1, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth),              
                       )
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        
        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride
          
    def forward(self, input):
        
        residual = (input)
        if self.downsample is not None:
          residual = self.downsample(((residual)))
        
        out = self.conv1(input)
        out = (self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = out+residual
        
        return out
                
class Wide_ResNet(nn.Module):
    def __init__(self, num_classes = 10, block=PABasicBlock, depth = 22, inflate = 1, multi_gpu = False, opt = None):
        super(Wide_ResNet, self).__init__()
        self.in_chs = 16*inflate
        drop_rate = 0.3
        
        n = (depth - 4)/6
        print('| Wide-ResNet %d%d' % (n,inflate))
        
        nStages = [16*inflate, 16*inflate, 32*inflate, 64*inflate]
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = nStages[0], kernel_size = 3, stride = 1, padding = 1, bias = True)
        #self.conv1.weight.data.normal_(0, 2/math.sqrt(nStages[0]))
        
        self.layer1 = self._make_layer(block = block, out_chs = nStages[1], num_blocks = n, stride = 1, activation_binarize = opt.full_binarize, skip_activation_binarize = opt.skip_activation_binarize, weight_binarize = opt.weight_binarize, skip_weight_binarize = opt.skip_weight_binarize, drop = False)
        self.layer2 = self._make_layer(block = block, out_chs = nStages[2], num_blocks = n, stride = 2, activation_binarize = opt.full_binarize, skip_activation_binarize = opt.skip_activation_binarize, weight_binarize = opt.weight_binarize, skip_weight_binarize = opt.skip_weight_binarize, drop = False)
        self.layer3 = self._make_layer(block = block, out_chs = nStages[3], num_blocks = n, stride = 2, activation_binarize = opt.full_binarize, skip_activation_binarize = opt.skip_activation_binarize, weight_binarize = opt.weight_binarize, skip_weight_binarize = opt.skip_weight_binarize, drop = False)
        
        self.linear = nn.Linear(nStages[3], num_classes)
        #self.linear.weight.data.normal_(0,2/math.sqrt(num_classes))
       
        self.bn1 = nn.BatchNorm2d(nStages[3], affine = True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU(inplace = True)
        
        init_model(self)
        
        self.optim_params = {
           'Adam': {'CIFAR10': {'Init_lr': 1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay':0,
                              },
                              
                    'CIFAR100': {'Init_lr': 1e-2,
                                'Betas': (0.9, 0.999),
                                'Weight_decay':0,
                              },
                              
                    'MultiStepLR': {'step': [60, 120, 150, 180, 210, 240, 270], 'ratio': 0.5} if not opt.dataset == 'ImageNet'  else {'step': [20, 40, 60, 80, 100, 120, 150, 180], 'ratio': 0.2}
                    },
                              
           'SGD': {'CIFAR10': {'Init_lr': 0.1,
                               'Weight_momentum':0.9,
                               'Weight_decay': 1e-4,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.9,
                                'Weight_decay': 0,
                              },
                  
                  'MultiStepLR': {'step': [60, 120, 180, 240, 280], 'ratio': 0.2},
                   },            
                               
           'BN_Momentum_Init': 0.5,
           'Criterion': nn.CrossEntropyLoss(),
           'Max_Epochs': 550,
           'Loss_reweight': 1,
           'Max_grad':5,
           'Weight_L2':0,  
        }
        
        self.input_transforms = {
            'CIFAR10': {
              'train': transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
              ])},
            
            'CIFAR100':{
              'train': transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ])},
              
            'ImageNet':{
              'train': transforms.Compose([
                transforms.RandomCrop(224, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ])},
              
              
        
        }
        
    def _make_layer(self, block, out_chs, num_blocks, stride=1, activation_binarize = True, skip_activation_binarize=True, skip_weight_binarize = True, skip_kernel = 3, weight_binarize = True, drop = False):
        downsample = None
        layers = []
        
        if num_blocks>=1:
          if stride != 1 or self.in_chs != out_chs:
            downsample = nn.Sequential(
                nn.ReLU(),
                nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
                nn.Conv2d(self.in_chs, out_chs, 1, stride = 1, padding = 0),
                #BinaryHyperConv3x3(self.in_chs, out_chs, kernel_size = skip_kernel, stride =2, padding = (skip_kernel - 1)/2, activation_binarize = skip_activation_binarize, weight_binarize = skip_weight_binarize),
                )
            
          layers.append(block(self.in_chs, out_chs, stride, downsample, activation_binarize = activation_binarize, drop = drop, weight_binarize = weight_binarize))
        
        self.in_chs = out_chs
        
        for i in range(1, num_blocks):
            
            layers.append(block(self.in_chs, out_chs, activation_binarize = activation_binarize, weight_binarize = weight_binarize))
        
        return nn.Sequential(*layers)
        
    
    def forward(self, x):
        out = self.conv1(x)
        #out = self.selu1(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.drop(out)
        out = self.relu(self.bn1(out))
        
        #out = self.avg_pool(self.bn2(self.conv2(self.selu2(self.bn1(out))))).view(out.size(0), -1)
        
        out = (self.linear(self.avg_pool(out).view(out.size(0), -1)))
        return out


class wrn22(Wide_ResNet):

    def __init__(self, num_classes = 10, block=PABasicBlock, depth = 22, inflate = 1, multi_gpu = False, opt = None):
        super(wrn22, self).__init__(num_classes = num_classes, block = block, depth = depth, inflate = inflate, opt = opt, multi_gpu = multi_gpu)
        self.name = 'wrn22'         
        
class wrn40(Wide_ResNet):
    def __init__(self, num_classes = 10, block=PABasicBlock, depth = 40, inflate = 4, multi_gpu = False, opt = None):
        super(wrn40, self).__init__(num_classes = num_classes, block = block, depth = depth, inflate = inflate, opt = opt, multi_gpu = multi_gpu)
        self.name = 'wrn40'
        
        
        
              
if __name__ == "__main__":
   net = WRN22(10, BasicBlock, [3,3,3], 10, 18).cuda()
   y = net(torch.randn(1, 3, 32, 32).cuda())
   
   print(y.size())
      
