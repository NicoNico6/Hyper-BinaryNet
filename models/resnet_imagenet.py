import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from hyper_modules import *
from resnet_binary import *


class BinaryHyperConv3x3(nn.Module):
    def __init__(self, 
        in_chs = 16, 
        out_chs = 16, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        bias = True, 
        z_dim = 9, 
        BinarizeHyper = BinarizedHypernetwork_Parrallel,
        identity = True,
        ste = 'clipped_elu',
        activation_binarize = True,
        weight_binarize = True,
        hyper_accumulation = True,
        depth = 1):
        super(BinaryHyperConv3x3, self).__init__()
        
        if hyper_accumulation:
          self.BinaryConv3x3 = BinarizeHyperConv2d(in_chs, out_chs, kernel_size, stride, padding, bias, z_dim, BinarizeHyper, identity, activation_binarize, ste, weight_binarize, depth = depth)
        
        else:
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
        z_dim = 9, 
        BinarizeHyper = BinarizedHypernetwork_Parrallel,
        identity = True,
        ste = 'clipped_elu',
        activation_binarize = True,
        weight_binarize = True,
        hyper_accumulation = True,
        depth = 1):
        super(BinaryHyperConv3x3, self).__init__()
        
        if hyper_accumulation:
          self.BinaryConv1x1 = BinarizeHyperConv2d(in_chs, out_chs, kernel_size, stride, padding, bias, z_dim, BinarizeHyper, identity, activation_binarize, ste, weight_binarize, depth = depth)
        
        else:
          self.BinaryConv1x1 = BinarizeConv2d(in_chs, out_chs, kernel_size, stride, padding, bias, weight_binarize, activation_binarize)
          
    def forward(self, input):
        output = self.BinaryConv1x1(input)
        return output
                
            
class BasicBlock(nn.Module):
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, activation_binarize = False, do_bntan=True, drop = False, z_dim = 9, ste = 'clipped_elu', weight_binarize = True, hyper_accumulation = True, depth = 1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth)
        '''
        if stride == 1:
          self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth)
        else:
          self.conv1 = nn.Sequential(
                              nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
                              BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = 1, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth)
                      )
        '''
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5)
        if not (not activation_binarize and ste == 'ELU'):
          self.selu1 = nn.ELU(alpha = (1e1/(1e1 - 1.0))) if ste == 'clipped_elu' else nn.Hardtanh()
        else:
          self.selu1 = nn.ELU()

        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, z_dim = z_dim, ste = ste, activation_binarize = activation_binarize, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5)
        if not (not activation_binarize and ste == 'ELU'):
          self.selu2 = nn.ELU(alpha = (1e1/(1e1 - 1.0))) if ste == 'clipped_elu' else nn.Hardtanh()
        else:
          self.selu2 = nn.ELU()
        
        
        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride
          
    def forward(self, input):
        residual = input
        if self.downsample is not None:
          residual = self.downsample(residual)
        
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.selu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        
        out = self.selu2(out)
        
        return out
        
class ResNet_ImageNet(nn.Module):

    def __init__(self, num_classes=10, inflate = 4, block=BasicBlock, depth=[2, 2, 2, 2], full_binary = False, z_dim = 18, multi_gpu = False, opt = None):
        super(ResNet_ImageNet, self).__init__()
        
        self.inflate = inflate
        self.in_chs = 16*self.inflate
        self.depth = depth
        expandsion = 4
        
        self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn1 = nn.BatchNorm2d(16*self.inflate, affine = True, momentum = 0.5)
        if not ((not opt.full_binarize) and opt.ste == 'ELU'):
          self.selu1 = nn.ELU(alpha = (1e1/(1e1 - 1.0))) if opt.ste == 'clipped_elu' else nn.Hardtanh()
        else:
          self.selu1 = nn.ELU()
          
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
          
        self.layer1 = self._make_layer(block, 16*self.inflate, depth[0], 1, opt.z_dim, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.ste, opt.weight_binarize, opt.hyper_accumulation, opt.depth)
        self.layer2 = self._make_layer(block, 32*self.inflate, depth[1], 2, opt.z_dim, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.ste, opt.weight_binarize, opt.hyper_accumulation, opt.depth)
        self.layer3 = self._make_layer(block, 64*self.inflate, depth[2], 2, opt.z_dim, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.ste, opt.weight_binarize, opt.hyper_accumulation, opt.depth)
        self.layer4 = self._make_layer(block, 128*self.inflate, depth[3], 2, opt.z_dim, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.ste, opt.weight_binarize, opt.hyper_accumulation, opt.depth)
        
        self.avgpool = nn.AvgPool2d(7)
        
        self.linear = nn.Linear(128*self.inflate, num_classes)
        
        self.name = 'resnet_imagenet'
        
        self.optim_params = {
           'Adam': {'CIFAR10': {'Init_lr': 5e-3,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 1e-4,
                              },
                              
                    'CIFAR100': {'Init_lr': 5e-3,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 1e-4,
                              },
                              
                    'ImageNet': {'Init_lr': 1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 1e-4,
                              },
                                        
                    'MultiStepLR': {'step': [30, 60, 80, 90], 'ratio': 0.1}
                    },
                              
           'SGD': {'CIFAR10': {'Init_lr': 0.1,
                               'Weight_momentum':0.5,
                               'Weight_decay': 1e-4,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.5,
                                'Weight_decay': 0,
                              },
                  
                  'MultiStepLR': {'step': [30, 50, 100, 150, 200, 240], 'ratio': 0.1},
                   },            
                               
           'BN_Momentum_Init': 0.5,
           'Criterion': nn.CrossEntropyLoss().cuda(),
           'Max_Epochs': 100,
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
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
              ]),
            
              'eval': transforms.Compose([
                transforms.Resize(256),
								transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
              ])},
              
              
        
        }
        
    def _make_layer(self, block, out_chs, blocks, stride=1, z_dim = 9, full_binarize = True, skip_activation_binarize=True, skip_weight_binarize = True, skip_kernel = 3, ste = 'clipped_elu', weight_binarize = True, hyper_accumulation = True, depth = 1):
        downsample = None
        if stride != 1 or self.in_chs != out_chs:
            downsample = nn.Sequential(
                BinaryHyperConv3x3(self.in_chs, out_chs, kernel_size = skip_kernel, stride =2, padding = (skip_kernel - 1)/2, z_dim = z_dim, ste = ste, activation_binarize = skip_activation_binarize, weight_binarize = skip_weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth),
                nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5),
                
                
                )
            
        layers = []
        layers.append(block(self.in_chs, out_chs, stride, downsample, activation_binarize = full_binarize, z_dim = z_dim, ste = ste, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth))
        self.in_chs = out_chs
        for i in range(1, blocks):
            
            layers.append(block(self.in_chs, out_chs, activation_binarize = full_binarize, z_dim = z_dim, ste = ste, weight_binarize = weight_binarize, hyper_accumulation = hyper_accumulation, depth = depth))
        
        return nn.Sequential(*layers)

    def forward(self, x):
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu1(x)
        x = self.maxpool(x)
        
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)

        return x        