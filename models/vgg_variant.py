import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from hyper_modules import *

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
        

class VGG_Small(nn.Module):
    def __init__(self, num_classes=10, opt = None):
        super(VGG_Small, self).__init__()
        
        self.block1 = nn.Sequential(
                      nn.Conv2d(3, 128, 3, 1, 1),
                      nn.BatchNorm2d(128, affine = True, momentum = 0.5),
                      nn.ELU(alpha = (1e1/(1e1 - 1.0))) if opt.ste == 'clipped_elu' else nn.ELU(),
                      BinaryHyperConv3x3(128, 128, kernel_size=3, stride=1, padding=1, z_dim = opt.z_dim, ste = opt.ste, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize, hyper_accumulation = opt.hyper_accumulation, depth = opt.depth),
                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                      nn.BatchNorm2d(128),
                      nn.ELU(alpha = (1e1/(1e1 - 1.0))) if opt.ste == 'clipped_elu' else nn.ELU(),
                )
        
        self.block2 = nn.Sequential(
                      BinaryHyperConv3x3(128, 256, kernel_size=3, stride=1, padding=1, z_dim = opt.z_dim, ste = opt.ste, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize, hyper_accumulation = opt.hyper_accumulation, depth = opt.depth),
                      nn.BatchNorm2d(256, affine = True, momentum = 0.5),
                      nn.ELU(alpha = (1e1/(1e1 - 1.0))) if opt.ste == 'clipped_elu' else nn.ELU(),
                      BinaryHyperConv3x3(256, 256, kernel_size=3, stride=1, padding=1, z_dim = opt.z_dim, ste = opt.ste, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize, hyper_accumulation = opt.hyper_accumulation, depth = opt.depth),
                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                      nn.BatchNorm2d(256),
                      nn.ELU(alpha = (1e1/(1e1 - 1.0))) if opt.ste == 'clipped_elu' else nn.ELU(),
                )
                
        self.block3 = nn.Sequential(
                      BinaryHyperConv3x3(256, 512, kernel_size=3, stride=1, padding=1, z_dim = opt.z_dim, ste = opt.ste, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize, hyper_accumulation = opt.hyper_accumulation, depth = opt.depth),
                      nn.BatchNorm2d(512, affine = True, momentum = 0.5),
                      nn.ELU(alpha = (1e1/(1e1 - 1.0))) if opt.ste == 'clipped_elu' else nn.ELU(),
                      BinaryHyperConv3x3(512, 512, kernel_size=3, stride=1, padding=1, z_dim = opt.z_dim, ste = opt.ste, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize, hyper_accumulation = opt.hyper_accumulation, depth = opt.depth),
                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                      nn.BatchNorm2d(512),
                      #nn.Dropout(p = 0.5),
                      nn.LeakyReLU()
                )
               
                
        self.block4 = nn.Sequential(
        							
                      nn.Linear(8192, 10),
                      #nn.BatchNorm1d(10),
                      
                )
                
                
    
    def forward(self, input):
        x1 = self.block1(input)
        
        x2 = self.block2(x1)
        
        x3 = self.block3(x2)
        
        x3 = x3.view(x3.size(0), -1)
        
        x4 = self.block4(x3)
        
        #x5 = self.block5(x3)
        
        return x4
        
        

class vgg_variant(VGG_Small):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, opt = None):
        super(vgg_variant, self).__init__(num_classes=num_classes, opt = opt)

        self.name = 'vgg_variant'
        
        self.optim_params = {
           'Adam': {'CIFAR10': {'Init_lr': 5e-3,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 5e-4,
                              },
                              
                    'CIFAR100': {'Init_lr': 5e-3,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 1e-4,
                              },
                              
                    'ImageNet': {'Init_lr': 5e-3,
                                'Betas': (0.5, 0.999),
                                'Weight_decay': 1e-4,
                              },
                                        
                    'MultiStepLR': {'step': [60, 120, 150, 180, 210, 240], 'ratio': 0.5},
                    },
                              
           'SGD': {'CIFAR10': {'Init_lr': 0.1,
                               'Weight_momentum':0.9,
                               'Weight_decay': 1e-4,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.9,
                                'Weight_decay': 0,
                              },
                  
                  'MultiStepLR': {'step': [30, 50, 100, 150, 200, 240], 'ratio': 0.1},
                   },            
                               
           'BN_Momentum_Init': 0.5,
           'Criterion': nn.CrossEntropyLoss() if opt.criterion == 'CrossEntropy' else HingeLoss(),
           'Max_Epochs': 250,
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
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.Resize(128),
                #transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ])},
              
              
        
        }