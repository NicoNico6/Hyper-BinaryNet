import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from hyper_modules import *
from groupnorm import GroupNorm2d

def init_model(model):
    for m in model.modules():
      if isinstance(m, nn.Linear):
        if hasattr(m.bias, 'data'):
          nn.init.constant_(m.bias, 0)
      
      elif isinstance(m, nn.BatchNorm2d):
        if hasattr(m.weight, 'data'):
          nn.init.constant_(m.weight, 1)
        if hasattr(m.bias, 'data'):
          nn.init.constant_(m.bias, 0)
          
      if isinstance(m, BasicBlock) or isinstance(m, Bottleneck):
        nn.init.constant_(m.bn2.weight, 0)
            
def BinaryHyperConv3x3(
        in_chs = 16, 
        out_chs = 16, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        dilation = 1,
        groups = 1,
        bias = False):
          
        return BConv2d(in_chs, out_chs, kernel_size, stride, padding, dilation, groups, bias)            
        

class VGG_Small(nn.Module):
    def __init__(self, num_classes=10, opt = None):
        super(VGG_Small, self).__init__()
        
        self.block1 = nn.Sequential(
                      nn.Conv2d(3, 128, 3, 1, 1),
                      nn.BatchNorm2d(128, affine = True, momentum = 0.5),
                      BinaryHyperConv3x3(in_chs = 128, out_chs = 128, stride = 1, dilation = 1, padding = 1),
                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                      nn.BatchNorm2d(128),
                )
        
        self.block2 = nn.Sequential(
                      BinaryHyperConv3x3(128, 256, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(256, affine = True, momentum = 0.5),
                    
                      BinaryHyperConv3x3(128, 256, kernel_size=3, stride=1, padding=1),
                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                      nn.BatchNorm2d(256),                      
                )
                
        self.block3 = nn.Sequential(
                      BinaryHyperConv3x3(256, 512, kernel_size=3, stride=1, padding=1),
                      nn.BatchNorm2d(512, affine = True, momentum = 0.5),
                      
                      BinaryHyperConv3x3(512, 512, kernel_size=3, stride=1, padding=1),
                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                      nn.BatchNorm2d(512),  
                )
              
        self.block4 = nn.Sequential(
                      BLinear(8192, 1024, bias = False, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize),
                      nn.BatchNorm1d(1024),
                      
                      BLinear(1024, 1024, bias, activation_binarize = opt.full_binarize, weight_binarize = opt.weight_binarize),
                      nn.BatchNorm1d(1024),
                      nn.ReLU(),
                )
                
        self.block5 = nn.Sequential(
                      nn.Linear(1024, num_classes),
                      nn.BatchNorm1d(num_classes),                      
                )
                
                
    
    def forward(self, input):
        x1 = self.block1(input)
        
        x2 = self.block2(x1)
        
        x3 = self.block3(x2)
        
        x3 = x3.contiguous().view(x3.size(0), -1)
        
        x4 = self.block4(x3)
        
        x5 = self.block5(x4)
        
        return x5
        
        

class vgg_small(VGG_Small):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, opt = None):
        super(vgg_small, self).__init__(num_classes=num_classes, opt = opt)

        self.name = 'vgg_small'
        
        self.optim_params = {
           'Adam': {'CIFAR10': {'Init_lr':1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay':0,
                              },
                              
                    'CIFAR100': {'Init_lr':1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 0,
                              },
                              
                              
                    'ImageNet': {'Init_lr':3e-4,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 0,
                              },
                                        
                    'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.5} if not opt.dataset == 'ImageNet'  else {'step': [30, 45, 60, 70, 80, 85, 90, 99], 'ratio': 0.5}
                    #'MultiStepLR': {'step': [100, 150, 200, 250], 'ratio': 0.1} if not opt.dataset == 'ImageNet'  else {'step': [30, 60, 90, 120, 150, 180], 'ratio': 0.2}  
                    },
                              
           'SGD': {'CIFAR10': {'Init_lr': 0.01,
                               'Weight_momentum':0.9,
                               'Weight_decay': 0,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.9,
                                'Weight_decay': 1e-4,
                              },
                              
                   'ImageNet': {'Init_lr':1e-2,
                                'Weight_momentum':0.9,
                                'Weight_decay': 0,
                              },
                  
                  'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.1} if not opt.dataset == 'ImageNet'  else {'step': [10, 20, 35, 60, 90, 120, 150, 180], 'ratio': 0.1}
                   },            
                               
           'BN_Momentum_Init': 0.5,
           'Criterion': nn.CrossEntropyLoss() if opt.criterion == 'CrossEntropy' else HingeLoss(),
           'Max_Epochs': 500 if not multi_gpu else 150,
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
                #transforms.Normalize((0., 0., 0.), (1, 1, 1))  
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
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)) 
              ])},
              
              
        
        }