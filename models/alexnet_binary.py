import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from .modules import *

__all__ = ['alexnet']

class AlexNet_Binary(nn.Module):
  
    def __init__(self, num_classes = 10, ratio = 1):
        super(AlexNet_Binary, self).__init__()
        self.name = 'alexnet'
        
        self.optim_params = {
           'Adam': {'CIFAR10': {'Init_lr': 5e-3,
                                'Betas': (0.5, 0.9),
                                'Weight_decay': 1e-4,
                              },
                              
                    'CIFAR100': {'Init_lr': 1e-2,
                                'Betas': (0.9, 0.95),
                                'Weight_decay': 0,
                              },
                              
                    'MultiStepLR': {'step': [150, 200, 250, 300, 330, 360, 390, 420, 450, 480, 510, 530], 'ratio': 0.5},
                    },
                              
           'SGD': {'CIFAR10': {'Init_lr': 0.1,
                               'Weight_momentum':0.9,
                               'Weight_decay': 0,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.9,
                                'Weight_decay': 0,
                              },
                  
                  'MultiStepLR': {'step': [150, 250, 350, 450], 'ratio': 0.1},
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
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ])},
              
              
        
        }
        
        self.features = nn.Sequential(
            BinarizeHyperConv2d(3, int(64*ratio), 11, 4, 2, identity = False),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(int(64*ratio), affine = True),
          	selu(),
          	
            BinarizeHyperConv2d(int(64*ratio), int(192*ratio), 5, 1, 2, identity = False),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(int(192*ratio), affine = True),
            selu(),
            
            BinarizeHyperConv2d(int(192*ratio), int(384*ratio), 3, 1, 1, identity = False),
            nn.BatchNorm2d(int(384*ratio), affine = True),
            selu(),
            
            BinarizeHyperConv2d(int(384*ratio), int(256*ratio), 3, 1, 1, identity = False),
            nn.BatchNorm2d(int(256*ratio), affine = True),
            selu(),
            
            BinarizeHyperConv2d(int(256*ratio), 256, 3, 1, 1, identity = False),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(256, affine = True),
            selu(),
            
        )
      
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            BinarizeHyperLinear(256*6*6, 4096),
            nn.BatchNorm1d(4096),
            selu(),
          
            #nn.Dropout(),
            BinarizeHyperLinear(4096, 4096),
            nn.BatchNorm1d(4096),
            selu(),
            
            BinarizeHyperLinear(4096, num_classes),
            nn.BatchNorm1d(num_classes),
          
            nn.LogSoftmax()
        )
      
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)

        return x
      
def alexnet(**kwargs):
    """Constructs a Alexnet with Binarized weight and activation  
    
    Args:
      num_classes(int): a number used to identify the num of classes, default 10
      ratio(int): a number used to control size of a singel convolution layer, default 1
      
    """
    model = AlexNet_Binary(**kwargs)
    
    return model
    