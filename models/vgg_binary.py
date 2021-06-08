import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from hyper_modules import *

__all__ = ['VGG', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']

def init_model(model):
    for m in model.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias,0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


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
        
                    
class VGG(nn.Module):
    
    def __init__(self, features, num_classes = 10, opt = None):
        super(VGG, self).__init__()
        
        self.name = 'vgg'
        
        
        self.optim_params = {
           'Adam': {'CIFAR10': {'Init_lr':1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 0,
                              },
                              
                    'CIFAR100': {'Init_lr':1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 0,
                              },
                              
                              
                    'ImageNet': {'Init_lr':1e-2,
                                'Betas': (0.5, 0.99),
                                'Weight_decay': 0,
                              },
                                        
                    'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.5} if not opt.dataset == 'ImageNet'  else {'step': [30, 60, 90, 120, 150, 180], 'ratio': 0.2}
                    #'MultiStepLR': {'step': [100, 150, 200, 250], 'ratio': 0.1} if not opt.dataset == 'ImageNet'  else {'step': [30, 60, 90, 120, 150, 180], 'ratio': 0.2}  
                    },
                              
           'SGD': {'CIFAR10': {'Init_lr': 0.1,
                               'Weight_momentum':0.9,
                               'Weight_decay': 1e-4,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.9,
                                'Weight_decay': 1e-4,
                              },
                  
                  'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.1},
                   },            
                               
           'BN_Momentum_Init': 0.5,
           'Criterion': nn.CrossEntropyLoss() if opt.criterion == 'CrossEntropy' else HingeLoss(),
           'Max_Epochs': 500 if not opt.multi_gpu else 150,
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
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ]),
            
              'eval': transforms.Compose([
                transforms.Resize(224),
                #transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ])},
              
              
        
        }
        
        self.features = features
        if opt.dataset == 'ImageNet':
          self.avgpool = nn.AdaptiveAvgPool2d((7,7))
          self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),     
          )
        
        else:
          self.avgpool = nn.AdaptiveAvgPool2d((1,1))
          self.classifier = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),     
          )        
        init_model(self)
        
    def forward(self, input):
      
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
        

def _make_layer(cfg, batch_norm, opt):
    
    layers = []
    in_channels = 3
    identity = True
    
    for v in cfg:
      
      if v == 'M':
        layers += [nn.AvgPool2d(kernel_size = 2, stride = 2)] if not opt.full_binarize else [nn.MaxPool2d(kernel_size = 2, stride = 2)]
          
      else:
        
        if in_channels == 3:
          
          conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, stride = 1, padding = 1)
        
        else:
          
          conv2d = BinaryHyperConv3x3(in_channels, v, kernel_size = 3, stride = 1, padding = 1, weight_binarize = opt.weight_binarize, activation_binarize = opt.full_binarize)
        
        if batch_norm:
          
          layers += [conv2d, nn.BatchNorm2d(v, affine = True)]
          
        else:
          
          layers += [conv2d]
        
        in_channels = v
        
    return nn.Sequential(*layers) 
    


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_bn(opt, **kwargs):
    """Constructs a vgg11_bn model with Binarized weight and activation  
    
    Args:
      num_classes(int): a number used to identify the num of classes, default 10
      
    """
    model = VGG(features = _make_layer(cfg = cfg['A'], batch_norm = True, opt= opt), opt = opt, **kwargs)
    model.name = 'vgg11_bn'
    return model
    

def vgg13_bn(opt,**kwargs):
    """Constructs a vgg13_bn model with Binarized weight and activation  
    
    Args:
      num_classes(int): a number used to identify the num of classes, default 10
      
    """
    model = VGG(features = _make_layer(cfg = cfg['B'], batch_norm = True, opt= opt), opt = opt,**kwargs)
    model.name = 'vgg13_bn'
    return model
    

def vgg16_bn(opt, **kwargs):
    """Constructs a vgg16_bn model with Binarized weight and activation  
    
    Args:
      num_classes(int): a number used to identify the num of classes, default 10
      
    """
    model = VGG(features = _make_layer(cfg = cfg['D'], batch_norm = True, opt= opt), **kwargs)
    model.name = 'vgg16_bn'
    return model

    
def vgg19_bn(opt, **kwargs):
    """Constructs a vgg19_bn model with Binarized weight and activation  
    
    Args:
      num_classes(int): a number used to identify the num of classes, default 10
      
    """
    model = VGG(features = _make_layer(cfg = cfg['E'], batch_norm = True, opt= opt), **kwargs)
    model.name = 'vgg19_bn'
    return model
    
    
if __name__ == "__main__":
   vgg19 = vgg19_bn(z_dim = 9,).cuda()
   vgg19.train()
   input = torch.randn(2, 3, 32, 32).cuda()
   output = vgg19(input)
   print(output.size())
   print(vgg19.features[3].Weight)
    