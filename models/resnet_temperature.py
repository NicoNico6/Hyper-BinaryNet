import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from hyper_modules import *
from groupnorm import GroupNorm2d
#from .modules import nn as NN

#__all__ = ['ResNet','resnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def init_model(model):
    for m in model.modules():
      #if isinstance(m, nn.Conv2d):
      #  nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
      #  if hasattr(m.bias, 'data'):
      #    nn.init.constant_(m.bias, 0)
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
       
def BinaryHyperConv1x1(
        in_chs = 16, 
        out_chs = 16, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        dilation = 1,
        groups = 1,
        bias = False):
        
        return BConv2d(in_chs, out_chs, kernel_size, stride, padding, dilation, groups, bias)       
                
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        
        self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, dilation = 1, padding = 1)
        #self.bn0 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.ReLU1 = nn.ReLU()
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, dilation = dilation, padding = dilation)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.ReLU2 = nn.ReLU()
        #self.bn3 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    def forward(self, x):
        
        identity = x
        if self.downsample is not None:
          identity = self.ReLU1(identity)
          if self.stride == 2:
            identity = F.avg_pool2d(identity, kernel_size = 2, stride = 2)
          identity = self.downsample(identity)
        #elif identity.size(1) == 64:
        #  identity = ((identity))
       
        out = self.conv1(x)
        middle = self.bn1(out)
        #print(middle.size())
        middle = middle + identity
        out = self.conv2((middle))
        out = self.bn2(out)
        
        out = out+self.bn3(self.ReLU2(middle))
        
        return (out)

'''                    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        self.downsample = downsample
        self.stride = stride
          
    def forward(self, x):
        
        identity = x
        if self.downsample is not None:
          identity = self.downsample(F.avg_pool2d(F.relu(identity), kernel_size = 2, stride = 2))
        #if self.stride == 2:
        # x = F.avg_pool2d(x, kernel_size = 2, stride = 2)
        out = self.conv1(x)
        middle = self.bn1(out)
        middle += identity
        out = self.conv2((middle))
        out = self.bn2(out)
        
        out = out+middle
        
        return (out)
'''
          
class Bottleneck(nn.Module):
    expansion = 4
   
    def __init__(self, in_chs, out_chs, stride = 1, downsample = None, dilation = None, base_width = 64):
        super(Bottleneck, self).__init__()
        width = int(out_chs * (base_width) // 64)
        self.conv1 = nn.Conv2d(in_chs, width, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(width, affine = True, momentum = 0.5, track_running_stats = True) 
        self.conv2 = BConv2d(in_chs = width, out_chs = width, stride = stride)
        self.bn2 = nn.BatchNorm2d(width, affine = True, momentum = 0.5, track_running_stats = True)
        self.conv3 = nn.Conv2d(width, out_chs*self.expansion, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(out_chs*self.expansion, affine = True, momentum = 0.5, track_running_stats = True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, input):
        residual = input
        
        out = self.conv1(input)
        out_middle = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
          residual = F.relu(residual)
          if self.stride == 2:
            residual = F.avg_pool2d(residual, kernel_size = 2, stride = 2)
          residual = self.downsample(residual) 
        out = out + residual
        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=10, inflate = 4, block=BasicBlock, depth=[2, 2, 2, 2], full_binary = False, z_dim = 18, multi_gpu = False, opt = None):
        super(ResNet, self).__init__()
        self.dilation = 1
        self.dilation_replace_stride = [False, False, False, False]
        self.inflate = inflate
        self.in_chs = 16*self.inflate
        if opt.dataset != 'ImageNet':
          self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias = False),
                            #nn.BatchNorm2d(16*self.inflate),
                            )
          
          self.maxpool = lambda x:x
        else:
          self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=7, stride=2, padding=3, bias = False)
          '''
          self.conv1 = nn.Sequential(
                            nn.Conv2d(3, 16*self.inflate, kernel_size=3, stride=2, padding=1, bias = False),
                            nn.BatchNorm2d(16*self.inflate),
                            nn.ReLU(),
                            nn.Conv2d(16*self.inflate, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias = False),
                            nn.BatchNorm2d(16*self.inflate),
                            nn.ReLU(),
                            nn.Conv2d(16*self.inflate, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias = False),
                            nn.BatchNorm2d(16*self.inflate),
                            )
          '''                 
          self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        #self.maxpool = lambda x:x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.layer1 = self._make_layer(block, 16*self.inflate, depth[0], self.dilation_replace_stride[0], 1)
        self.layer2 = self._make_layer(block, 32*self.inflate, depth[1], self.dilation_replace_stride[1], 2)
        self.layer3 = self._make_layer(block, 64*self.inflate, depth[2], self.dilation_replace_stride[2], 2)
        self.layer4 = self._make_layer(block, 128*self.inflate, depth[3], self.dilation_replace_stride[3], 2)
        self.ReLU = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128 * self.inflate * block.expansion, num_classes)
                
        #init_model(self)
        self.name = 'resnet'
        
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
        
    def _make_layer(self, block, out_chs, blocks, dilation = False, stride=1):
        downsample = None
        layers = []
        previous_dilation = self.dilation
        if dilation:
          self.dilation *= stride
          stride=1
        if stride != 1 or self.in_chs != (out_chs * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_chs, out_chs*block.expansion, 1, stride = 1, bias = False),
                #nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
                nn.BatchNorm2d(out_chs*block.expansion, affine = True, momentum = 0.5, track_running_stats = True),
                )
            
        layers.append(block(self.in_chs, out_chs, stride, downsample, dilation = previous_dilation))
        
        self.in_chs = out_chs*block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_chs, out_chs, dilation = self.dilation))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(self.ReLU(x))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x        




class resnet(ResNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, inflate = 4, block=BasicBlock, depth=[2, 2, 2, 2], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet, self).__init__(num_classes=num_classes, inflate = inflate, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)

        self.name = 'resnet'
        

                 
class resnet18(ResNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes = 10, inflate = 4, block = BasicBlock, depth = [2, 2, 2, 2], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet18, self).__init__(num_classes=num_classes, inflate = inflate, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)

        self.name = 'resnet18'

        
class resnet20(ResNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, inflate = 1, block=BasicBlock, depth=[3, 3, 3, 0], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet20, self).__init__(num_classes=num_classes, inflate = 1, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)

        self.name = 'resnet20'

        
class resnet32(ResNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, inflate = 1, block=BasicBlock, depth=[5, 5, 5, 0], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet32, self).__init__(num_classes=num_classes, inflate = 1, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)

        self.name = 'resnet20'
        
        
class resnet34(ResNet):
    """Constructs a resnet34 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    
    """
    def __init__(self, num_classes=10, inflate = 4, block = BasicBlock, depth = [3, 4, 6, 3], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet34, self).__init__(num_classes=num_classes, inflate = inflate, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)
        
        self.name = 'resnet34'


class resnet44(ResNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, inflate = 1, block=BasicBlock, depth=[7, 7, 7, 0], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet44, self).__init__(num_classes=num_classes, inflate = 1, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)

        self.name = 'resnet20'
                
                
class resnet50(ResNet):
    """Constructs a resnet50 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    
    """
    def __init__(self, num_classes=10, inflate = 4, z_dim = 9, block = Bottleneck, depth = [3, 4, 6, 3], multi_gpu = False, opt = None):
        super(resnet50, self).__init__(num_classes=num_classes, inflate = inflate, block=block, depth=depth, opt = opt, multi_gpu = multi_gpu)
        
        self.name = 'resnet50'


class resnet56(ResNet):
    """Constructs a resnet18 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    """
    def __init__(self, num_classes=10, inflate = 1, block=BasicBlock, depth=[5, 5, 5, 0], z_dim = 9, multi_gpu = False, opt = None):
        super(resnet56, self).__init__(num_classes=num_classes, inflate = 1, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)

        self.name = 'resnet20'

        
class resnet101(ResNet):
    """Constructs a resnet101 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    
    """
    def __init__(self, num_classes=10, inflate = 4, z_dim = 9, block = Bottleneck, depth = [3, 4, 23, 3], multi_gpu = False, opt = None):
        super(resnet101, self).__init__(num_classes=num_classes, inflate = inflate, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)
      
        self.name = 'resnet101'
        

        

class resnet152(ResNet):
    """Constructs a resnet152 model with Binarized weight and activation.  
    
    Args:
      num_classes(int): an int number used to identify the num of classes, default 10
      inflate(int): an int number used to control the width of a singel convolution layer, default 4(4*16)
    
    """
    def __init__(self, num_classes=10, inflate = 4, z_dim = 9, block = Bottleneck, depth = [3, 8, 36, 3], multi_gpu = False, opt = None):
        super(resnet152, self).__init__(num_classes=num_classes, inflate = inflate, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)
        
        self.name = 'resnet152'
        
        
        
        

if __name__ == "__main__":
   resnet = ResNet()
   resnet18 = resnet18().cuda()
   resnet34 = resnet34().cuda()
   resnet50 = resnet50().cuda()
   resnet101 = resnet101().cuda()
   input = torch.randn(1, 3, 32, 32).cuda()
   output = resnet101(input)
   print(output.size())