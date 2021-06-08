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
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
        if hasattr(m.bias, 'data'):
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        if hasattr(m.bias, 'data'):
          nn.init.constant_(m.bias, 0)
      
      elif isinstance(m, nn.BatchNorm2d):
        if hasattr(m.weight, 'data'):
          nn.init.constant_(m.weight, 1)
        if hasattr(m.bias, 'data'):
          nn.init.constant_(m.bias, 0)
      
          
class Sin(nn.Module):
    def __init__(self, T =2, N = 1):
        super(Sin, self).__init__()
        self.T = T
        self.N = N
        self.w = 2*math.pi/T
        #self.T = nn.Parameter(torch.ones(1)*T)
        #self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, input):
        output = 0
        for n in range(1, self.N, 1):
          output+=torch.sin((2*n-1)*self.w*input)/(2*n-1)
        return output*4/math.pi
        
                                                                
                                                               
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
                
            
class BasicBlock(nn.Module):
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, activation_binarize = False, drop = False, weight_binarize = True):
        super(BasicBlock, self).__init__()
        if stride == 1:
          self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, activation_binarize = activation_binarize, weight_binarize = weight_binarize)
        else:
          self.conv1 = nn.Sequential(
                       nn.AvgPool2d(kernel_size =2, padding = 0, stride = 2),
                       BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = 1, activation_binarize = activation_binarize, weight_binarize = weight_binarize),
                       #nn.PReLU(),              
                       #nn.MaxPool2d(kernel_size =2, padding = 0, stride = 2),
                       )
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, activation_binarize = activation_binarize, weight_binarize = weight_binarize)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        
        self.downsample = downsample
        
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
          
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_chs, out_chs, stride = 1, downsample = None, base_width = 64, dilation = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, z_dim = z_dim, binarize = full_binarize, ste = ste)
        #self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True) 
        if not (activation_binarize and ste == 'ELU'):
          self.selu1 = nn.ELU(alpha = (1e1/(1e1 - 1.0))) if ste == 'clipped_elu' else nn.Hardtanh()
        else:
          self.selu1 = nn.ReLU()
        self.conv2 = BinaryHyperConv1x1(in_chs = out_chs, out_chs = out_chs, stride = stride, z_dim = z_dim, binarize = full_binarize, ste = ste)
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True) 
        if not (activation_binarize and ste == 'ELU'):
          self.selu2 = nn.ELU(alpha = (1e1/(1e1 - 1.0))) if ste == 'clipped_elu' else nn.Hardtanh()
        else:
          self.selu2 = nn.ReLU()
        
        self.conv3 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, z_dim = z_dim, binarize = full_binarize, ste = ste)
        #self.conv3 = nn.Conv2d(out_chs, out_chs, kernel_size = 1)
        self.bn3 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True)
        if not (activation_binarize and ste == 'ELU'):
          self.selu3 = nn.ELU(alpha = (1e1/(1e1 - 1.0))) if ste == 'clipped_elu' else nn.Hardtanh()
        else:
          self.selu3 = nn.ReLU()
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, input):
        residual = input
        
        out = self.conv1(input)
        out = self.bn1(out)
        #out = self.selu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.selu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out = out + residual
        out = self.selu3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=10, inflate = 4, block=BasicBlock, depth=[2, 2, 2, 2], full_binary = False, z_dim = 18, multi_gpu = False, opt = None):
        super(ResNet, self).__init__()
        
        expansion = 1
        
        self.inflate = inflate
        self.in_chs = 16*self.inflate
        self.depth = depth
        if opt.dataset != 'ImageNet':
          if not full_binary:
            self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias = False)
          else:
            self.conv1 = BinaryConv3x3(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, weight_binarize = False, activation_binarize = False)   
          self.maxpool = lambda x:x
            
        else:
          self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=7, stride=2, padding=3, bias = False)
          self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
           
        self.bn1 = nn.BatchNorm2d(16*self.inflate, affine = True, momentum = 0.5, track_running_stats = True)
        
        
        self.activation = (nn.Hardtanh() if opt.dataset != 'ImageNet' else nn.PReLU())
        self.bn2 = nn.BatchNorm1d(128*self.inflate)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.LogSoftmax = nn.LogSoftmax(dim = -1)
        self.layer1 = self._make_layer(block, 16*self.inflate, depth[0], 1, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.weight_binarize, opt.dataset)
        self.layer2 = self._make_layer(block, 32*self.inflate, depth[1], 2, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.weight_binarize, opt.dataset)
        self.layer3 = self._make_layer(block, 64*self.inflate, depth[2], 2, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.weight_binarize, opt.dataset)
        self.layer4 = self._make_layer(block, 128*self.inflate, depth[3], 2, opt.full_binarize, opt.skip_activation_binarize, opt.skip_weight_binarize, opt.skip_kernel_size, opt.weight_binarize, opt.dataset)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        if not full_binary:
         self.linear = nn.Linear((64 if depth[3]==0 else 128)*self.inflate, num_classes)
        else:
         self.linear = BinarizeHyperLinear(128*self.inflate, num_classes, weight_binarize =  False)
                
        init_model(self)
        self.name = 'resnet'
        
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
                               'Weight_decay': 0,
                               },
                               
                   'CIFAR100': {'Init_lr': 0.1,
                                'Weight_momentum':0.9,
                                'Weight_decay': 1e-4,
                              },
                  
                  'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.1},
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
        
    def _make_layer(self, block, out_chs, blocks, stride=1, full_binarize = True, skip_activation_binarize=True, skip_weight_binarize = True, skip_kernel = 3, weight_binarize = True, dataset = None):
        downsample = None
        layers = []
        
        if blocks>=1:
          if stride != 1 or self.in_chs != out_chs:
            downsample = nn.Sequential(
                nn.ReLU() if dataset != 'ImageNet' else nn.PReLU(),
                nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
                nn.Conv2d(self.in_chs, out_chs, 1, stride = 1, padding= 0, bias = False),
                
                #nn.PReLU(),
                nn.BatchNorm2d(out_chs, affine = True, momentum = 0.5, track_running_stats = True),
                )
            
          layers.append(block(self.in_chs, out_chs, stride, downsample, activation_binarize = full_binarize, weight_binarize = weight_binarize))
        
        self.in_chs = out_chs
        
        for i in range(1, blocks):
            
            layers.append(block(self.in_chs, out_chs, activation_binarize = full_binarize, weight_binarize = weight_binarize))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        #for m in self.modules():
        #  if isinstance(m, nn.BatchNorm2d):
        #    if hasattr(m.weight, 'data'):
        #      m.weight.data.clamp_(min=0.01)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.activation(x)
        x = self.avgpool(x)
        
        x = x.contiguous().view(x.size(0), -1)
        x = self.bn2(x)
        x = F.hardtanh(x)
        x = self.linear(x)
        x = self.bn3(x)
        x = self.LogSoftmax(x)
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
        super(resnet50, self).__init__(num_classes=num_classes, inflate = inflate, block=BasicBlock, depth=depth, opt = opt, multi_gpu = multi_gpu)
        
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