import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
from numpy.random import random

import gc
import math
import numpy as np

def weight_norm(weight):
    shape = weight.size()
    return weight * math.sqrt(2.0/(shape[1]*shape[2]*shape[3]))


def reweight(weight):
    shape = weight.size()
    
    return math.sqrt(2. / (shape[0] * shape[1] * shape[2] * shape[3])) * weight

    
def reweight_mean(weight):
    shape = weight.size()
    return torch.mean((abs(weight + 1e-8)).view(shape[0], -1), -1, keepdim = True).unsqueeze(-1).unsqueeze(-1) * weight


def polynomial(input):
    input_pos = F.relu(input)
    input_neg = -F.relu(-input)
    
    output_pos = (2 - input_pos)*input_pos
    output_neg = (2 + input_neg)*input_neg
    
    
    return output_pos+output_neg

        
class BinarizeF(Function):
    @staticmethod   
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()
        
    @staticmethod   
    def backward(ctx, grad_output):
        k = 2.8
        factor = (1 - math.sqrt(1 - 2/k))
        x = ctx.saved_tensors[0]
        abs_x = torch.abs(x)
        mask_0 = abs_x<=1
        
        mask_1 = abs_x>=factor
        mask_2 = abs_x<factor
        x_grad = k * ((1 - abs_x)).to(grad_output.dtype) * mask_2.to(grad_output.dtype) + (mask_1 * mask_0).to(grad_output.dtype)
        #x_grad = k * ((1 - abs_x)).to(grad_output.dtype) * (mask_1 * mask_0).to(grad_output.dtype) + mask_2.to(grad_output.dtype)
        
        return (grad_output *  x_grad)


class BinarizeWF(Function):
    @staticmethod   
    def forward(ctx, input):
        ctx.real_input = input
        return input.sign()
        
    @staticmethod   
    def backward(ctx, grad_output):
        
        x = ctx.real_input
        abs_x = torch.abs(x)
        mask_0 = (abs_x<=1)
        x_grad = mask_0.to(grad_output.dtype)
        
        del x, ctx.real_input, mask_0
        #gc.collect()
        return (grad_output * x_grad)



class StochasticBinarizeF(Function):
    @staticmethod   
    def forward(ctx, input):
        output = input
        output = output.add_(1).div_(2).add_(torch.rand(output.size()).cuda().add(-0.5)).clamp_(0, 1).floor_().mul_(2).add_(-1)
        
        return output
        
    @staticmethod   
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input


binarize = BinarizeF.apply
binarizew = BinarizeWF.apply
stbinarize = StochasticBinarizeF.apply

        
class Binarize(nn.Module):
    def __init__(self, channels = 16, max = 4):
        super(Binarize, self).__init__()
        self.channels = channels
        self.max = max
        #self.conv_1 = nn.Conv2d(channels, max * channels, 1, padding = 0, bias = True)
        #self.conv_1.weight.data[:max * channels // 2, :, :, :].fill_(2.0/(max * channels // 2))
        #self.conv_1.weight.data[max * channels // 2:, :, :, :].fill_(-2.0/(max * channels // 2))
        #self.conv_1.weight.data.fill_(1.0/(channels))
        #self.conv_1.bias.data.fill_(0)
        #print(self.conv_1.weight.data.size())     
        #self.conv_2 = nn.Conv2d(max * channels, channels,  1, padding = 0, bias = True)
        #self.PReLU = nn.ELU()
        #self.scale = Parameter(torch.ones(1))
        #self.bias = Parameter(torch.zeros(1))
        #self.scale = Parameter(torch.ones(1))
        #self.bias = Parameter(torch.zeros(1))
        #self.A = Parameter(torch.ones(1))
        
    def maxout(self, input, maxout):
        shape = input.size()
        assert (shape[1]%maxout == 0)
        reshaped = input.view(shape[0], shape[1]//maxout, maxout, shape[2], shape[3])
        maxout, _ = torch.max(reshaped, dim = 2)
        
        return maxout
        
    def mean(self, input, G):
        #mean = (input).view(input.size(0), input.size(1), -1).mean(dim = -1, keepdim = True).unsqueeze(-1).expand_as(input)
        N,C,H,W = input.size()
        input = (input).view(N, G, -1)
        mean = (torch.mean(input, dim = -1, keepdim = True)).expand_as(input).view(N,C,H,W)
        std = ((torch.std(input, dim = -1, keepdim = True)).expand_as(input) + 1e-8).view(N,C,H,W)
        sample = None
        
        return mean, std, sample
        
    def forward(self, input):
        #input = torch.sin(self.scale*input + self.bias)
        #replace = self.maxout(self.conv_1(input), self.max)
        #replace = self.conv_2(self.PReLU(self.conv_1(input)))
        #replace = self.PReLU(input)
        #input = self.A*torch.sin(self.scale*input + self.bias)
        input = ((input).clamp(-1,1))
        replace = (input)
        if self.training:
          #shape = replace.size()  
          #replace += torch.FloatTensor(shape[0], shape[1], 1, 1).normal_(0,1).cuda()
          Mask = input.sign()
          #Mask = replace.add_(1).div_(2).add_(torch.rand(output.size()).cuda().add(-0.5)).clamp_(0, 1).floor_().mul_(2).add_(-1)
          #Mask[Mask==-1] = 0
          #replace = (input - sample).clamp(-1,1)
          output = (Mask - replace).detach() + replace
        else:
          output = input.sign()
          #output[output==0] = 1
        return output
        

class Softsign_W(nn.Module):
      def __init__(self, in_channels, out_channels):
          super(Softsign_W, self).__init__()
          
          #self.ELU = nn.ELU(alpha = (1e1/(1e1 - 1.0)))
          self.Temperature = 1
          self.Adaptive_T = 1
          #self.scale_1 = Parameter(torch.ones(1))
          #self.scale_2 = Parameter(torch.ones(1))
          #self.bias = Parameter(torch.zeros(1))
          #self.A = Parameter(torch.ones(1))
          #self.Sin = Sin()
      def softsign(self, input, temperature):
          output = input.div(input.abs().add((temperature **2 + 1e-20)** 0.5))
          return output
      
      def softsign_mean(self, input, temperature):
          shape = input.size()
          output = input.div((input.abs()).div((input.abs().view(shape[0], shape[1], -1).mean(-1,keepdim = True).unsqueeze(-1))).add((temperature **2 + 1e-20)** 0.5))
          return output
          
      def hardtanh(self, input, temperature):
          return F.hardtanh(input.div((temperature)))
          #return F.hardtanh(input.sign())
      def hardtanh_mean(self, input, temperature):
          with torch.no_grad():
            shape = input.size()
            output = input.div(temperature).clamp(-1,1)
            mask = (output.abs()==1)
            mean = input.abs().view(shape[0], shape[1], -1).mean(-1,keepdim = True).unsqueeze(-1).expand_as(input)
            
            return torch.where(mask, mean*output, output)
          
      def tanh(self, input, temperature):
          return torch.tanh(input.div(temperature))
          
      def elu(self, input, temperature):
          return F.elu(input.div(temperature)).clamp(-1,1)
      
      def relu(self, input, temperature):
          
          return F.relu(input.div(temperature)).clamp(0,1) 
          
      def polynomial(self, input, temperature):
          input = input.div((temperature**2 + 1e-20)**0.5).clamp(-1,1)
          pos_output = F.relu(input)
          neg_output = -F.relu(-input)  
          
          return pos_output.mul(2).add(-pos_output.pow(2)) + neg_output.mul(2).add(neg_output.pow(2))
          
      def forward(self, input):
          shape = input.size()
          #input = F.relu(input)
          if self.training:
            #replace = self.hardtanh_mean(input, self.Adaptive_T)
            replace = F.hardtanh(input)
            #replace = self.hardtanh(input, self.Adaptive_T)
            softsign = (self.hardtanh(input, self.Temperature) - replace).detach()  + replace
            
          else:
            softsign = self.hardtanh(input, self.Temperature)
          '''
          mask = (softsign.abs()==1)
          mean = input.abs().view(shape[0], -1).mean(-1,keepdim = True).unsqueeze(-1).unsqueeze(-1)
          softsign = torch.where(mask, mean*softsign, softsign)
          '''
          return softsign
          
class Softsign_A(nn.Module):
      def __init__(self, in_channels, out_channels):
          super(Softsign_A, self).__init__()
          
          #self.ELU = nn.ELU(alpha = (1e1/(1e1 - 1.0)))
          self.Temperature = 1
          self.Adaptive_T = 1
          #self.binarize = False
          #self.Sin = Sin()
      
      def softsign(self, input, temperature):
          output = input.div(input.abs().add((temperature **2 + 1e-20)** 0.5))
          return output
      
      def softsign_mean(self, input, temperature):
          shape = input.size()
          output = input.div((input.abs()).div((input.abs().view(shape[0], shape[1], -1).mean(-1,keepdim = True).unsqueeze(-1))).add((temperature **2 + 1e-20)** 0.5))
          return output
          
      def hardtanh(self, input, temperature):
          return F.hardtanh(input.div((temperature)))
          #return F.hardtanh(input.sign())
      def hardtanh_mean(self, input, temperature):
          with torch.no_grad():
            shape = input.size()
            output = input.div(temperature).clamp(-1,1)
            mask = (output.abs()==1)
            mean = input.abs().view(shape[0], shape[1], -1).mean(-1,keepdim = True).unsqueeze(-1).expand_as(input)
            
            return torch.where(mask, mean*output, output)
          
      def tanh(self, input, temperature):
          return torch.tanh(input.div(temperature))
          
      def elu(self, input, temperature):
          return F.elu(input.div(temperature)).clamp(-1,1)
      
      def relu(self, input, temperature):
          
          return F.relu(input.div(temperature)).clamp(0,1) 
      
      def polynomial(self, input, temperature):
          input = input.div((temperature**2 + 1e-20)**0.5).clamp(-1,1)
          pos_output = F.relu(input)
          neg_output = -F.relu(-input)  
          
          return pos_output.mul(2).add(-pos_output.pow(2)) + neg_output.mul(2).add(neg_output.pow(2))
          
      def forward(self, input):
          #shape = input.size()
          #input = F.relu(input)
          if self.training:
            #replace = self.hardtanh_mean(input, self.Adaptive_T)
            replace = F.hardtanh(input)
            #replace = self.hardtanh(input, self.Adaptive_T)
            softsign = (self.hardtanh(input, self.Temperature) - replace).detach()  + replace
            
          else:
            softsign = self.hardtanh(input, self.Temperature)
          '''
          mask = (softsign.abs()==1)
          mean = input.abs().mean(1,keepdim = True)
          softsign = torch.where(mask, mean*softsign, softsign)
          '''
          return softsign
 
              
class BinarySELU(nn.Module):
      def __init__(self):
          super(BinarySELU, self).__init__()
          self.selu = nn.SELU()
          self.binary = BinarizeWF.apply
          
      def forward(self, input):
          output = self.selu(input)
          output = self.binary(output)
          
          return output
          

class BinaryELU(nn.Module):
      def __init__(self):
          super(BinaryELU, self).__init__()
          self.elu = nn.ELU(alpha = (1e1/(1e1 - 1.0)))
          self.binary = BinarizeWF.apply
          
      def forward(self, input):
          output = self.elu(input)
          output = self.binary(output)
          
          return output
          
                    
class HardTanh(nn.Module):
      def __init__(self, value):
          super(HardTanh, self).__init__()
          self.value = value
      def forward(self, input):
          return input.clamp_(-self.value, self.value)

          
class BinaryTanh(nn.Module):
      def __init__(self, mode = 'det'):
          super(BinaryTanh, self).__init__()
          self.mode = mode
          self.hardtanh = nn.Hardtanh()
          self.tanh = nn.Tanh()
                
      def polynomial(self, input):
          output_pos = input.clone()
          output_neg = input.clone()
          output_pos[input <= 0] = 0
          output_neg[input >= 0] = 0
          output_pos = 2*output_pos - output_pos**2
          output_neg = 2*output_neg + output_neg**2
          output = output_pos + output_neg
          output[output >= 1] = 1
          output[output <= -1] = -1
          
          return output
          
      def forward(self, input):
          shape = input.size()
          output = self.hardtanh(input)
          
          output = self.polynomial(output)
          mean = reweight_mean(input)
          if self.mode == 'det':
            output.data = mean * (binarize(output.data))
          else:
            output = stbinarize(output)
          return output 


class BinaryReLU(nn.Module):
      def __init__(self, mode = 'det'):
          super(BinaryTanh, self).__init__()
          self.mode = mode
          self.relu = nn.ReLU(inplace = True)
          
      def forward(self, input):
          shape = input.size()
          #mean = reweight_mean(input)
          output = self.relu(input)
          #output = input
          #weight = torch.mean(output.view(shape[0], -1), dim = -1, keepdim = True).unsqueeze(-1).unsqueeze(-1)
          if self.mode == 'det':
            output.data = reweight(binarize(output.data))
          else:
            output = stbinarize(output)
          return output 

                    
class BinarySigmoid(nn.Module):
      def __init__(self, mode = 'det'):
          super(BinaryTanh, self).__init__()
          self.mode = mode
          self.sigmoid = nn.Sigmoid()
          
      def forward(self, input):
          output = self.sigmoid(input)
          if self.mode == 'det':
            output = binarize(output)
          else:
            output = stbinarize(output)
          return output

class alpha_drop(nn.Module):
    
    def __init__(self, p = 0.05, alpha=-1.7580993408473766, fixedPointMean=0, fixedPointVar=1):
        super(alpha_drop, self).__init__()
        keep_prob = 1 - p
        self.a = np.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * pow(alpha-fixedPointMean,2) + fixedPointVar)))
        self.b = fixedPointMean - self.a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        self.alpha = alpha
        self.keep_prob = 1 - p
        self.drop_prob = p
    
    def forward(self, x):
        if self.keep_prob == 1 or not self.training:
            # print("testing mode, direct return")
            return x
        else:
            random_tensor  = self.keep_prob + torch.rand(x.size())
            
            binary_tensor = Variable(torch.floor(random_tensor))

            if torch.cuda.is_available():
                binary_tensor = binary_tensor.cuda()
            
            x = x.mul(binary_tensor)
            ret = x + self.alpha * (1-binary_tensor)
            ret.mul_(self.a).add_(self.b)
            return ret

class maxout(nn.Module):
    def __init__(self, num_layers = 2):
        super(maxout, self).__init__()
        self.num_layers = num_layers
        self.activation = nn.ModuleList([nn.PReLU(1)])
        self.activation.extend([nn.PReLU(1) for i in range(1, self.num_layers-1)])
        
    def forward(self, input):
        max_output = self.activation[0](input)
        for _, layer in enumerate(self.activation, start=1):
          max_output = torch.max(max_output, layer(input))
        
        return max_output 

class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha_1 = 1.6732632423543772848170429916717
        #self.scale_1 = 0.945444145
        self.scale_1 = 0.945444145
        #self.alpha = Parameter(torch.FloatTensor(1).fill_(1.6732632423543772848170429916717))
        #self.scale_2 = Parameter(torch.FloatTensor(1).fill_(1.0507009873554804934193349852946))
        
    def forward(self, x):
        temp1 = self.scale_1 * F.relu(x)
        temp2 = self.scale_1 * self.alpha_1 * (F.elu(-1*F.relu(-1*x)))  
        return (temp1 + temp2).clamp_(-1, 1)

class elu(nn.Module):
    def __init__(self, alpha):
        super(elu, self).__init__()
        self.alpha = Parameter(torch.ones(1).fill_(alpha))
        
    def forward(self, x):
        self.alpha.data = self.alpha.clamp((1e1/(1e1 - 1.0))*0.9, (1e1/(1e1 - 1.0))*1.1)
        x = F.relu(x) + self.alpha * (torch.exp(-1*F.relu(-1*x)) - 1)
        return x
        
class WeightNormlinear(nn.Module):
    def __init__(self, in_channels, out_channels, scale = False, bias = False, init_factor = 2, init_scale = 1, one_scale = False):
        super(WeightNormlinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyper_weight = Parameter(torch.Tensor(out_channels, in_channels))
        #self.dropout = nn.Dropout()
        self.init_scale = init_scale 
        if bias:
            self.bias = Parameter(torch.zeros(1, out_channels))
        else:
           self.register_parameter('bias', None)
        if scale:
           if one_scale:
              self.scale = Parameter(torch.Tensor(1, 1).fill_(init_scale))
           else:
              self.scale = Parameter(torch.Tensor(1, out_channels).fill_(init_scale))
        else:
            self.register_parameter('scale', None)
        self.reset_parameters(init_factor)


    def reset_parameters(self, factor):
        stdv =  1. * factor / math.sqrt(self.hyper_weight.size(1))
        self.hyper_weight.data.uniform_(-stdv, stdv)
        #self.hyper_weight.data.normal_(0, stdv)
        if self.bias is not None:
           self.bias.data.normal_(0, stdv)

    def weight_norm(self, weight):
        return weight.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
        
    def input_norm(self, input):
        return input.pow(2).sum().add(1e-8).sqrt()
        
    def norm_scale_bias(self, input):
        output = F.linear(input, self.hyper_weight)
        
        #output = output.div(self.weight_norm(self.hyper_weight).transpose(0,1).expand_as(output))
       
        if self.scale is not None:
          
           output = output.mul((self.scale))
        if self.bias is not None:
            output = output.add(self.bias.expand_as(output))
        return output

    def forward(self, input):
        return self.norm_scale_bias(input)
  

    def __repr__(self):
        return self.__class__.__name__ + '(' \
          + str(self.in_channels) + '->' \
          + str(self.out_channels) + ')'
          

class WeightNormConv(nn.Module):

    def __init__(self, in_channels, out_channels, scale = False, bias = False, init_factor = 2, init_scale = 1, one_scale = False):
        super(WeightNormConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hyper_weight = Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_channels))
        else:
           self.register_parameter('bias', None)
        if scale:
           if one_scale:
              self.scale = Parameter(torch.Tensor(1, 1).fill_(init_scale))
           else:
              #self.scale = Parameter(torch.Tensor(1, out_channels).fill_(init_scale))
              self.scale = Parameter(torch.Tensor(1, out_channels).normal_(0, 1))
        else:
            self.register_parameter('scale', None)
        self.reset_parameters(init_factor)


    def reset_parameters(self, factor):
        stdv =  1. * factor / math.sqrt(self.hyper_weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        self.hyper_weight.data.uniform_(-stdv, stdv)
        #self.hyper_weight.data.normal_(0, stdv)
        if self.bias is not None:
           self.bias.data.normal_(0, stdv)

    def weight_norm(self):
        return self.hyper_weight.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
        #return self.weight.pow(2).sum().add(1e-6).sqrt()
    def norm_scale_bias(self, input):
        output = F.linear(input, self.hyper_weight)
        #output = output.div(self.weight_norm())
        output = output.div(self.weight_norm().transpose(0,1).expand_as(output))
        if self.scale is not None:
           output = output.mul((self.scale))
        if self.bias is not None:
            output = output.add(self.bias.expand_as(output))
        return output

    def forward(self, input):
        return self.norm_scale_bias(input)
 

    def __repr__(self):
        return self.__class__.__name__ + '(' \
          + str(self.in_channels) + '->' \
          + str(self.out_channels) + ')'
          
class GroupNorm(nn.Module):

    def __init__(self, num_features, num_groups=32, eps=1e-5):

        super(GroupNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))

        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))

        self.num_groups = num_groups

        self.eps = eps



    def forward(self, x):

        N,C,H,W = x.size()

        G = self.num_groups

        assert C % G == 0



        x = x.view(N,G,-1)

        mean = x.mean(-1, keepdim=True)

        std = x.std(-1, keepdim=True) + 1e-8



        x = (x-mean) / std

        x = x.view(N,C,H,W)

        return x * self.weight + self.bias
        #return x
        
                                      
class BinarizedHypernetwork_Parrallel(nn.Module):
    """Static hypernetwork described in https://arxiv.org/pdf/1609.09106.pdf"""
    def __init__(self, 
        embed_vec_dim = 9, 
        in_channels = 1, 
        out_channels = 1, 
        kernel_size = 3,
        z_num = (1,1), 
        bias=True,
        binarize = True,
        reweight = False,
        ste = 'clipped_elu',
        depth = 1):
        super(BinarizedHypernetwork_Parrallel, self).__init__()
        
        self.embed_vec_dim_0 = embed_vec_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.binarize = binarize
        self.reweight = reweight
        
        self.hypernetwork = self.build_hypernetwork(depth, embed_vec_dim, kernel_size, ste)
        #self.Binarize_Function = Binarize()
        self.Binarize = Softsign_W() 
        #self.ELU = nn.ELU(alpha = (1e1/(1e1 - 1.0)))
        #self.hyper_weight = Parameter(torch.Tensor(kernel_size**2, embed_vec_dim))
        #stdv =  2. / math.sqrt(self.hyper_weight.size(1))
        #self.hyper_weight.data.uniform_(-stdv, stdv)
        
        #self.scale = Parameter(torch.Tensor(1, 1).fill_(1))
        #self.LRN = nn.LocalResponseNorm(3)
        '''
        self.hyper_weight_1 = Parameter(torch.Tensor(kernel_size**2, embed_vec_dim))
        self.hyper_weight_2 = Parameter(torch.Tensor(in_channels, in_channels))
        stdv =  2. / math.sqrt(self.hyper_weight_1.size(1))
        self.hyper_weight_1.data.uniform_(-stdv, stdv)
        self.hyper_weight_2.data.uniform_(-stdv, stdv)
        self.scale_1 = Parameter(torch.Tensor(1, 1).fill_(1))
        self.scale_2 = Parameter(torch.Tensor(1, 1).fill_(1))
        ''' 
            
    def build_hypernetwork(self, depth, width, kernel_size, ste):
        layers = []
        
        if depth == 1:
          layers.append(WeightNormlinear(in_channels = width, out_channels = kernel_size * kernel_size, scale = True, bias=False, init_scale = 1, one_scale = True))
        else:
          layers.append(WeightNormlinear(in_channels = width, out_channels = width, scale = True, bias=False, init_scale = 1, one_scale = True))
          
        if ste == 'satured':
          layers.append(nn.Hardtanh())
        #elif ste == 'clipped_elu':
        #  layers.append(nn.ELU(alpha = (1e1/(1e1 - 1.0))))
       
        for i in range(1, depth):
          layers.append(WeightNormlinear(in_channels = width, out_channels = kernel_size * kernel_size, scale = True, bias=False, init_scale = 1, one_scale = True))
             
          if ste == 'satured':
            layers.append(nn.Hardtanh())
          elif ste == 'clipped_elu':
            layers.append(nn.ELU(alpha = (1e1/(1e1 - 1.0))))
            
        return nn.Sequential(*layers)
        
    def mean(self, input):
        mean = ((input)).view(input.size(0), input.size(1), -1).mean(dim = -1, keepdim = True).unsqueeze(-1).expand_as(input)
        std = ((input)).view(input.size(0), input.size(1), -1).std(dim = -1, keepdim = True).unsqueeze(-1).expand_as(input)
        #max, _ = torch.max(input.view(input.size(0), input.size(1), -1), dim = -1, keepdim = True)
        #min, _ = torch.min(input.view(input.size(0), input.size(1), -1), dim = -1, keepdim = True)
        #max = max.unsqueeze(-1).expand_as(input)
        #min = min.unsqueeze(-1).expand_as(input)
        #print(max)
        #norm = (input - min)/(max - min)
        
        #mean = (input).view(input.size(0), -1).mean(dim = -1, keepdim = True).unsqueeze(-1).unsqueeze(-1).expand_as(input)
        #std = (input).view(input.size(0), -1).std(dim = -1, keepdim = True).unsqueeze(-1).unsqueeze(-1).expand_as(input)
        
        #mean = (input).view(input.size(0), input.size(1) -1).mean(dim = -1, keepdim = True).unsqueeze(-1).expand_as(input)
        #std = (input).view(input.size(0), input.size(1) -1).std(dim = -1, keepdim = True).unsqueeze(-1).expand_as(input)
        
        #std = (input).view(input.size(0), -1).std(dim = -1, keepdim = True).unsqueeze(-1).unsqueeze(-1).expand_as(input)
        #sample = mean + std * torch.randn(mean.size()).normal_(0,1).cuda()
        
        return (input - mean)/std
        #return norm
                    
    def reset_parameters(self):
        for module in self.modules():
          if isinstance(module, nn.Linear):
            stdv =  1./ math.sqrt(module.weight.size(1))
            module.weight.data.uniform_(-stdv, stdv)
            #module.weight.data.normal_(0, stdv)
            if module.bias is not None:
              module.bias.data.normal_(0, stdv)
               
    def forward(self, embed_vec):
       
        N_O, N_I, E = embed_vec.size()
        #self.sampler = torch.distributions.uniform.Normal(mean = torch.zeros_like(embed_vec), var = torch.ones_like(embed_vec), validate_args=None)
        #if self.training:
        embed_vec = embed_vec
        self.Weight = (self.hypernetwork(embed_vec))
        weight = (self.Weight.view(N_O, N_I, self.kernel_size, self.kernel_size))
        #self.weight = weight
        
        if self.binarize:
          weight = self.Binarize((weight))
        self.weight = (weight)   
        '''
        
        shape = self.hyper_weight.size()
        #Mask = torch.from_numpy(np.triu(np.ones((shape[0], shape[1])))).float().cuda()
        #hyper_weight = self.hyper_weight * Mask
        #hyper_weight += hyper_weight.transpose(0,1)
        #hyper_weight = self.hyper_weight
         
        #
        #print(hyper_weight.cpu().data)
        #hyper_weight = self.hyper_weight
        #hyper_weight =  hyper_weight * (torch.eye(shape[0], shape[1]) == 0).float().cuda()
        
        for i in range(1, 2):
          if i == 1:
            weight = ((F.linear((((embed_vec))), self.hyper_weight)))
            #weight =(self.mean(F.linear((((embed_vec))), hyper_weight)))
          else:
            weight = ((F.linear(((weight)), hyper_weight)))
          #hyper_weight = hyper_weight.transpose(0, 1)
        
        
        weight = weight.div(self.hyper_weight.pow(2).sum(1, keepdim = True).add(1e-8).sqrt().transpose(0,1).expand_as(weight)).mul(self.scale)
        weight = ((self.ELU(weight))) 
       # weight = self.LRN(weight.view(N_O, N_I, -1, 1).transpose(1,2)).transpose(1,2)
        weight = weight.view(N_O, N_I, self.kernel_size, self.kernel_size)
        
        #weight = F.linear(weight.transpose(1,2), self.hyper_weight_2)
        #weight = weight.div(self.hyper_weight_2.pow(2).sum(1, keepdim = True).add(1e-8).sqrt().transpose(0,1).expand_as(weight)).transpose(1,2)  
        #LayerNorm = nn.LayerNorm(weight.size()[1:], elementwise_affine=False)
        #weight = LayerNorm(weight)    
        #weight = self.ELU(weight)
        #weight = self.ELU(weight)

        #static_weight = weight.view(weight.size(0), weight.size(1), -1)
        #mean = torch.mean(static_weight, dim = -1, keepdim = True).unsqueeze(-1).expand_as(weight)
        #var = torch.var(static_weight, dim = -1, keepdim = True).unsqueeze(-1).expand_as(weight)
        #weight = ((weight - mean)/var)
        self.Weight = weight
        if self.binarize:
          #print(1)
          weight = self.Binarize((weight))  
        self.weight = (weight)
        
        
        hyper_weight_1 = self.hyper_weight_1
        hyper_weight_2 = self.hyper_weight_2
        
        weight_1 = self.Binarize_Function(F.linear(((embed_vec)), hyper_weight_1).div(hyper_weight_1.pow(2).sum(1, keepdim = True).add(1e-8).sqrt().transpose(0,1).expand_as(embed_vec)).mul(self.scale_1))  
        weight_2 = self.Binarize_Function(F.linear(((embed_vec)), hyper_weight_2).div(hyper_weight_2.pow(2).sum(1, keepdim = True).add(1e-8).sqrt().transpose(0,1).expand_as(embed_vec)).mul(self.scale_2))  
        weight = (weight_1*weight_2).view(N_O, N_I, self.kernel_size, self.kernel_size)
        
        self.Weight = weight
        self.weight = weight
        '''
        return  weight

class Sin(nn.Module):
    def __init__(self, T =10, N = 5):
        super(Sin, self).__init__()
        self.T = T
        self.N = N
        self.w = 2*math.pi/T
        #self.T = nn.Parameter(torch.ones(1)*T)
        #self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, input):
        output = torch.zeros_like(input)
        #for n in range(1, self.N, 1):
        # if n%2 != 0:
        #   output+=torch.sin((2*n-1)*self.w*(input))/(2*n-1)
        for n in range(1, self.N, 2):
          #if n%2 != 0:
          output+=torch.sin(n*input)/n
        #return output*4/math.pi - F.relu(-input).clamp(-1,1)
        return output*4/math.pi
                

class BinaryKernels(nn.Module):
    def __init__(self,out_chs, in_chs, height, width):
        super(BinaryKernels, self).__init__()
        self.out_chs = out_chs
        self.in_chs = in_chs
        self.height = height
        self.width = width
        self.weight = torch.bernoulli(torch.ones(out_chs, in_chs, height, width)*0.5).cuda()
        self.weight[self.weight == 0] = -1
        self.kernels = Variable(self.weight, requires_grad = True)
        self.grad = 0
        self.binary = 0
        self.t = 1
        
    def update(self):
        alpha = 0.1
        grad = self.kernels.grad
        '''
        grad_max, _ = torch.max(grad.view(self.out_chs, self.in_chs, -1), -1, keepdim = True)
        grad_max = grad_max.unsqueeze(-1).expand_as(grad)
        grad_min, _ = torch.min(grad.view(self.out_chs, self.in_chs, -1), -1, keepdim = True)
        grad_min = grad_min.unsqueeze(-1).expand_as(grad)
        grad_norm = ((grad - grad_min)/(grad_max - grad_min) * 2) - 1
        '''
        self.grad = (self.grad * (1 - alpha) + grad * alpha)/(1 - (alpha**self.t))
        grad_max, _ = torch.max(abs(self.grad).view(self.out_chs, self.in_chs, -1), -1, keepdim = True)
        grad_max = grad_max.unsqueeze(-1).expand_as(grad)
        
        #print(self.grad)
        self.t = self.t + 1
        #self.binary = (self.binary + grad.sign())
        ratio = abs(self.grad)/(grad_max + 1e-20)
        random = torch.bernoulli(ratio)
        binary_grad = (self.grad).sign() * random
        #print(binary_grad)
        #binary_grad[binary_grad == 0] = 1
        #data = self.kernels.data
        #new = (data - binary_grad).clamp(-1, 1)
        #Mask = ((new == 0) == 0).float()
        #self.kernels.data = ((data - binary_grad).clamp(-1, 1) - data * Mask)
        #self.weight.data = (self.weight.data - 0.1 * binary_grad)
        self.kernels.data = (self.kernels.data - binary_grad).clamp(-1, 1)
        #self.kernels.data[self.kernels.data == 0] = 1
        #print(Mask)
        
        
    def forward(self):
        return self.kernels
    
class BConv2d(nn.modules.conv._ConvNd):
      
    def __init__(self, 
        in_chs = 1, 
        out_chs = 1, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1,
        dilation = 1,
        groups = 1, 
        bias = False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BConv2d, self).__init__(in_chs, out_chs, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias)
        stdv = 2 / math.sqrt(self.weight.size(-1))       
        self.weight.data.uniform_(-stdv, stdv)
        self.Binarize_Function_A = Softsign_A(in_chs, out_chs)
        self.Binarize_Function_W = Softsign_W(in_chs, out_chs)
        self.ReLU = nn.ReLU()
        
    def forward(self, input):
        
        activation = self.Binarize_Function_A(input)
        #activation = F.relu(input)
        
        self.Weight = self.Binarize_Function_W(self.weight[1:,:,:,:])
        #self.Weight = F.relu(self.weight[1:self.out_channels//64:,:,:,:])
        
        out_1 = F.conv2d(F.relu(input).mean(1, keepdim = True), F.hardtanh(self.weight.mean(1, keepdim = True).mean(0, keepdim = True)), self.bias, self.stride, self.padding, self.dilation, self.groups)
        out_2 = F.conv2d(activation, self.Weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                   
        out = torch.cat((out_1, out_2), 1)
        '''
        self.Weight = self.Binarize_Function_W(self.weight)
        activation = self.Binarize_Function_A(input)
        out_1 = F.conv2d(activation, self.Weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #out_2 = F.conv2d(self.PReLU(input.mean(1,keepdim = True)), (self.weight.view(-1, 3, 3).mean(0,keepdim = True).unsqueeze(0)), self.bias, self.stride, self.padding, self.dilation, self.groups)
        out = out_1
        '''
        return out        

        
class BLinear(nn.Module):
    def __init__(self,
        in_chs = 1, 
        out_chs = 1, 
        bias = True, 
        activation_binarize = True,
        weight_binarize = True):
        super(BLinear, self).__init__()
        
        self.activation_binarize = activation_binarize
        self.weight_binarize = weight_binarize
        self.Binarize_function = Binarize()
        self.Binarize_Function_A = Softsign_A(in_chs, out_chs)
        self.Binarize_Function_W = Softsign_W(in_chs, out_chs)
        self.weight = Parameter(torch.FloatTensor(out_chs, in_chs).cuda())     
        stdv = 2 / math.sqrt(self.weight.size(-1))       
        self.weight.data.uniform_(-stdv, stdv)
         
    def forward(self, input):
        input = self.Binarize_Function_A(input)
        self.Weight = self.Binarize_Function_W(self.weight)
        
        output = F.linear(input, self.Weight.squeeze())
        
        return output
           
        
class BinarizeConv2d(nn.Module):
      
    def __init__(self, 
        in_chs = 1, 
        out_chs = 1, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        bias = True,
        weight_binarize = True,
        activation_binarize = True):
        super(BinarizeConv2d, self).__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 
        self.bias = bias
        self.weight_binarize = weight_binarize
        self.activation_binarize = activation_binarize
        self.weight = None
        
        self.reweight = Parameter(nn.init.kaiming_uniform_(torch.randn(out_chs-out_chs//32, in_chs, kernel_size, kernel_size)))
        self.conv = nn.Conv2d(in_chs, out_chs//32, kernel_size, stride = stride, padding = padding, bias = False)
        #self.K = Variable(torch.ones(1,1,kernel_size, kernel_size).mul(1.0/(kernel_size**2)).cuda())
        #self.bias = Parameter((torch.randn(out_chs)))
        stdv = 2 / math.sqrt(self.reweight.size(-1))       
        self.reweight.data.uniform_(-stdv, stdv)
        self.Binarize_Function_A = Softsign_A(in_chs, out_chs)
        self.Binarize_Function_W = Softsign_W(in_chs, out_chs)
       
        self.init = self.reweight.clone().detach()
        self.weight = self.reweight.data.detach()
        self.Weight = self.reweight.data.detach()
      
        
    def forward(self, input):
        #if input.size(1) != 3 and self.activation_binarize:
        self.Activation = (self.Binarize_Function_A(input))
        #else:
        #self.Activation = F.relu(input)
        
        #if self.weight_binarize:
        self.Weight = self.Binarize_Function_W(self.reweight)
        #else:
        #self.Weight = (self.reweight)
        '''
        A = input.abs().mean(1, keepdim = True)
        K = F.conv2d(A, self.K, stride = self.stride, padding = self.padding) 
        alpha = self.reweight.abs().view(self.reweight.shape[0], -1).mean(-1,keepdim = True).unsqueeze(0).unsqueeze(-1)
        '''   
        out_1 = F.conv2d(self.Activation, self.Weight, stride = self.stride, padding = self.padding)
        out_2 = self.conv(F.hardtanh(input))                           
        return torch.cat((out_1, out_2),1)
        #return out_1
        
class BinarizeHyperConv2d(nn.Module):
    def __init__(self, 
        in_chs = 1, 
        out_chs = 1, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        bias = True, 
        z_dim = 9, 
        BinarizeHyper = BinarizedHypernetwork_Parrallel,
        identity = True,
        activation_binarize = True,
        ste = 'clipped_elu',
        weight_binarize = True,
        depth = 1):
        super(BinarizeHyperConv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.z_dim = z_dim
        self.activation_binarize = activation_binarize
        #self.softsign = Softsign_A()
        self.Binarize_function = Binarize(channels = in_chs)
        #self.ELU = nn.ELU(alpha = (1e1/(1e1 - 1.0)))
        
        if identity:
          self.in_chs = in_chs/1
          self.out_chs = out_chs/1
          
          self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = self.in_chs, out_channels = self.out_chs, kernel_size = kernel_size, z_num = (self.out_chs, self.in_chs), bias = bias, binarize = weight_binarize, ste = ste, depth = 1).cuda()
          self.random_z = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(self.out_chs, self.in_chs, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))
          
        else:
          self.in_chs = in_chs
          self.out_chs = out_chs
          
          self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = self.in_chs, out_channels = self.out_chs, kernel_size = kernel_size, z_num = (1, 1), bias = bias, binarize = weight_binarize, ste = ste, depth = depth).cuda()
          self.random_z = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(1, 1, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))
          
        stdv = 2 / math.sqrt(self.random_z.size(-1))       
        self.random_z.data.uniform_(-stdv, stdv)
        self.Weight = self.BH(embed_vec = self.random_z).detach()
        self.init = self.Weight.clone().detach() 
        #self.random_z.data.normal_(-stdv, stdv)
        
        
        
    def forward(self, input):
      
        if self.activation_binarize:
          #self.Activation = self.Binarize_function((input))
          self.Activation = self.Binarize_function((input))
        else:
          self.Activation = (input)
        #if self.training:
        self.Weight = self.BH(embed_vec = self.random_z) 
        output = F.conv2d(self.Activation, self.Weight, stride = self.stride, padding = self.padding)
        
        return output


class BinarizeHyperLinear(nn.Module):
    def __init__(self,
        in_chs = 1, 
        out_chs = 1, 
        kernel_size = 1, 
        bias = True, 
        z_dim = 9, 
        BinarizeHyper = BinarizedHypernetwork_Parrallel,
        identity = True,
        activation_binarize = True,
        ste = 'clipped_elu',
        weight_binarize = True,
        depth = 1):
        super(BinarizeHyperLinear, self).__init__()
        
        #self.Weight = 0.0
        #self.Bias = 0.0
        
        self.kernel_size = kernel_size
        self.z_dim = z_dim
        self.activation_binarize = activation_binarize
        self.Binarize_function = Binarize()
        
        #self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = in_chs, out_channels = out_chs, kernel_size = kernel_size,  bias = bias, binarize = weight_binarize, ste = ste, depth = depth)
        self.reweight = Parameter(nn.init.kaiming_normal_(torch.FloatTensor(out_chs, in_chs).cuda(), mode = 'fan_out', nonlinearity = 'linear'))     
          
        #stdv = 2 / math.sqrt(self.reweight.size(-1))       
        #self.reweight.data.uniform_(-stdv, stdv).sign_()
        
         
    def forward(self, input):
        if self.activation_binarize:
          input = self.Binarize_function(input)
        
        #if self.training:
        self.Weight = self.reweight
        
        #print(input.max())
        #print(self.Weight.squeeze().size())  
        output = F.linear(input, self.Weight.squeeze())
        
        return output
        

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        
        return self.hinge_loss(input,target)          