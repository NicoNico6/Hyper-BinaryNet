import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
from numpy.random import random
import math

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
    mask = torch.le(input,0)
    output_pos = (2 - input)*input
    output_neg = (2 + input)*input
    output_pos[mask] = output_neg[mask]
    
    return output_pos

        
class BinarizeF(Function):
    @staticmethod   
    def forward(ctx, input):
      ctx.save_for_backward(input)
      
      return input.sign()
        
    @staticmethod   
    def backward(ctx, grad_output):
        k = 2.5
        factor = (1 - math.sqrt(1 - 2/k))
        x = ctx.saved_tensors[0]
        abs_x = torch.abs(x)
        mask_0 = abs_x<=1
        
        mask_1 = abs_x>=factor
        mask_2 = abs_x<factor
        x_grad =k * ((1 - abs_x)).to(grad_output.dtype) * mask_2.to(grad_output.dtype) + (mask_0 * mask_1).to(grad_output.dtype)
        return grad_output *  x_grad


class BinarizeWF(Function):
    @staticmethod   
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()
        
    @staticmethod   
    def backward(ctx, grad_output):
        k = 10
        factor = (1 - math.sqrt(1 - 2/k))
        x = ctx.saved_tensors[0]
        abs_x = torch.abs(x)
        mask_0 = abs_x<=1
        
        mask_1 = abs_x>=factor
        mask_2 = abs_x<factor
        x_grad =k * ((1 - abs_x)).to(grad_output.dtype) * mask_2.to(grad_output.dtype) + (mask_0 * mask_1).to(grad_output.dtype)

        return grad_output *  (mask_0).to(grad_output.dtype)


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

          
class BinarizedHypernetwork_Parrallel(nn.Module):
    """Static hypernetwork described in https://arxiv.org/pdf/1609.09106.pdf"""
    def __init__(self, 
        embed_vec_dim = 64, 
        in_channels = 16, 
        out_channels = 16, 
        kernel_size = 3,
        z_num = (1,1), 
        bias=True,
        binarize = True,
        reweight = False):
        super(BinarizedHypernetwork_Parrallel, self).__init__()
        
        self.embed_vec_dim = embed_vec_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.binarize = binarize
        self.reweight = reweight
        
        self.fc0 = WeightNormlinear(in_channels = self.embed_vec_dim*z_num[1], out_channels = self.out_channels, scale = False, bias = False, one_scale = True)        
        self.fc2 = WeightNormlinear(in_channels = self.embed_vec_dim, out_channels = self.out_channels * self.embed_vec_dim, scale = False, bias=False, one_scale = True)
        self.fc3 = WeightNormlinear(in_channels = self.embed_vec_dim, out_channels = self.in_channels * self.kernel_size * self.kernel_size, scale = False, bias=False, one_scale = True)

    def forward(self, embed_vec):
       
        N_O, N_I, E = embed_vec.size()
        
        N = N_I * N_O
     
        bias = (self.fc0(embed_vec.view(N_O, -1)).view(N_O*self.out_channels, 1)) 
  
        weight = (self.fc2(embed_vec))
        weight = F.dropout(weight.view(N_O, N_I, self.out_channels, self.embed_vec_dim))
        
        weight = (self.fc3(weight))
        weight = (weight.view(N_O, N_I, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).permute(0, 2, 1, 3, 4, 5).contiguous().view(N_O * self.out_channels, N_I * self.in_channels, self.kernel_size, self.kernel_size))

        if self.binarize:
              weight = (binarizew((weight)))
              bias = binarizew((bias))          
    
        if self.reweight:
              weight = reweight(weight)

        return  weight, bias
        

class BinarizeHyperConv2d(nn.Module):
    def __init__(self, 
        in_chs = 16, 
        out_chs = 16, 
        kernel_size = 3, 
        stride = 1, 
        padding = 1, 
        bias = True, 
        z_dim = 64, 
        BinarizeHyper = None,
        identity = True,
        binarize = True,
        share = False,
        res = True):
        super(BinarizeHyperConv2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.z_dim = z_dim
        self.share = share
        self.res = res
        
        self.Weight = 0.0
        self.Bias = 0.0
       
        if identity:
          self.in_chs = in_chs/16/1
          self.out_chs = out_chs/16/1
          if not share:
            self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = 16*1, out_channels = 16*1, kernel_size = kernel_size, z_num = (self.out_chs, self.in_chs), bias = bias, binarize = binarize)
          self.random_z = Parameter(nn.init.kaiming_normal_(torch.FloatTensor(self.out_chs, self.in_chs, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))
        
        else:
          self.in_chs = in_chs
          self.out_chs = out_chs
          if not share:
            self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = self.in_chs, out_channels = self.out_chs, kernel_size = kernel_size, z_num = (1, 1), bias = bias, binarize = binarize)
          self.random_z = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(1, 1, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))
        
        stdv = 5. / math.sqrt(self.random_z.size(-1))       
        self.random_z.data.uniform_(-stdv, stdv)
        
        
    def forward(self, input):
        if input.size(1) != 3:
          input = binarize(input)

        
        self.Weight, self.Bias = self.BH(embed_vec = self.random_z)
        output = F.conv2d(input, self.Weight, stride = self.stride, padding = self.padding)
        output += self.Bias.view(1, -1, 1, 1).expand_as(output)
      
        return output

class BinarizeHyperLinear(nn.Module):
    def __init__(self,
        in_chs = 16,
        out_chs = 16,
        kernel_size = 1,
        bias = True,
        z_dim = 64,
        BinarizeHyper = None,
        binarize = True,
        share = False,
        res = True):
        super(BinarizeHyperLinear, self).__init__()
        
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.z_dim = z_dim
        self.res = res
        
        self.Weight = 0.0
        self.Bias = 0.0
        
        if not share:
          self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = self.in_chs, out_channels = self.out_chs, kernel_size = kernel_size, bias = bias, binarize = binarize)
        else:
          self.BH = None        
        self.random_z = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(1, 1, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))     
        
        stdv = 5. / math.sqrt(self.random_z.size(-1))
        self.random_z.data.uniform_(-stdv, stdv)
         
    def forward(self, input):
        
        input = binarize((input))
        self.Weight, self.Bias = self.BH(embed_vec = self.random_z)
        output = F.linear(input, self.Weight.squeeze())

        return output          
