import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
from torchvision.transforms import transforms
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
        k = 2.7
        factor = (1 - math.sqrt(1 - 2/k))
        #factor = (1.0 - 1.0/k)
        #factor = (5 - math.sqrt(13))/6
        #weight = k*(1 - factor)
        x = ctx.saved_tensors[0]
        #factor = x.max()
        abs_x = torch.abs(x)
        mask_0 = abs_x<=1
        
        mask_1 = abs_x>=factor
        mask_2 = abs_x<factor
        #x_grad =k * ((1 - abs_x)).to(grad_output.dtype) * (mask_0).to(grad_output.dtype) + 1.*(mask_2 * mask_1).to(grad_output.dtype)
        x_grad =k * ((1 - abs_x)).to(grad_output.dtype) * mask_2.to(grad_output.dtype) + (mask_0 * mask_1).to(grad_output.dtype)
        #x_grad = ((1 - (abs(x))))
        #x_grad =  k * ((1 - abs_x/1.0)).to(grad_output.dtype) * mask_0.to(grad_output.dtype)
        
        #return grad_output * mask_0.to(grad_output.dtype)
        return grad_output *  x_grad

class BinarizeFC(Function):
    @staticmethod   
    def forward(ctx, input, quant_mode = 'det'):
      ctx.save_for_backward(input)
      if quant_mode=='det':
              return input.sign()
      else:
              tensor = input.data
              input.data = tensor.add_(1).div_(2).add_((torch.rand(tensor.size()).cuda()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
              return input
        
    @staticmethod   
    def backward(ctx, grad_output):
        k = 2.7
        factor = (1 - math.sqrt(1 - 2/k))
        #factor = (1.0 - 1.0/k)
        #factor = (5 - math.sqrt(13))/6
        #weight = k*(1 - factor)
        x = ctx.saved_tensors[0]
        #factor = x.max()
        abs_x = torch.abs(x)
        mask_0 = abs_x<=1
        
        mask_1 = abs_x>=factor
        mask_2 = abs_x<factor
        #x_grad =k * ((1 - abs_x)).to(grad_output.dtype) * (mask_0).to(grad_output.dtype) + 1.*(mask_2 * mask_1).to(grad_output.dtype)
        x_grad =k * ((1 - abs_x)).to(grad_output.dtype) * mask_2.to(grad_output.dtype) + (mask_0 * mask_1).to(grad_output.dtype)
        #x_grad = ((1 - (abs(x))))
        #x_grad =  k * ((1 - abs_x/1.0)).to(grad_output.dtype) * mask_0.to(grad_output.dtype)
        
        #return grad_output * mask_0.to(grad_output.dtype)
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

class HardtanhF(Function):
    @staticmethod   
    def forward(ctx, input):
        ctx.save_for_backward(input)
      
        return input
          
        
    @staticmethod   
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        mask = torch.abs(x)<=1
        x_grad = 5 * ((1 - 0.1*(abs(x))))
        
        return grad_output * (mask).to(grad_output.dtype) * x_grad.to(grad_output.dtype)


class Weight_Norm(Function):
    @staticmethod   
    def forward(ctx, input):
        ctx.save_for_backward(input)
      
        return input.pow(2).sum(dim = 1, keepdim = True).add(1e-6).sqrt()
          
        
    @staticmethod   
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        mask = torch.abs(x)<=1
        x_grad = 10 * ((1 -  (abs(x)))) + 0.1
        
        return grad_output * (mask).to(grad_output.dtype) * x_grad.to(grad_output.dtype)

        
class StochasticBinarizeF(Function):
    @staticmethod   
    def forward(ctx, input):
        output = input
        output = output.add_(1).div_(2).add_(torch.rand(output.size()).cuda().add(-0.5)).clamp_(0, 1).floor_().mul_(2).add_(-1)
        #output = torch.bernoulli(output).mul_(2).add_(-1)
        return output
        
    @staticmethod   
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        
        return grad_input

binarize = BinarizeF.apply
binarizec = BinarizeFC.apply
binarizew = BinarizeWF.apply
stbinarize = StochasticBinarizeF.apply
hardtanh = HardtanhF.apply
weightnorm = Weight_Norm.apply

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
          #output = input
          #weight = torch.mean(output.view(shape[0], -1), dim = -1, keepdim = True).unsqueeze(-1).unsqueeze(-1)
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
      

class WeightNormlinear(nn.Module):

    def __init__(self, in_channels, out_channels, scale = True, bias = True, init_factor = 4, init_scale = 1, one_scale = True):
        super(WeightNormlinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(out_channels, in_channels))
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
        stdv =  1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data.normal_(0, stdv)
  
        if self.bias is not None:
           self.bias.data.normal_(0, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1, keepdim = True).add(1e-6).sqrt()
  
    def norm_scale_bias(self, input):
        output = F.linear(input, self.weight)
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
    
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        
        return out
        
class IdentityLayer(nn.Module):

    def forward(self, x):
        return x

class ResNetBlock(nn.Module):

    def __init__(self, in_size=16, out_size=16, downsample = False):
        super(ResNetBlock,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.downsample = downsample
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x, conv1_w, conv2_w, conv3_w):
        if self.downsample:
          residual = F.conv2d(x, conv3_w, stride = 2, padding = 0)
        else:
          residual = x
    
        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
        out = self.bn2(F.conv2d(out, conv2_w, padding=1))

        out += residual

        out = F.relu(out)

        return out
        
class ResNetBlock_A(nn.Module):

    def __init__(self, in_size=16, out_size=16, kernel_size = 5, downsample = False):
        super(ResNetBlock_A,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.downsample = downsample
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride = self.stride1, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()
        self.conv1 = nn.Conv2d(in_channels = in_size, out_channels = out_size, kernel_size = kernel_size, stride = self.stride1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = out_size, out_channels = out_size, kernel_size = kernel_size, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        
        residual = self.reslayer(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual

        out = F.relu(out)

        return out
        
        
class ResNetBlock_B(nn.Module):

    def __init__(self, in_size=16, out_size=16, downsample = False):
        super(ResNetBlock_B,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.downsample = downsample
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(out_size, affine = True)
        self.bn2 = nn.BatchNorm2d(out_size, affine = True)
        self.bn3 = nn.BatchNorm2d(out_size, affine = True)
        self.htanh1 = nn.Hardtanh(inplace=True)
        self.htanh2 = nn.Hardtanh(inplace=True)
        
    def forward(self, x, conv1_w, conv2_w, conv3_w):
        '''
        self.w1.org = conv1_w.data
        self.w2.org = conv2_w.data
        self.w3.org = conv3_w.data
        
        
        self.w1_binary = Variable(conv1_w.data, requires_grad = True)
        self.w2_binary = Variable(conv2_w.data, requires_grad = True)
        self.w3_binary = Variable(conv3_w.data, requires_grad = True)
        
        self.w1.data = binarize(self.w1.org)
        self.w2.data = binarize(self.w2.org)
        self.w3.data = binarize(self.w3.org)
        '''
        #conv1_w = self.bnntanh1(conv1_w)
        #conv2_w = self.bnntanh2(conv2_w)
        #conv3_w = self.bnntanh3(conv3_w)
        '''
        conv1_w, w1_mean = conv1_w[0], conv1_w[1]
        conv2_w, w2_mean = conv2_w[0], conv2_w[1]
        if len(conv3_w) == 2:
          conv3_w, w3_mean = conv3_w[0], conv3_w[1]
          #print(conv3_w.size())
          #print(w3_mean.size())
        '''
        x.data = polynomial(x.data)
        if self.downsample:
          residual = x.clone()
          residual.data = binarize((residual.data))
          residual = self.bn3(F.conv2d(residual, ((conv3_w)), stride = 2, padding = 0))
        else:
          residual = x.clone()
        
        x.data = binarize((x.data))
        out_1 = self.htanh1(self.bn1(F.conv2d(x, ((conv1_w)), stride = self.stride1, padding=1)))
        
        out_1.data = binarize(polynomial(out_1.data))
        out_2 = self.bn2(F.conv2d(out_1, ((conv2_w)), stride = 1, padding=1))
        out_2 =  out_2 + residual
        out_2 = self.htanh2(out_2)
        
        return out_2


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            

       
class StaticHypernetwork_Parrallel(nn.Module):
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
        super(StaticHypernetwork_Parrallel, self).__init__()
        
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
    
        
        #weight_mean = reweight_mean(weight)
        #weight_mean = reweight_mean(weight)
        #weight_mean = self.fc3(weight_1).view(N_O, N_I, self.out_channels)
        #weight_mean = (F.sigmoid(self.fc7(self.fc6(weight_mean))) * weight_mean).permute(0, 2, 1).contiguous().view(N_O * self.out_channels, N_I, 1, 1).mean(1, keepdim = True)
        #weight_mean = (F.sigmoid(self.fc7(self.fc6(weight_mean))) * weight_mean).view(N_O * self.out_channels, N_I, 1, 1).mean(1, keepdim = True)
        
        

                        
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
        self.Weight = 0.0
        self.Bias = 0.0
        self.share = share
        self.res = res

        if identity:
          self.in_chs = in_chs/16/1
          self.out_chs = out_chs/16/1
          if not share:
            self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = 16*1, out_channels = 16*1, kernel_size = kernel_size, z_num = (self.out_chs, self.in_chs), bias = bias, binarize = binarize)
          self.random_z = Parameter(nn.init.kaiming_normal_(torch.FloatTensor(self.out_chs, self.in_chs, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))
          #self.random_z = Parameter(torch.rand(self.out_chs, self.in_chs, z_dim).cuda())
        else:
          self.in_chs = in_chs
          self.out_chs = out_chs
          if not share:
            self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = self.in_chs, out_channels = self.out_chs, kernel_size = kernel_size, z_num = (1, 1), bias = bias, binarize = binarize)
          self.random_z = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(1, 1, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))
        #self.random_z = Parameter(torch.rand(1, 1, z_dim).cuda())
        #stdv = math.sqrt( 2. / self.random_z.size(-1))           
        stdv = 5. / math.sqrt(self.random_z.size(-1))       
        self.random_z.data.uniform_(-stdv, stdv)
        #self.random_z.data.normal_(0, stdv)
        
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
        if not share:
          self.BH = BinarizeHyper(embed_vec_dim = z_dim, in_channels = self.in_chs, out_channels = self.out_chs, kernel_size = kernel_size, bias = bias, binarize = binarize)
        else:
          self.BH = None        
        self.random_z = Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(1, 1, z_dim).cuda(), mode = 'fan_out', nonlinearity = 'linear'))     
        #stdv = math.sqrt( 2. / self.random_z.size(-1))       
        stdv = 5. / math.sqrt(self.random_z.size(-1))
        #self.random_z.data.normal_(0, stdv)
        self.random_z.data.uniform_(-stdv, stdv)
        #self.random_z.data.normal_(0, stdv)

        #self.random_z = Parameter(torch.rand(1, 1, z_dim).cuda())
        self.Weight = 0.0
        self.Bias = 0.0
        
    def forward(self, input):
        
        input = binarize((input))
        self.Weight, self.Bias = self.BH(embed_vec = self.random_z)
        output = F.linear(input, self.Weight.squeeze())

        return output


class BinaryHyperConv3x3(nn.Module):
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
                res = True,):
                  
        super(BinaryHyperConv3x3, self).__init__()
        self.BinaryConv3x3 = BinarizeHyperConv2d(in_chs, out_chs, kernel_size, stride, padding, bias, z_dim, BinarizeHyper, identity, binarize, share, res)
        
    def forward(self, input):
        output = self.BinaryConv3x3(input)
        return output
            
class BasicBlock(nn.Module):
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, share = False, do_bntan=True):
        super(BasicBlock, self).__init__()
        self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, BinarizeHyper = StaticHypernetwork_Parrallel, share = share)
        self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, BinarizeHyper = StaticHypernetwork_Parrallel, share = share)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1)
        
        #if not downsample:
        # self.bn4 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, input):
        
        x = input
        residual = x.clone()
        if self.downsample is not None:
          residual = self.downsample(residual)

        
        out_1 = self.conv1(x)     
        out = self.bn1(out_1)
        
        out_2 = self.conv2(out)
        out_2 = self.bn2(out_2)
        out_2 += residual
        
        return out_2
          
          
class CombinatorialConv2d(nn.Module):
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
          super(CombinatorialConv2d, self).__init__()
          
          self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, kernel_size = kernel_size, stride=stride, padding = padding, z_dim = z_dim, BinarizeHyper = StaticHypernetwork_Parrallel, share = False, binarize = True, res = True)
          self.conv2 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, kernel_size = kernel_size, stride=stride, padding = padding, z_dim = z_dim, BinarizeHyper = StaticHypernetwork_Parrallel, share = False, binarize = True, res = True)
          self.weight = Parameter(torch.FloatTensor(2).fill_(1.)).cuda()
          self.bn1 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1)
          self.conv0 = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding = padding)
      
      def forward(self, x):
          x1 = self.conv1(x).mul(self.weight[0])
          x2 = self.conv2(x).mul(self.weight[1])
          
          x3 = x1 + x2
          x3 = self.bn1(x3)
          
          return x3
                    
class BNReLUConv2d(nn.Module):
    def __init__(self, in_chs = 16, out_chs = 16, stride=1, downsample=None, share = False, do_bntan=True):
        super(BNReLUConv2d, self).__init__()
        self.conv1 = BinaryHyperConv3x3(in_chs = in_chs, out_chs = out_chs, stride = stride, BinarizeHyper = StaticHypernetwork_Parrallel, share = share)
        self.bn1 = nn.BatchNorm2d(in_chs, affine = True, momentum = 0.1)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = BinaryHyperConv3x3(in_chs = out_chs, out_chs = out_chs, BinarizeHyper = StaticHypernetwork_Parrallel, share = share)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1)
        
        #if not downsample:
        # self.bn4 = nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1)

        self.downsample = downsample
        self.do_bntan=do_bntan;
        self.stride = stride

    def forward(self, input):
        x = input
        residual = x.clone()
        if self.downsample is not None:
          residual = self.downsample(residual)

        out = self.bn1(out_1)
        out_1 = self.conv1(x)     
        
        
        out_2 = self.conv2(out)
        out_2 = self.bn2(out_2)
        out_2 += residual
        
        return out_2


        

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, out_chs, blocks, stride=1, z_dim = 64, share = False, do_bntan=True):
        downsample = None
        if stride != 1 or self.in_chs != out_chs:
            #downsample = nn.Sequential(CombinatorialConv2d(self.in_chs, out_chs, kernel_size=1, stride=stride, padding = 0, z_dim = z_dim, BinarizeHyper = StaticHypernetwork_Parrallel, share = False, binarize = True, res = False))
            if out_chs <= 128:
              kernel_size = 3
              padding = 1
              
            else:
              kernel_size = 3
              padding = 1
              
            downsample = nn.Sequential(
                #BinarizeHyperConv2d(self.in_chs, out_chs, kernel_size=3, stride=stride, padding = 1, z_dim = z_dim, BinarizeHyper = StaticHypernetwork_Parrallel, share = False, binarize = True, res = False),
                BinarizeHyperConv2d(self.in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding = padding, z_dim = z_dim, BinarizeHyper = StaticHypernetwork_Parrallel, share = False, binarize = True, res = False),
                nn.BatchNorm2d(out_chs, affine = True, momentum = 0.1),
            )
            
        layers = []
        layers.append(block(self.in_chs, out_chs, stride, downsample, share = share))
        self.in_chs = out_chs
        for i in range(1, blocks-1):
            layers.append(block(self.in_chs, out_chs, share = share))
        layers.append(block(self.in_chs, out_chs, share = share, do_bntan=do_bntan))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.dropout2d(x, p = 0.5)
        
        x = self.layer1(x)
        x = F.dropout2d(x, p = 0.5)
        
        x = self.layer2(x)
        x = F.dropout2d(x, p = 0.5)
        
        x = self.layer3(x)
        x = F.dropout2d(x, p = 0.5)
        
        x = self.layer4(x)
        x = F.dropout2d(x, p = 0.5)
        
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.bn2(x)
        
        x = self.logsoftmax(x.view(x.size(0), -1))

        return x

       
class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 4
        self.in_chs = 16*self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=7, stride=2, padding=3)        
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate, affine = True, momentum = 0.1)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n, z_dim = 64)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2, z_dim = 64)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2, z_dim = 64)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        
        self.bn3 = nn.BatchNorm2d(10, affine = True, momentum = 0.95)
        self.conv2 = BinaryHyperConv3x3(64*self.inflate, num_classes, kernel_size=3, stride=1, padding=1, BinarizeHyper = StaticHypernetwork_Parrallel, identity = False, binarize = True, res = False)   
        self.logsoftmax = nn.LogSoftmax()
  

class ResNet_cifar10_share(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10_share, self).__init__()
        self.inflate = 2
        self.in_chs = 16*self.inflate
        n = int((depth - 2) / 6)
        self.bn0 = nn.BatchNorm2d(3, affine = True, momentum = 0.1)
        self.conv1 = BinaryHyperConv3x3(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, BinarizeHyper = StaticHypernetwork_Parrallel, identity = False, binarize = True, share = False, res = False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate, affine = True, momentum = 0.1)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n, z_dim = 64, share = True)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2, z_dim = 64, share = True)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2, z_dim = 64, share = True, do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(64*self.inflate, affine = True, momentum = 0.1)
        self.bn3 = nn.BatchNorm1d(10, affine = False, momentum = 0.99)
        self.fc = BinarizeHyperLinear(64*self.inflate, num_classes, BinarizeHyper = StaticHypernetwork_Parrallel, binarize = True)
        self.logsoftmax = nn.LogSoftmax()
              
        self.HyperNetwork = StaticHypernetwork_Parrallel(embed_vec_dim = 64, in_channels = 16*self.inflate, out_channels = 16*self.inflate, kernel_size = 3, bias = True, binarize = True)
  
        for module in self.modules():
            if isinstance(module, BinarizeHyperConv2d) and module.share == True:
                module.BH = self.HyperNetwork 
        
        
        
         
class ResNet18(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet18, self).__init__()
        
        self.inflate = 4
        self.in_chs = 16*self.inflate
        n = int((depth - 2) / 6)
        
        self.conv1 = nn.Conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1)
        #self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(16*self.inflate, affine = True, momentum = 0.1)
        
        self.layer1 = self._make_layer(block, 16*self.inflate, n, z_dim = 64)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2, z_dim = 64)
        self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2, z_dim = 64)
        self.layer4 = self._make_layer(block, 128*self.inflate, n, stride=2, z_dim = 64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv2 = BinaryHyperConv3x3(128*self.inflate, num_classes, kernel_size=3, stride=1, padding=1, BinarizeHyper = StaticHypernetwork_Parrallel, identity = False, binarize = True, res = False)   
        self.bn2 = nn.BatchNorm2d(num_classes, affine = True, momentum = 0.95)
        self.logsoftmax = nn.LogSoftmax()
        
        
        
        self.input_transforms = {
            'train': transforms.Compose([
                transforms.Scale(36),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
            ]),
            
            'eval': transforms.Compose([
                transforms.Scale(36),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
            ])
              
        }
        
if __name__ == "__main__":
   conv = BinaryHyperConv3x3(1,3,3,1,1, BinarizeHyper = StaticHypernetwork_Parrallel, identity = False).cuda()
   input = torch.randn(1, 1, 12, 12).cuda()
   output = conv(input)
   
   print(output.size())