import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import cv2


def AdaptiveBinarize_spatial_channel(tensor_input, quant_mode = 'det', kernel_size = 3):
		'''
		Inputs: weight tensor of shape [N, C, H, W]
		Outputs: binarized weight tensor of shape [N, C, H, W]
		'''
		if quant_mode == 'det':
			
			
			N, C, H, W = tensor_input.size()
			#neg_tensor = torch.where(neg_index, tensor, torch.zeros(1).cuda())
			#pos_tensor = torch.where(pos_index, tensor, torch.zeros(1).cuda())
			tensor = torch.mean(tensor_input, dim = 1, keepdim = True)
			
			if kernel_size == 2:
				mean = F.avg_pool2d(F.pad(tensor, (1,0,1,0)), kernel_size = (kernel_size, kernel_size), stride = 2)
				
				mean_col = torch.cat((mean.unsqueeze(2), mean.unsqueeze(2)), 2).transpose(2,3).contiguous().view(N, 1, H, W//2)
				mean_row = torch.cat((mean_col.unsqueeze(3), mean_col.unsqueeze(3)), 3).transpose(3,4).contiguous().view(N, 1, H, W)
				
				
				#neg_mean = F.avg_pool2d(F.pad(neg_tensor, (1,0,1,0)), kernel_size = (kernel_size, kernel_size), stride = 1)
				#pos_mean = F.avg_pool2d(F.pad(pos_tensor, (1,0,1,0)), kernel_size = (kernel_size, kernel_size), stride = 1)		
				#assert neg_mean.size() == tensor.size()
				
			elif kernel_size == 3:
				mean = F.avg_pool2d(tensor, kernel_size = (kernel_size, kernel_size), stride = 2, padding = 1)
				#print(tensor.size())
				#print(mean.size())
				#neg_mean = F.avg_pool2d(neg_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 1)
				#pos_mean = F.avg_pool2d(pos_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 1)
				#assert neg_mean.size() == tensor.size()
				
				mean_col = torch.cat((mean.unsqueeze(2), mean.unsqueeze(2)), 1).transpose(2,3).contiguous().view(N, 1, H, W//2)
				mean_row = torch.cat((mean_col.unsqueeze(3), mean_col.unsqueeze(3)), 1).transpose(3,4).contiguous().view(N, 1, H, W)
				
			elif kernel_size == 5:
				mean = F.avg_pool2d(tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 2)
				#neg_mean = F.avg_pool2d(neg_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 2)
				#pos_mean = F.avg_pool2d(pos_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 2)
				#assert neg_mean.size() == tensor.size()

				mean_col = torch.cat((mean.unsqueeze(2), mean.unsqueeze(2)), 1).transpose(2,3).contiguous().view(N, 1, H, W//2)
				mean_row = torch.cat((mean_col.unsqueeze(3), mean_col.unsqueeze(3)), 1).transpose(3,4).contiguous().view(N, 1, H, W)
				
			tensor = torch.where(tensor>=mean_row, torch.ones(1).cuda(), -torch.ones(1).cuda())
			#neg_tensor = torch.where(neg_tensor<=neg_mean, -torch.ones(1).cuda(), torch.ones(1).cuda())
			#pos_tensor = torch.where(pos_tensor>=pos_mean, torch.ones(1).cuda(), -torch.ones(1).cuda())
			#tensor_quantized = torch.zeros_like(tensor)
			
			#tensor[neg_index] = neg_tensor[neg_index]
			#tensor[pos_index] = neg_tensor[pos_index]
		tensor = tensor.expand(N, C, H, W)
			
		return tensor
		
		
def AdaptiveBinarize_spatial(tensor, quant_mode = 'det', kernel_size = 3):
		'''
		Inputs: weight tensor of shape [N, C, H, W]
		Outputs: binarized weight tensor of shape [N, C, H, W]
		'''
		if quant_mode == 'det':
			
			neg_index = tensor<=0
			pos_index = tensor>0
			
			
			#neg_tensor = torch.where(neg_index, tensor, torch.zeros(1).cuda())
			#pos_tensor = torch.where(pos_index, tensor, torch.zeros(1).cuda())
			N, C, H, W = tensor.size()
			
			if kernel_size == 2:
				mean = F.avg_pool2d(F.pad(tensor, (1,0,1,0)), kernel_size = (kernel_size, kernel_size), stride = 1)
				mean_col = torch.cat((mean.unsqueeze(2), mean.unsqueeze(2)), 1).transpose(2,3).contiguous().view(N, C, H, W//2)
				mean_row = torch.cat((mean_col.unsqueeze(3), mean_col.unsqueeze(3)), 1).transpose(3,4).contiguous().view(N, C, H, W)
				#neg_mean = F.avg_pool2d(F.pad(neg_tensor, (1,0,1,0)), kernel_size = (kernel_size, kernel_size), stride = 1)
				#pos_mean = F.avg_pool2d(F.pad(pos_tensor, (1,0,1,0)), kernel_size = (kernel_size, kernel_size), stride = 1)		
				#assert neg_mean.size() == tensor.size()
				
			elif kernel_size == 3:
				mean = F.avg_pool2d(tensor, kernel_size = (kernel_size, kernel_size), stride = 2, padding = 1)
				mean_col = torch.cat((mean.unsqueeze(2), mean.unsqueeze(2)), 1).transpose(2,3).contiguous().view(N, C, H, W//2)
				mean_row = torch.cat((mean_col.unsqueeze(3), mean_col.unsqueeze(3)), 1).transpose(3,4).contiguous().view(N, C, H, W)
				#neg_mean = F.avg_pool2d(neg_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 1)
				#pos_mean = F.avg_pool2d(pos_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 1)
				#assert neg_mean.size() == tensor.size()
				
			elif kernel_size == 5:
				mean = F.avg_pool2d(tensor, kernel_size = (kernel_size, kernel_size), stride = 2, padding = 2)
				mean_col = torch.cat((mean.unsqueeze(2), mean.unsqueeze(2)), 1).transpose(2,3).contiguous().view(N, C, H, W//2)
				mean_row = torch.cat((mean_col.unsqueeze(3), mean_col.unsqueeze(3)), 1).transpose(3,4).contiguous().view(N, C, H, W)
				#neg_mean = F.avg_pool2d(neg_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 2)
				#pos_mean = F.avg_pool2d(pos_tensor, kernel_size = (kernel_size, kernel_size), stride = 1, padding = 2)
				#assert neg_mean.size() == tensor.size()

			tensor = torch.where(tensor>=mean_row, torch.ones(1).cuda(), -torch.ones(1).cuda())
			#neg_tensor = torch.where(neg_tensor<=neg_mean, -torch.ones(1).cuda(), torch.ones(1).cuda())
			#pos_tensor = torch.where(pos_tensor>=pos_mean, torch.ones(1).cuda(), -torch.ones(1).cuda())
			#tensor_quantized = torch.zeros_like(tensor)
			
			#tensor[neg_index] = neg_tensor[neg_index]
			#tensor[pos_index] = neg_tensor[pos_index]
						
		return tensor


def AdaptiveBinarize_channel(tensor, quant_mode = 'det', kernel_size = 3):
		'''
		Inputs: weight tensor of shape [N, C, H, W]
		Outputs: binarized weight tensor of shape [N, C, H, W]
		'''
		if quant_mode == 'det':
			mean = torch.mean(tensor, dim = 1, keepdim = True).expand_as(tensor)
			tensor = torch.where(tensor>=mean, torch.ones(1).cuda(), -torch.ones(1).cuda())
						
		return tensor
		
				
def AdaptiveBinarize_cpu(tensor, quant_mode = 'det', kernel_size = 3):
		'''
		Inputs: weight tensor of shape [N, C, H, W]
		Outputs: binarized weight tensor of shape [N, C, H, W]
		'''
		if quant_mode == 'det':
			tensor = tensor.cpu().numpy()
			for Out in tensor:
				for weight in Out:
					#print(weight)
					weight = cv2.adaptiveThreshold(np.uint8(weight), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, 0)
					weight[weight==0]=-1
			tensor = torch.from_numpy(tensor).float().cuda()
			#('binarized shape:{}'.format(tensor.size()))		
			return tensor
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
'''
class spatial_attention(tensor):
		def __init__(self):
				super(spatial_attention)
		shape = tensor.size()
		assert len(shape) == 4, 'Only 4-D tensors are supported'
		
		N, C, H, W = shape
		
		tensor = 
'''


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

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

import torch.nn._functions as tnnf


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            #input.data=AdaptiveBinarize_channel(input.data)
	    input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
	self.weight.data = Binarize(self.weight.org)
        #self.weight.data=AdaptiveBinarize_channel(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            self.bias.data = Binarize(self.bias.org)
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
        	input.data = Binarize(input.data)
        	#input.data = AdaptiveBinarize_channel(input.data, kernel_size = 5)
        	'''
            if input.size(3)>8:
            	#input.data = Binarize(input.data)
            	input.data = AdaptiveBinarize_channel(input.data, kernel_size = 5)
            elif input.size(3)>1:
            	input.data = AdaptiveBinarize_channel(input.data, kernel_size = 3)
            else: 
            	input.data = AdaptiveBinarize_channel(input.data, kernel_size = 2)
        	'''
        	#print(input.size())
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()
        if self.weight.data.size(-1) == 1:
        	self.weight.data = Binarize(self.weight.org)
       	elif self.weight.data.size(-1) == 3:
       		self.weight.data = Binarize(self.weight.org)
        	#self.weight.data = AdaptiveBinarize_channel(self.weight.org, kernel_size = 2)
        #assert self.weight.data.size() == self.weight.org.data.size(), 'Binarized tensor must have the same size as original one'

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            self.bias.data = Binarize(self.bias.org)
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
