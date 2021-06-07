import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from resnet_blocks import BinaryTanh, binarize

class BinarizedHypernetwork_Parrallel(nn.Module):
    """Static hypernetwork described in https://arxiv.org/pdf/1609.09106.pdf"""
    def __init__(self, embed_vec_dim = 64, in_channels = 16, out_channels = 16, kernel_size = 3, bias=True):
        super(StaticHypernetwork_Parrallel, self).__init__()
        self.embed_vec_dim = embed_vec_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.fc0 = nn.Linear(self.embed_vec_dim, self.embed_vec_dim, bias=True)
        #self.bn0 = nn.BatchNorm1d(self.out_channels * self.embed_vec_dim)
        self.fc1 = nn.Linear(self.embed_vec_dim, self.out_channels * self.embed_vec_dim, bias=True)

        self.fc2 = nn.Linear(self.embed_vec_dim, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        #
        #self.fc3 = nn.Linear(self.out_channels * self.embed_vec_dim, self.out_channels * self.embed_vec_dim/64, bias=True)
        #self.fc4 = nn.Linear(self.out_channels * self.embed_vec_dim/64, self.out_channels * self.embed_vec_dim, bias=True)
        
        self.fc5 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size, self.in_channels * self.kernel_size * self.kernel_size/16, bias=True)
        self.fc6 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size/16, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        #self.BT = BinaryTanh()
        #self.fc5 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        
        self.weight_norm()
        
    def weight_norm(self):
    		for module in self.modules():
    			if isinstance(module, nn.Linear):
    				weight = module.weight.data
    				shape = weight.size()
    				#module.weight.data.normal_(0.0, torch.sqrt(4.0 / (shape[0] * shape[1])))
    				module.weight.data = nn.init.kaiming_normal(weight, mode = 'fan_out', nonlinearity = 'linear') 
    				   
    def forward(self, embed_vec):
        
        try:
        	#self.weight_norm(self.fc1)
        	#self.weight_norm(self.fc2)
        	N, E = embed_vec.size()
        	#if N >= 2:
        	#	embed_vec = self.bn0(embed_vec)
        	#embed_vec = (self.fc0(embed_vec)) * embed_vec
        	weight = (self.fc1(embed_vec))
        	#weight = F.sigmoid(self.fc4((self.fc3(weight)))) * weight
        	weight = weight.view(N, self.out_channels, self.embed_vec_dim)
        	weight = self.fc2(weight)
        	#weight = (F.sigmoid(self.fc6((self.fc5(weight)))).mul(2).add(-1)) * weight
        	weight = (F.softmax(self.fc6((self.fc5(weight)))), -1) * weight
        	return  weight.view(N, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        finally:	
        	pass
        	
class ProbabilityHypernetwork_Parrallel(nn.Module):
    """Static hypernetwork described in https://arxiv.org/pdf/1609.09106.pdf"""
    def __init__(self, embed_vec_dim = 64, in_channels = 16, out_channels = 16, kernel_size = 3, bias=True):
        super(ProbabilityHypernetwork_Parrallel, self).__init__()
        self.embed_vec_dim = embed_vec_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.fc1 = nn.Linear(self.embed_vec_dim, self.out_channels * self.embed_vec_dim, bias=True)

        self.fc2 = nn.Linear(self.embed_vec_dim, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        #self.weight_norm()
            
    
    def weight_norm(self):
    		for module in self.modules():
    			if isinstance(module, nn.Linear):
    				weight = module.weight.data
    				module.weight.data = nn.init.kaiming_normal(weight, mode = 'fan_out', nonlinearity = None) 
    				   
    def forward(self, embed_vec):
        
        try:
        	#print(embed_vec.size())
        	N, E = embed_vec.size()
        	
        	weight = self.fc1(embed_vec).view(N, self.out_channels, self.embed_vec_dim)
        	#print(weight.size())
        	weight = weight.view(N, self.out_channels, self.embed_vec_dim)
        	weight = F.softmax(self.fc2(weight), -1)
        	#weight = (self.fc2(weight).view(N, self.out_channels*self.in_channels, self.kernel_size, self.kernel_size))
        	
        	return  weight.view(N, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        finally:	
        	pass
        	
class StaticHypernetwork_Parrallel(nn.Module):
    """Static hypernetwork described in https://arxiv.org/pdf/1609.09106.pdf"""
    def __init__(self, embed_vec_dim = 64, in_channels = 16, out_channels = 16, kernel_size = 3, bias=True):
        super(StaticHypernetwork_Parrallel, self).__init__()
        self.embed_vec_dim = embed_vec_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        #self.p_relu1 = nn.PReLU()
        #self.p_relu2 = nn.PReLU()
				
        #self.fc0 = nn.Linear(self.embed_vec_dim, self.embed_vec_dim, bias=True)
        #self.bn0 = nn.BatchNorm1d(self.out_channels * self.embed_vec_dim)
        self.fc1 = nn.Linear(self.embed_vec_dim, self.out_channels * self.embed_vec_dim, bias=True)
        #self.bn1 = nn.BatchNorm1d(self.out_channels * self.embed_vec_dim)
        #self.fc2 = nn.Linear(self.embed_vec_dim, self.in_channels * self.embed_vec_dim, bias=True)
        #self.bn2 = nn.BatchNorm1d(self.out_channels * self.in_channels * self.embed_vec_dim)
        self.fc2 = nn.Linear(self.embed_vec_dim, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        #
        #self.fc3 = nn.Linear(self.embed_vec_dim, self.in_channels, bias=True)
        self.fc3 = nn.Linear(self.embed_vec_dim, 1, bias=True)
        #self.bn3 = nn.BatchNorm1d(self.out_channels * self.in_channels * self.kernel_size * self.kernel_size)
        #self.fc4 = nn.Linear(self.out_channels * self.embed_vec_dim/64, self.out_channels * self.embed_vec_dim, bias=True)
        
        self.fc5 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size, self.in_channels * self.kernel_size * self.kernel_size / 9, bias=True)
        self.fc6 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size / 9, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        self.fc7 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        #self.fc5 = nn.Linear(self.kernel_size * self.kernel_size, self.kernel_size * self.kernel_size/3, bias=True)
        #self.fc6 = nn.Linear(self.kernel_size * self.kernel_size/3, self.kernel_size * self.kernel_size, bias=True)
        '''
        self.fc8 = nn.Linear(self.in_channels, self.in_channels / 4, bias=True)
        self.fc9 = nn.Linear(self.in_channels / 4, self.in_channels, bias=True)
        self.fc10 = nn.Linear(self.in_channels, self.in_channels, bias=True)
        '''
        self.fc8 = nn.Linear(self.out_channels, self.out_channels / 4, bias=True)
        self.fc9 = nn.Linear(self.out_channels / 4, self.out_channels, bias=True)
        self.fc10 = nn.Linear(self.out_channels, self.out_channels, bias=True)
        self.fc11 = nn.Linear(self.embed_vec_dim, 1, bias=True)
        #self.fc5 = nn.Linear(self.in_channels * self.kernel_size * self.kernel_size, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
        
        self.weight_norm()
        
    def weight_norm(self):
    		for module in self.modules():
    			if isinstance(module, nn.Linear):
    				weight = module.weight.data
    				module.weight.data = nn.init.kaiming_normal_(weight, mode = 'fan_in', nonlinearity = 'linear') 
    				   
    def forward(self, embed_vec):
        
        try:
        	#self.weight_norm(self.fc1)
        	#self.weight_norm(self.fc2)
        	N, E = embed_vec.size()
        	weight_1 = (self.fc1((embed_vec)))
        	weight_1 = weight_1.view(N, self.out_channels, self.embed_vec_dim)
        	weight_2 = (self.fc2(weight_1))
        	weight_2 = (F.sigmoid(self.fc6(self.fc5(weight_2))) * (weight_2)).view(N, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        	#weight_mean = self.fc11(weight_1)
        	'''
        	weight_mean = self.fc3(weight_1).squeeze(-1)
        	weight_mean = (F.sigmoid(self.fc9(self.fc8(weight_mean))) * (weight_mean))
        	weight_mean = weight_mean.view(N, self.out_channels, 1, 1, 1)
        	'''
        	return  weight_2, weight_mean.squeeze(-1)
        	return  weight_2
        finally:	
        	pass
        	
class StaticHypernetwork(nn.Module):
    """Static hypernetwork described in https://arxiv.org/pdf/1609.09106.pdf"""
    def __init__(self, embed_vec_dim = 64, in_channels = 16, out_channels = 16, kernel_size = 3, bias=True):
        super(StaticHypernetwork, self).__init__()
        self.embed_vec_dim = embed_vec_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.fc1 = nn.Linear(self.embed_vec_dim, self.out_channels * self.embed_vec_dim, bias=True)

        self.fc2 = nn.Linear(self.embed_vec_dim, self.in_channels * self.kernel_size * self.kernel_size, bias=True)
       
            
    
    def weight_norm(self, module):
    		weight = module.weight.data
    		module.weight.data = weight/(torch.norm(weight).view(1,1))    
    				   
    def forward(self, embed_vec):
        self.weight_norm(self.fc1)
        self.weight_norm(self.fc2)
        #weight = F.leaky_relu(self.in0(self.fc0(embed_vec).unsqueeze(0).unsqueeze(0))).squeeze()
        #weight = (self.fc0_0(embed_vec))
        #weight = (self.fc0_1(weight))
        try:
        	#weight = self.fc3(embed_vec).view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).clamp_(-1, 1)
        	weight = F.relu(self.fc1(embed_vec).view(self.out_channels, self.embed_vec_dim))
        	weight = self.fc2(weight).view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        	
        	return  weight
        finally:
        	pass
					
class HyperNetwork(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.randn(self.z_dim, self.out_size*self.f_size*self.f_size).cuda())
        self.b1 = Parameter(torch.randn(self.out_size*self.f_size*self.f_size).cuda())

        self.w2 = Parameter(torch.randn(self.z_dim, self.in_size*self.z_dim).cuda())
        self.b2 = Parameter(torch.randn(self.in_size*self.z_dim).cuda())
        
        self.kernel = 0.0
        
    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)
        
        #h_in = F.hardtanh(h_in)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
        
        #kernel = kernel.clamp_(-0.1, 0.1)
        #kernel = F.hardtanh(kernel)
        #kernel = self.InstanceNorm2d(kernel)

        return kernel





