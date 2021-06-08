import torch
import numpy as np

 
#from models.alexnet_binary import *
#from models.resnet_binary import *
#from models.vgg_binary import *
#from models.wide_resnet_binary import *
#from models.modules import *

from matplotlib import pyplot as plt 
from tensorboardX import SummaryWriter

def kernel_heatmap(model, state_dict_dir):
    modules_num = 0
    
    model = model.load_state_dict(torch.load(state_dict_dir))
    
    for name, module in model.named_module():
      if name == "BinarizeHyperConv2d":
        modules_num += 1
        kernels = module.Weight.numpy()     #N_out x N_in x 3 x 3


def drow_heatmap(x, y):
    
    
    #plt.subplot(131)
    print(x)
    print(y)
    hist, xedges, yedges = np.histogram2d(x,y, bins = (3,3))
    print(hist)
    print(xedges)
    #X,Y = np.meshgrid(xedges, yedges)
    plt.imshow(hist)
    #plt.grid()
    plt.colorbar()
    plt.show()
    
    #plt.subplot(132)
    #plt.imshow(hist, interpolation = 'nearest')
    #plt.grid(True)
    #plt.colorbar()
    #plt.show()
    
    #plt.subplot(133)
    #plt.hist2d(X, Y, bins = 10)
    #plt.grid()
    #plt.colorbar()
    #plt.show()
    
    
if __name__ == "__main__":
    mean =[0, 0]
    var = [0,1], [1,0]
    x, y = np.random.multivariate_normal(mean, var, 3).T
    
    drow_heatmap(x, y)
     
    
    
            
        
    
