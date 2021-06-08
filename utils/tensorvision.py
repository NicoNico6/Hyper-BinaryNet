import torch
from tensorboardX import SummaryWriter

if __name__ == "__main__":
	 
	 writer = SummaryWriter(log_dir = 'logs')
	 for i in range(100):
	 		y1 = i**2 * torch.randn(1)
	 		y2 = i**3
	 		writer.add_scalars('data/scalar_group',{'y1':y1,
	 																					'y2':y2}, i)
	 
	 