import csv  
from matplotlib import pyplot as plt  
from datetime import datetime  
from pylab import *
import numpy as np

filename_1='/home/opt48/gnh/HyperNetworks/Plots/Abalation/Weight_L2_Comparision/Weight_L2.csv'


with open(filename_1) as f:
    reader_1=csv.reader(f)    
    header_row_1=next(reader_1)
    secound_row_1=next(reader_1)  
    epoch = []
    hyper_clipped_elu_weight_L2, no_hyper_clipped_elu_weight_L2, hyper_saturate_weight_L2, no_hyper_saturate_L2 = [], [], [], [] 
    #saturate_prec1_train, saturate_prec5_train, saturate_prec1_test, saturate_prec5_test = [], [], [], []
       
    for row in reader_1:
	print(row)  
        epoch.append(int(row[0]))   
        hyper_clipped_elu_weight_L2.append(float(row[1]))
	no_hyper_clipped_elu_weight_L2.append(float(row[2]))   
	hyper_saturate_weight_L2.append(float(row[3]))
	no_hyper_saturate_L2.append(float(row[4]))
 	


fig=plt.figure(dpi = 80, figsize=(8,8))
ax1 = plt.subplot(111)
ax1.plot(epoch,hyper_clipped_elu_weight_L2,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_clipped_elu')
ax1.plot(epoch,no_hyper_clipped_elu_weight_L2,c='springgreen',alpha=1, linewidth = 2.5,linestyle='-', label = 'classical_accumulation_clipped_elu')
ax1.plot(epoch,hyper_saturate_weight_L2,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_saturate')
ax1.plot(epoch, no_hyper_saturate_L2,c='orange',alpha=1, linewidth = 2.5, linestyle='-', label = 'classical_accumulation_saturate')


#ax1.plot([195,195],[0,92.13], 'k--', linewidth = 1.)
#ax1.plot([0,195],[92.13,92.13], 'k--', linewidth = 1.)
#ax1.annotate("(epoch:195, best val prec1:92.13%)", xy = (195, 92.13), xytext = (45, 45), arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
#plt.fill_between(epoch,no_hyper_saturate_L2,no_hyper_clipped_elu_weight_L2,facecolor='red',alpha=0.2)

ax1.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax1.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3)
xminorLocator = MultipleLocator(20)
ax1.xaxis.set_minor_locator(xminorLocator)

plt.title('Weight L2 Norm Comparision With Different Combinations',fontsize=15)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Weight L2 Norm',fontsize=18) 
legend(loc = 'lower right')
plt.tick_params(axis='both',which='major',labelsize=16)  
#fig.autofmt_xdate()
savefig('Weight_L2_Comparision.png') 
plt.show()
     
     
