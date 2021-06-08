import csv  
from matplotlib import pyplot as plt  
from datetime import datetime  
from pylab import *
import numpy as np

filename_1='/home/opt48/gnh/HyperNetworks/Plots/Abalation/Module_Contribution/ResNet18_hyper_clipped_elu/ResNet18_hyper_clipped_elu.csv'
filename_2='/home/opt48/gnh/HyperNetworks/Plots/Abalation/Module_Contribution/ResNet18_no_hyper_clipped_elu/ResNet18_no_hyper_clipped_elu.csv'
filename_3='/home/opt48/gnh/HyperNetworks/Plots/Abalation/Module_Contribution/ResNet18_hyper_satured/ResNet18_hyper_satured.csv'
filename_4='/home/opt48/gnh/HyperNetworks/Plots/Abalation/Module_Contribution/ResNet18_no_hyper_satured/ResNet18_no_hyper_satured.csv'
filename_5='/home/opt48/gnh/HyperNetworks/Plots/Abalation/Weight_L2_Comparision/Weight_L2.csv'

with open(filename_1) as f:
    reader_1=csv.reader(f)    
    header_row_1=next(reader_1)
    secound_row_1=next(reader_1)  
    epoch = []
    hyper_clipped_elu_prec1_train_1, hyper_clipped_elu_prec1_test_1 = [], [] 
    #satured_prec1_train, satured_prec5_train, satured_prec1_test, satured_prec5_test = [], [], [], []
       
    for row in reader_1:
	print(row)  
        epoch.append(int(row[0]))   
        hyper_clipped_elu_prec1_test_1.append(float(row[1]))
	hyper_clipped_elu_prec1_train_1.append(float(row[2]))   
	

with open(filename_2) as f:
    reader_2=csv.reader(f)    
    header_row_2=next(reader_2)
    secound_row_2=next(reader_2)  
    epoch = []
    no_hyper_clipped_elu_prec1_train_1, no_hyper_clipped_elu_prec1_test_1 = [], [] 
    #satured_prec1_train, satured_prec5_train, satured_prec1_test, satured_prec5_test = [], [], [], []
       
    for row in reader_2:
	print(row)  
        epoch.append(int(row[0]))   
        no_hyper_clipped_elu_prec1_test_1.append(float(row[1]))
	no_hyper_clipped_elu_prec1_train_1.append(float(row[2]))   
	

with open(filename_3) as f:
    reader_3=csv.reader(f)    
    header_row_3=next(reader_3)
    secound_row_3=next(reader_3)  
    epoch = []
    hyper_satured_prec1_train_1, hyper_satured_prec1_test_1 = [], [] 
    #satured_prec1_train, satured_prec5_train, satured_prec1_test, satured_prec5_test = [], [], [], []
       
    for row in reader_3:
	print(row)  
        epoch.append(int(row[0]))   
        hyper_satured_prec1_test_1.append(float(row[1]))
	hyper_satured_prec1_train_1.append(float(row[2]))   
	      

with open(filename_4) as f:
    reader_4=csv.reader(f)    
    header_row_4=next(reader_4)
    secound_row_4=next(reader_4)  
    epoch = []
    no_hyper_satured_prec1_train_1, no_hyper_satured_prec1_test_1 = [], [] 
    #satured_prec1_train, satured_prec5_train, satured_prec1_test, satured_prec5_test = [], [], [], []
       
    for row in reader_4:
	print(row)  
        epoch.append(int(row[0]))   
        no_hyper_satured_prec1_test_1.append(float(row[1]))
	no_hyper_satured_prec1_train_1.append(float(row[2]))   

with open(filename_5) as f:
    reader_5=csv.reader(f)    
    header_row_1=next(reader_5)
    secound_row_1=next(reader_5)  
    epoch_w = []
    hyper_clipped_elu_weight_L2, no_hyper_clipped_elu_weight_L2, hyper_satured_weight_L2, no_hyper_satured_L2 = [], [], [], [] 
    #satured_prec1_train, satured_prec5_train, satured_prec1_test, satured_prec5_test = [], [], [], []
       
    for row in reader_5:
	print(row)  
        epoch_w.append(int(row[0]))   
        hyper_clipped_elu_weight_L2.append(float(row[1]))
	no_hyper_clipped_elu_weight_L2.append(float(row[2]))   
	hyper_satured_weight_L2.append(float(row[3]))
	no_hyper_satured_L2.append(float(row[4]))

fig=plt.figure(dpi = 256, figsize=(24,6))

ax1 = plt.subplot(131)
ax1.plot(epoch,hyper_clipped_elu_prec1_test_1,c='red',alpha=1, linewidth = 1.5, linestyle='-', marker = '1', label = 'hyper_accumulation_clipped_elu_test')
ax1.plot(epoch,no_hyper_clipped_elu_prec1_test_1,c='springgreen',alpha=1, linewidth = 1.5,linestyle='-', marker = '1', label = 'classical_accumulation_clipped_elu_test')
ax1.plot(epoch,hyper_satured_prec1_test_1,c='blue',alpha=1, linewidth = 1.5, linestyle='-', marker = '1', label = 'hyper_accumulation_saturate_test')
ax1.plot(epoch, no_hyper_satured_prec1_test_1,c='orange',alpha=1, linewidth = 1.5, linestyle='-', marker = '1', label = 'classical_accumulation_saturate_test')


#ax2.plot(epoch,hyper_clipped_elu_prec1_train_1,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_clipped_elu_train')
#ax2.plot(epoch,no_hyper_clipped_elu_prec1_train_1,c='springgreen',alpha=1, linestyle='-', label = 'classical_accumulation_clipped_elu_train')
#ax2.plot(epoch,hyper_satured_prec1_train_1,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_satured_train')
#ax2.plot(epoch, no_hyper_satured_prec1_train_1,c='orange',alpha=1, linewidth = 2.5, linestyle='-', label = 'classical_accumulation_satured_train')

#plt.fill_between(epoch,hyper_clipped_elu_prec1_test_1,no_hyper_satured_prec1_test_1,facecolor='blue',alpha=0.2)
#plt.fill_between(epoch,hyper_clipped_elu_prec1_train_1, no_hyper_clipped_elu_prec1_train_1,facecolor='limegreen',alpha=0.4)
#plt.fill_between(epoch,no_hyper_clipped_elu_prec1_train_1, hyper_satured_prec1_train_1,facecolor='red',alpha=0.2)
#plt.fill_between(epoch,hyper_satured_prec1_train_1, no_hyper_satured_prec1_train_1,facecolor='blue',alpha=0.2)

plt.fill_between(epoch,hyper_clipped_elu_prec1_test_1, no_hyper_clipped_elu_prec1_test_1,facecolor='limegreen',alpha=0.4)
plt.fill_between(epoch,no_hyper_clipped_elu_prec1_test_1, hyper_satured_prec1_test_1,facecolor='red',alpha=0.2)
plt.fill_between(epoch,hyper_satured_prec1_test_1, no_hyper_satured_prec1_test_1,facecolor='blue',alpha=0.2)
#plt.fill_between(epoch,no_hyper_satured_prec1_test_1, 0,facecolor='lightsteelblue',alpha=0.4)  

#ax1.plot([5,0],[5,80], 'k--', linewidth = 1.)
#ax1.plot([0,195],[92.13,92.13], 'k--', linewidth = 1.)
#ax2.annotate("(epoch:195, best val prec1:92.13%)", xy = (195, 92.13), xytext = (85, 85), arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))



ax1.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax1.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax1.set_ylim(35, 90)
ax1.set_xlim(0, 27)
xminorLocator = MultipleLocator(2)
ax1.xaxis.set_minor_locator(xminorLocator)
plt.title('Top-1 Accuracy Comparision With Different Combinations',fontsize=12)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Classification Accuracy (%)',fontsize=18)
 
legend(loc = 'lower right')
plt.tick_params(axis='both',which='major',labelsize=16)

ax2 = plt.subplot(132)
ax2.plot(epoch,hyper_clipped_elu_prec1_test_1,c='red',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'hyper_accumulation_clipped_elu_test')
ax2.plot(epoch,no_hyper_clipped_elu_prec1_test_1,c='springgreen',alpha=1, linewidth = 0.5,linestyle='-', marker = '1', label = 'classical_accumulation_clipped_elu_test')
ax2.plot(epoch,hyper_satured_prec1_test_1,c='blue',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'hyper_accumulation_saturate_test')
ax2.plot(epoch, no_hyper_satured_prec1_test_1,c='orange',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'classical_accumulation_saturate_test')


ax2.plot(epoch,hyper_clipped_elu_prec1_train_1,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_clipped_elu_train')
ax2.plot(epoch,no_hyper_clipped_elu_prec1_train_1,c='springgreen',alpha=1, linestyle='-', label = 'classical_accumulation_clipped_elu_train')
ax2.plot(epoch,hyper_satured_prec1_train_1,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_saturate_train')
ax2.plot(epoch, no_hyper_satured_prec1_train_1,c='orange',alpha=1, linewidth = 2.5, linestyle='-', label = 'classical_accumulation_saturate_train')

#plt.fill_between(epoch,hyper_clipped_elu_prec1_test_1,no_hyper_satured_prec1_test_1,facecolor='blue',alpha=0.2)
plt.fill_between(epoch,hyper_clipped_elu_prec1_train_1, no_hyper_clipped_elu_prec1_train_1,facecolor='limegreen',alpha=0.4)
plt.fill_between(epoch,no_hyper_clipped_elu_prec1_train_1, hyper_satured_prec1_train_1,facecolor='red',alpha=0.2)
plt.fill_between(epoch,hyper_satured_prec1_train_1, no_hyper_satured_prec1_train_1,facecolor='blue',alpha=0.2)

plt.fill_between(epoch,hyper_clipped_elu_prec1_test_1, no_hyper_clipped_elu_prec1_test_1,facecolor='limegreen',alpha=0.4)
plt.fill_between(epoch,no_hyper_clipped_elu_prec1_test_1, hyper_satured_prec1_test_1,facecolor='red',alpha=0.2)
plt.fill_between(epoch,hyper_satured_prec1_test_1, no_hyper_satured_prec1_test_1,facecolor='blue',alpha=0.2)
#plt.fill_between(epoch,no_hyper_satured_prec1_test_1, 0,facecolor='peachpuff',alpha=0.4)  

#ax1.plot([195,195],[0,92.13], 'k--', linewidth = 1.)
#ax1.plot([0,195],[92.13,92.13], 'k--', linewidth = 1.)
#ax1.annotate("(epoch:195, best val prec1:92.13%)", xy = (195, 92.13), xytext = (85, 85), arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))



ax2.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax2.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax2.set_ylim(80, 100)
ax2.set_xlim(30, 250)
xminorLocator = MultipleLocator(20)
ax2.xaxis.set_minor_locator(xminorLocator)
plt.title('Top-1 Accuracy Comparision With Different Combinations',fontsize=12)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Classification Accuracy (%)',fontsize=18)
 
legend(loc = 'lower right')
plt.tick_params(axis='both',which='major',labelsize=16)

ax3 = plt.subplot(133)
ax3.plot(epoch_w,hyper_clipped_elu_weight_L2,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_clipped_elu')
ax3.plot(epoch_w,no_hyper_clipped_elu_weight_L2,c='springgreen',alpha=1, linewidth = 2.5,linestyle='-', label = 'classical_accumulation_clipped_elu')
ax3.plot(epoch_w,hyper_satured_weight_L2,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'hyper_accumulation_saturate')
ax3.plot(epoch_w, no_hyper_satured_L2,c='orange',alpha=1, linewidth = 2.5, linestyle='-', label = 'classical_accumulation_saturate')


#ax1.plot([195,195],[0,92.13], 'k--', linewidth = 1.)
#ax1.plot([0,195],[92.13,92.13], 'k--', linewidth = 1.)
#ax1.annotate("(epoch:195, best val prec1:92.13%)", xy = (195, 92.13), xytext = (45, 45), arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
#plt.fill_between(epoch,no_hyper_satured_L2,no_hyper_clipped_elu_weight_L2,facecolor='red',alpha=0.2)

ax3.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax3.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3)
xminorLocator = MultipleLocator(20)
ax3.xaxis.set_minor_locator(xminorLocator)

plt.title('Weight L2 Norm Comparision With Different Combinations',fontsize=12)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Weight L2 Norm',fontsize=18) 
legend(loc = 'lower right')
plt.tick_params(axis='both',which='major',labelsize=16)  

#fig.autofmt_xdate()
savefig('Abalation_Comparision.png') 
plt.show()
     
     
