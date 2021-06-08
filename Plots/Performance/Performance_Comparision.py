import csv  
from matplotlib import pyplot as plt  
from datetime import datetime  
from pylab import *
import numpy as np

filename_1='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet18_1_1_CIFAR10/ResNet18_1_1_CIFAR10.csv'
filename_2='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet18_1_32_CIFAR10/ResNet18_1_32_CIFAR10.csv'
filename_3='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet18_1_1_CIFAR100/ResNet18_1_1_CIFAR100.csv'
filename_4='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet18_1_32_CIFAR100/ResNet18_1_32_CIFAR100.csv'

filename_5='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet34_1_1_CIFAR10/ResNet34_1_1_CIFAR10.csv'
filename_6='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet34_1_32_CIFAR10/ResNet34_1_32_CIFAR10.csv'
filename_7='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet34_1_1_CIFAR100/ResNet34_1_1_CIFAR100.csv'
filename_8='/home/opt48/gnh/HyperNetworks/Plots/Performance/ResNet34_1_32_CIFAR100/ResNet34_1_32_CIFAR100.csv'

filename_9='/home/opt48/gnh/HyperNetworks/Plots/Performance/VGG_Small_1_1_CIFAR10/VGG_Small_1_1_CIFAR10.csv'
filename_10='/home/opt48/gnh/HyperNetworks/Plots/Performance/VGG_Small_1_32_CIFAR10/VGG_Small_1_32_CIFAR10.csv'
filename_11='/home/opt48/gnh/HyperNetworks/Plots/Performance/VGG_Small_1_1_CIFAR100/VGG_Small_1_1_CIFAR100.csv'
filename_12='/home/opt48/gnh/HyperNetworks/Plots/Performance/VGG_Small_1_32_CIFAR100/VGG_Small_1_32_CIFAR100.csv'

with open(filename_1) as f:
    reader_1=csv.reader(f)    
    header_row_1=next(reader_1)
    secound_row_1=next(reader_1)  
    epoch = []

    resnet18_1_1_cifar10_train, resnet18_1_1_cifar10_test = [], [] 
       
    for row in reader_1:
	print(row)  
        epoch.append(int(row[0]))   
        resnet18_1_1_cifar10_train.append(float(row[2]))
	resnet18_1_1_cifar10_test.append(float(row[1]))   
	

with open(filename_2) as f:
    reader_2=csv.reader(f)    
    header_row_2=next(reader_2)
    secound_row_2=next(reader_2)  
    epoch = []
    resnet18_1_32_cifar10_train, resnet18_1_32_cifar10_test = [], [] 
       
    for row in reader_2:
	print(row)  
        epoch.append(int(row[0]))   
        resnet18_1_32_cifar10_train.append(float(row[2]))
	resnet18_1_32_cifar10_test.append(float(row[1]))   
	

with open(filename_3) as f:
    reader_3=csv.reader(f)    
    header_row_3=next(reader_3)
    secound_row_3=next(reader_3)  
    epoch = []
    resnet18_1_1_cifar100_train, resnet18_1_1_cifar100_test = [], [] 
       
    for row in reader_3:
	print(row)  
        epoch.append(int(row[0]))   
        resnet18_1_1_cifar100_train.append(float(row[2]))
	resnet18_1_1_cifar100_test.append(float(row[1]))   
	      

with open(filename_4) as f:
    reader_4=csv.reader(f)    
    header_row_4=next(reader_4)
    secound_row_4=next(reader_4)  
    epoch = []
    resnet18_1_32_cifar100_train, resnet18_1_32_cifar100_test = [], [] 
       
    for row in reader_4:
	print(row)  
        epoch.append(int(row[0]))   
        resnet18_1_32_cifar100_train.append(float(row[2]))
	resnet18_1_32_cifar100_test.append(float(row[1]))   

with open(filename_5) as f:
    reader_5=csv.reader(f)    
    header_row_1=next(reader_5)
    secound_row_1=next(reader_5)  
    epoch_w = []
    resnet34_1_1_cifar10_train, resnet34_1_1_cifar10_test = [], [] 
       
    for row in reader_5:
	print(row)  
        resnet34_1_1_cifar10_train.append(float(row[2]))
	resnet34_1_1_cifar10_test.append(float(row[1]))


with open(filename_6) as f:
    reader_6=csv.reader(f)    
    header_row_6=next(reader_6)
    secound_row_6=next(reader_6)  
    epoch = []
    resnet34_1_32_cifar10_train, resnet34_1_32_cifar10_test = [], [] 
       
    for row in reader_6:
	print(row)  
        epoch.append(int(row[0]))   
        resnet34_1_32_cifar10_train.append(float(row[2]))
	resnet34_1_32_cifar10_test.append(float(row[1]))   
	

with open(filename_7) as f:
    reader_7=csv.reader(f)    
    header_row_7=next(reader_7)
    secound_row_7=next(reader_7)  
    epoch = []
    resnet34_1_1_cifar100_train, resnet34_1_1_cifar100_test = [], [] 
       
    for row in reader_7:
	print(row)  
        epoch.append(int(row[0]))   
        resnet34_1_1_cifar100_train.append(float(row[2]))
	resnet34_1_1_cifar100_test.append(float(row[1]))   
	      

with open(filename_8) as f:
    reader_8=csv.reader(f)    
    header_row_8=next(reader_8)
    secound_row_8=next(reader_8)  
    epoch = []
    resnet34_1_32_cifar100_train, resnet34_1_32_cifar100_test = [], [] 
       
    for row in reader_8:
	print(row)  
        epoch.append(int(row[0]))   
        resnet34_1_32_cifar100_train.append(float(row[2]))
	resnet34_1_32_cifar100_test.append(float(row[1]))   


with open(filename_9) as f:
    reader_9=csv.reader(f)    
    header_row_9=next(reader_9)
    secound_row_9=next(reader_9)  
    epoch_w = []
    vgg_small_1_1_cifar10_train, vgg_small_1_1_cifar10_test = [], [] 
       
    for row in reader_9:
	print(row)  
        vgg_small_1_1_cifar10_train.append(float(row[2]))
	vgg_small_1_1_cifar10_test.append(float(row[1]))


with open(filename_10) as f:
    reader_10=csv.reader(f)    
    header_row_10=next(reader_10)
    secound_row_10=next(reader_10)  
    epoch = []
    vgg_small_1_32_cifar10_train, vgg_small_1_32_cifar10_test = [], [] 
       
    for row in reader_10:
	print(row)  
        epoch.append(int(row[0]))   
        vgg_small_1_32_cifar10_train.append(float(row[2]))
	vgg_small_1_32_cifar10_test.append(float(row[1]))   
	

with open(filename_11) as f:
    reader_11=csv.reader(f)    
    header_row_11=next(reader_11)
    secound_row_11=next(reader_11)  
    epoch = []
    vgg_small_1_1_cifar100_train, vgg_small_1_1_cifar100_test = [], [] 
       
    for row in reader_11:
	print(row)  
        epoch.append(int(row[0]))   
        vgg_small_1_1_cifar100_train.append(float(row[2]))
	vgg_small_1_1_cifar100_test.append(float(row[1]))   
	      

with open(filename_12) as f:
    reader_12=csv.reader(f)    
    header_row_12=next(reader_12)
    secound_row_8=next(reader_12)  
    epoch = []
    vgg_small_1_32_cifar100_train, vgg_small_1_32_cifar100_test = [], [] 
       
    for row in reader_12:
	print(row)  
        epoch.append(int(row[0]))   
        vgg_small_1_32_cifar100_train.append(float(row[2]))
	vgg_small_1_32_cifar100_test.append(float(row[1]))
  
fig=plt.figure(dpi = 256, figsize=(15,12))


######################################
######################################

ax1 = plt.subplot(221)
ax1.plot(epoch,resnet18_1_1_cifar10_train,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'resnet18_1_1_cifar10_train')
ax1.plot(epoch,resnet34_1_1_cifar10_train,c='springgreen',alpha=1, linewidth = 2.5,linestyle='-', label = 'resnet34_1_1_cifar10_train')
ax1.plot(epoch,vgg_small_1_1_cifar10_train,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'convnet_1_1_cifar10_train')

ax1.plot(epoch,resnet18_1_1_cifar10_test,c='red',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'resnet18_1_1_cifar10_test')
ax1.plot(epoch,resnet34_1_1_cifar10_test,c='springgreen',alpha=1, linewidth = 0.5,linestyle='-', marker = '1', label = 'resnet34_1_1_cifar10_test')
ax1.plot(epoch,vgg_small_1_1_cifar10_test,c='blue',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'vgg_small_1_1_cifar10_test')

ax1.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax1.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 

ax1.set_ylim(85, 100)
xminorLocator = MultipleLocator(20)
ax1.xaxis.set_minor_locator(xminorLocator)

plt.fill_between(epoch, resnet34_1_1_cifar10_train, resnet18_1_1_cifar10_train,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, vgg_small_1_1_cifar10_train, resnet18_1_1_cifar10_train, facecolor='orange',alpha=0.2)

plt.fill_between(epoch, resnet34_1_1_cifar10_test, resnet18_1_1_cifar10_test,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, resnet18_1_1_cifar10_test, vgg_small_1_1_cifar10_test,facecolor='orange',alpha=0.2)



plt.title('Top-1 Accuracy On CIFAR10 with 1/1',fontsize=15)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Classification Accuracy (%)',fontsize=18)
plt.tick_params(axis='both',which='major',labelsize=16)
legend(loc = 'lower right')

######################################
######################################

ax2 = plt.subplot(222)
ax2.plot(epoch,resnet18_1_32_cifar10_train,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'resnet18_1_32_cifar10_train')
ax2.plot(epoch,resnet34_1_32_cifar10_train,c='springgreen',alpha=1, linewidth = 2.5,linestyle='-', label = 'resnet34_1_32_cifar10_train')
ax2.plot(epoch,vgg_small_1_32_cifar10_train,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'convnet_1_32_cifar10_train')

ax2.plot(epoch,resnet18_1_32_cifar10_test,c='red',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'resnet18_1_32_cifar10_test')
ax2.plot(epoch,resnet34_1_32_cifar10_test,c='springgreen',alpha=1, linewidth = 0.5,linestyle='-', marker = '1', label = 'resnet34_1_32_cifar10_test')
ax2.plot(epoch,vgg_small_1_32_cifar10_test,c='blue',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'convnet_1_32_cifar10_test')

ax2.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax2.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 

ax2.set_ylim(85, 100)
xminorLocator = MultipleLocator(20)
ax2.xaxis.set_minor_locator(xminorLocator)

plt.fill_between(epoch, resnet34_1_32_cifar10_train, resnet18_1_32_cifar10_train,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, vgg_small_1_32_cifar10_train, resnet18_1_32_cifar10_train, facecolor='orange',alpha=0.2)

plt.fill_between(epoch, resnet34_1_32_cifar10_test, resnet18_1_32_cifar10_test,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, resnet18_1_32_cifar10_test, vgg_small_1_32_cifar10_test,facecolor='orange',alpha=0.2)

plt.title('Top-1 Accuracy On CIFAR10 with 1/32',fontsize=15)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Classification Accuracy (%)',fontsize=18)
plt.tick_params(axis='both',which='major',labelsize=16)
legend(loc = 'lower right')

######################################
######################################

ax3 = plt.subplot(223)
ax3.plot(epoch,resnet18_1_1_cifar100_train,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'resnet18_1_1_cifar100_train')
ax3.plot(epoch,resnet34_1_1_cifar100_train,c='springgreen',alpha=1, linewidth = 2.5,linestyle='-', label = 'resnet34_1_1_cifar100_train')
ax3.plot(epoch,vgg_small_1_1_cifar100_train,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'convnet_1_1_cifar100_train')

ax3.plot(epoch,resnet18_1_1_cifar100_test,c='red',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'resnet18_1_1_cifar100_test')
ax3.plot(epoch,resnet34_1_1_cifar100_test,c='springgreen',alpha=1, linewidth = 0.5,linestyle='-', marker = '1', label = 'resnet34_1_1_cifar100_test')
ax3.plot(epoch,vgg_small_1_1_cifar100_test,c='blue',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'vgg_small_1_1_cifar100_test')

ax3.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax3.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 

ax3.set_ylim(50, 100)
xminorLocator = MultipleLocator(20)
ax3.xaxis.set_minor_locator(xminorLocator)

plt.fill_between(epoch, resnet34_1_1_cifar100_train, resnet18_1_1_cifar100_train,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, vgg_small_1_1_cifar100_train, resnet18_1_1_cifar100_train, facecolor='orange',alpha=0.2)

plt.fill_between(epoch, resnet34_1_1_cifar100_test, resnet18_1_1_cifar100_test,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, resnet18_1_1_cifar100_test, vgg_small_1_1_cifar100_test,facecolor='orange',alpha=0.2)

plt.title('Top-1 Accuracy On CIFAR100 with 1/1',fontsize=15)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Classification Accuracy (%)',fontsize=18)
plt.tick_params(axis='both',which='major',labelsize=16)
legend(loc = 'lower right')


######################################
######################################

ax4 = plt.subplot(224)
ax4.plot(epoch,resnet18_1_32_cifar100_train,c='red',alpha=1, linewidth = 2.5, linestyle='-', label = 'resnet18_1_32_cifar100_train')
ax4.plot(epoch,resnet34_1_32_cifar100_train,c='springgreen',alpha=1, linewidth = 2.5,linestyle='-', label = 'resnet34_1_32_cifar100_train')
ax4.plot(epoch,vgg_small_1_32_cifar100_train,c='blue',alpha=1, linewidth = 2.5, linestyle='-', label = 'convnet_1_32_cifar100_train')

ax4.plot(epoch,resnet18_1_32_cifar100_test,c='red',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'resnet18_1_32_cifar100_test')
ax4.plot(epoch,resnet34_1_32_cifar100_test,c='springgreen',alpha=1, linewidth = 0.5,linestyle='-', marker = '1', label = 'resnet34_1_32_cifar100_test')
ax4.plot(epoch,vgg_small_1_32_cifar100_test,c='blue',alpha=1, linewidth = 0.5, linestyle='-', marker = '1', label = 'convnet_1_32_cifar100_test')

ax4.xaxis.grid(which='minor', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 
ax4.yaxis.grid(which='major', color = 'b', linestyle = '--', linewidth = 1, alpha = 0.3) 

ax4.set_ylim(50, 100)
xminorLocator = MultipleLocator(20)
ax4.xaxis.set_minor_locator(xminorLocator)

plt.fill_between(epoch, resnet34_1_32_cifar100_train, resnet18_1_32_cifar100_train,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, vgg_small_1_32_cifar100_train, resnet18_1_32_cifar100_train, facecolor='orange',alpha=0.2)

plt.fill_between(epoch, resnet34_1_32_cifar100_test, resnet18_1_32_cifar100_test,facecolor='blue',alpha=0.2)
plt.fill_between(epoch, resnet18_1_32_cifar100_test, vgg_small_1_32_cifar100_test,facecolor='orange',alpha=0.2)

plt.title('Top-1 Accuracy On CIFAR100 with 1/32',fontsize=15)  
plt.xlabel('Epochs',fontsize=18)  
plt.ylabel('Classification Accuracy (%)',fontsize=18)
plt.tick_params(axis='both',which='major',labelsize=16)
legend(loc = 'lower right')


savefig('Training_Procedure.png') 
plt.show()
     
     
