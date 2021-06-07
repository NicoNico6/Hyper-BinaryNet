import math
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from resnet_blocks import ResNet18, ResNet_cifar10, ResNet_cifar10_share
from models.alexnet_binary import alexnet
from tensorboardX import SummaryWriter

torch.cuda.set_device(2)


def update_learning_rate(optimizer, epoch, lr_init):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        """ 
        lr = lr_init *  (0.9 ** (epoch // 70))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
def update_weight(weight_init, epoch, last_epoch):
        """
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        """ 
        weight = weight_init *  3 ** (epoch // 50)
        #weight = weight_init *  2 ** (math.log(epoch + 1)/math.log(3))
        #weight = weight_init *  math.log(epoch + 2) / math.log(2)
        
        return weight
        
########### Data Loader ###############

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10/', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='data/CIFAR10/', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=4, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#############################

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()



############
writer = SummaryWriter('logs')
net = ResNet18()
best_accuracy = 0.

#if args.resume:
#    ckpt = torch.load('./hypernetworks_cifar_paper.pth')
#    net.load_state_dict(ckpt['net'])
#    best_accuracy = ckpt['acc']

net.cuda()

restarts = [2, 4, 8, 16, 32, 64, 128, 256, 512]
learning_rate_Adam = 0.01
learning_rate_SGD = 1
weight_decay = 0.
#milestones = [168000 / (len(trainset) / 128), 336000 / (len(trainset) / 128), 400000 / (len(trainset) / 128), 450000 / (len(trainset) / 128), 550000 / (len(trainset) / 128), 600000 / (len(trainset) / 128)]
#max_iter = 1000000  / (len(trainset) /128 / 128)
milestones = [168000/10, 336000/10, 400000/10, 450000/10, 550000/10, 600000/10]
#milestones_adam = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000]
milestones_adam = [50, 100, 150, 180, 210, 240, 270, 300, 330, 360, 390, 410, 430, 450]
#milestones_adam = [40000, 80000, 120000, 160000, 220000, 240000]

milestones_sgd = [2000, 5000, 10000, 30000, 70000, 90000, 150000, 180000, 300000, 450000]
max_iter = 200000
optimizer = optim.Adam(net.parameters(), lr=learning_rate_Adam, weight_decay=weight_decay, amsgrad = False, betas = (0.9, 0.95))
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_adam, gamma=0.5)
criterion = nn.CrossEntropyLoss()
max_epochs = 500
total_iter = 0.0
epochs = 0.
print_freq = 10
weight = 1

while epochs < max_epochs:
  
    avg_loss = 0.0
    running_loss = 0.0
    running_l2_loss = 0.0
    train_correct = 0.
    train_total = 0.
    rate = 1.0
    L2 = 0.0
    
    lr_scheduler.step()
    
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
      
            module.momentum = max(0.1, 0.95 - epochs/max_epochs)
    #net.bn1.momentum = 0.1  
   
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        
        net.train()
        

        outputs = net(inputs)
        _, predicted = torch.max(outputs.cpu().data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels.data.cpu()).sum()

        loss = criterion(outputs, labels)
        loss_l2 = L2 * 0.0000005
        #loss_l2 = L2 
        loss_sum = loss*2
        
        optimizer.zero_grad()
        
        loss_sum.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        avg_loss += loss.data[0]
        
        if i % print_freq == 0 and i != 0:
            print("[Epoch %d, Total Iterations %6d] Loss: %.4f L2: %.4f Lr: %.8f, weight: %.3f, momentum: %.4f, best test accuracy: %.4f" % (epochs + 1, total_iter + 1, running_loss/print_freq, running_l2_loss/print_freq, optimizer.param_groups[0]['lr'], weight, net.bn1.momentum, best_accuracy))
            running_loss = 0.0
            running_l2_loss = 0.0

        total_iter += 1.0

    
    avg_loss = avg_loss/i
    
        
    
    test_correct = 0.
    test_total = 0.
    test_loss = 0.
    with torch.no_grad():
      for n, tdata in enumerate(testloader):
        timages, tlabels = tdata
        tlabels = Variable(tlabels.cuda())
        net.eval()
        toutputs = net(Variable(timages.cuda()))
        _, predicted = torch.max(toutputs.cpu().data, 1)
        test_total += tlabels.size(0)
        test_correct += (predicted == tlabels.cpu().data).sum()
        test_loss += criterion(toutputs, tlabels).data.cpu()
    
    test_loss = test_loss/n
    train_accuracy = (100.0 * train_correct).float() / train_total
    test_accuracy = (100.0 * test_correct).float() / test_total
    
    epochs += 1
    torch.cuda.empty_cache()
    
    print('After epoch %d, train accuracy: %.4f, test accuracy: %.4f, test loss: %4f' % (epochs, train_accuracy, test_accuracy, test_loss))
    
   
    
    if test_accuracy > best_accuracy:
        print('Saving model...')
        state = {
            'net': net.state_dict(),
            'acc': test_accuracy
        }
        torch.save(state, './hypernetworks_cifar_paper.pth')
        best_accuracy = test_accuracy
    
    writer.add_scalars('ResNet/CIFAR10_Loss_1', {'train_loss':avg_loss, 'test_loss':test_loss}, epochs)
    writer.add_scalars('ResNet/CIFAR10_Acc_1', {'train_acc':train_accuracy, 'test_acc':test_accuracy}, epochs)
    writer.add_scalars('ResNet/CIFAR10_best_acc_1', {'best_acc':best_accuracy}, epochs)

print('Finished Training')

