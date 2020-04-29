from os import makedirs
import json

import torch
import torchvision
from torchvision.models.resnet import resnet18, resnet152 #ResNet, BasicBlock
from torchvision.models import mobilenet_v2 #ResNet, BasicBlock
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


#def resnet18(num_classes):
#    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc


def _trainval(trainloader, testloader, net, optimizer, lr_scheduler, criterion, reporter, num_epochs, save_dir, save_period):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if 0: #i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        #train_acc = test(net, trainloader)
        val_acc = test(net, testloader)
        reporter(epoch=epoch, accuracy=val_acc) #, train_acc=train_acc)
        lr_scheduler.step()
        #print('Current LR ', lr_scheduler.get_lr())

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            torch.save(net.state_dict(), '%s/epoch-%d.pth'%(save_dir, epoch))
    if save_period and save_dir:
        torch.save(net.state_dict(), '%s/epoch-%d.pth'%(save_dir, num_epochs-1))


def trainval(args, reporter):
    root = '/home/yichaoxiong/.torch/datasets/cifar10'
    # runtime
    runtime = args.runtime
    batch_size = runtime.batch_size
    num_workers = runtime.num_workers
    save_dir = runtime.save_dir
    save_period = runtime.save_period
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    save_dir = runtime.save_dir + '/task_%d' % args.task_id
    makedirs(save_dir)
    args_fp = '%s/args.json' % save_dir
    with open(args_fp, 'w') as fd:
        json.dump(args, fd)

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    # model
    model_name = args.model.name
    net = eval(model_name)(num_classes=10).cuda()
    if args.model.resume_from:
        net.load_state_dict(torch.load(args.model.resume_from))
    criterion = nn.CrossEntropyLoss()
    cfg = args.optimizer
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd, momentum=cfg.momentum)
    cfg = args.lr_scheduler
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg.milestones)
    num_epochs = cfg.num_epochs

    _trainval(trainloader, testloader, net, optimizer, lr_scheduler, criterion, reporter, num_epochs, save_dir, save_period)
