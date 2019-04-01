#importing neccesary things
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import glob
import matplotlib.pyplot as plt
import scipy.misc
import time
import os
from os import listdir
from os.path import isfile, join
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))

class Enhancer(nn.Module):
    def __init__(self, name):
        super(Enhancer, self).__init__()
        self.name = name
        self.enhancer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(inplace=True),                                    
                                      )
        self.enhancer2= nn.Sequential(  
                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1),
                                      nn.Sigmoid(),
                                      )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.enhancer1(x)
        x= self.enhancer2(x)

        
        return x

    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
    
        return x + residual    
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self, name):
        super(Discriminator, self).__init__()
        self.name = name
        self.discriminator = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3),
                                          nn.LeakyReLU( inplace=True),
                                          nn.Conv2d(in_channels=64,out_channels=128,kernel_size=7,padding=3),
                                          nn.BatchNorm2d(128), 
                                          nn.LeakyReLU( inplace=True),
                                          nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1), 
                                          )
    def forward(self, x):
        x = self.discriminator(x)
        x = F.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        x = x.view(-1)
        return x

def validate_enhancer(model, path, prev_model_path, batch_size=1, image_num=None):
    
    with torch.no_grad():    

        print("Validating enhancer...")
        sys.stdout.write('000%')
        sys.stdout.flush()
        model.eval()
        transform=transforms.Compose([transforms.ToTensor()])
        dataset=torchvision.datasets.ImageFolder(path, transform=transform)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        if(image_num == None):
            image_num = len(dataset)
        if torch.cuda.device_count() > 1:
            device_ids = [i for i in range(torch.cuda.device_count())]
            torch.cuda.empty_cache()
            name = model.name
            model = nn.DataParallel(model, device_ids)
            model.name = name
        if(prev_model_path != ""):
            model.load_state_dict(torch.load(prev_model_path,) )#map_location=torch.device('cpu')))       
        model.to(device)
        
        criterion = nn.MSELoss()
        total_loss = 0
        fake_images_count = 0
        
        for i, images in enumerate(dataloader, 0):
            if(fake_images_count >= image_num):
                break
            data = F.pad(images[0], (0,images[0].shape[3]%2,0,images[0].shape[2]%2), "constant", 0)
            data = data.to(device)
            puffer = (model(F.interpolate(data, scale_factor=0.5)));
            total_loss += criterion(puffer, data)
            fake_images_count += puffer.shape[0]
            sys.stdout.write('\r\r\r\r{:0>3d}%'.format(int((fake_images_count)/image_num*100)))
            sys.stdout.flush()

            puffer= None
            del puffer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if fake_images_count >= len(dataset):
                print('true')
                break
            
    print("Loss: ", total_loss/i)
    torch.set_grad_enabled(True)
    del model
    torch.cuda.empty_cache()
    return (total_loss/i)

def validate_discriminator(model, discriminator, path, prev_model_path, prev_discriminator_path, batch_size=1, image_num=None):
    
    with torch.no_grad():    

        print("Validating enhancer...")
        sys.stdout.write('000%')
        sys.stdout.flush()
        model.eval()
        transform=transforms.Compose([transforms.ToTensor()])
        dataset=torchvision.datasets.ImageFolder(path, transform=transform)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        if(image_num == None):
            image_num = len(dataset)
        if torch.cuda.device_count() > 1:
            device_ids = [i for i in range(torch.cuda.device_count())]
            torch.cuda.empty_cache()
            name = model.name
            model = nn.DataParallel(model, device_ids)
            model.name = name
            name = discriminator.name
            discriminator = nn.DataParallel(discriminator, device_ids)
            discriminator.name = name
        if(prev_model_path != ""):
            model.load_state_dict(torch.load(prev_model_path,) )#map_location=torch.device('cpu')))       
        model.to(device)
        if(prev_discriminator_path != ""):
            discriminator.load_state_dict(torch.load(prev_discriminator_path,) )#map_location=torch.device('cpu')))       
        discriminator.to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        fake_images_count = 0
        
        for i, images in enumerate(dataloader, 0):
            if(fake_images_count >= image_num):
                break
            data = F.pad(images[0], (0,images[0].shape[3]%2,0,images[0].shape[2]%2), "constant", 0)
            data = data.to(device)
            puffer = discriminator(model(F.interpolate(data, scale_factor=0.5)));
            total_loss += criterion(puffer, torch.ones(puffer.shape[0]).to(device))
            fake_images_count += puffer.shape[0]
            sys.stdout.write('\r\r\r\r{:0>3d}%'.format(int((fake_images_count)/image_num*100)))
            sys.stdout.flush()

            puffer= None
            del puffer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if fake_images_count >= len(dataset):
                print('true')
                break
            
    print("Loss: ", total_loss/i)
    torch.set_grad_enabled(True)
    del model
    torch.cuda.empty_cache()
    return (total_loss/i)