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
import imageio
imageio.plugins.ffmpeg.download()
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

    
def video_enhancer(input_path, model_path, output_path, batch_size=1)
    vid = imageio.get_reader(input_path,  'ffmpeg')
    input_tensor = []
    for i in range(len(vid)-1):
        image = vid.get_data(i)
        image = torch.from_numpy(image/255).to(torch.float).permute(2,0,1)
        input_tensor.append(image)
    input_tensor = torch.stack(input_tensor)
    dataset = data_utils.TensorDataset(input_tensor)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    enhancer = Enhancer("enhancer6")
    if torch.cuda.device_count() > 1:
        device_ids = [i for i in range(torch.cuda.device_count())]    
        name = enhancer.name
        enhancer = nn.DataParallel(enhancer, device_ids)
        enhancer.name = name
    enhancer.load_state_dict(torch.load("enhancer42l_bs32_lr9e-05_epoch4/enhancer42l_bs32_lr9e-05_epoch4"))

    new_vid = []
    enhancer.to(device)
    for i, img in enumerate(dataloader, 0):
        img = img[0]
        img = img.to(device)
        new_vid.append(Variable(enhancer(img),volatile=True).detach().cpu())

    writer = imageio.get_writer(output_path, fps=vid.get_meta_data()['fps'])
    for i in range(new_vid.shape[0]):
        writer.append_data(new_vid[i])
    writer.close()
    
    return