#importing neccesary things
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
import glob
import matplotlib.pyplot as plt
import time
import os
from os import listdir
from os.path import isfile, join

def load_data(path, do_padding=True, divisor=1):
    if not os.path.exists(path):
        return -1
    image_numpy_list = []
    image_tensor_list = []
    image_paths = glob.glob("{}/*.jpg".format(path))
    max_length, max_width = 0, 0
    for i in range(len(image_paths)):
            image = plt.imread(image_paths[i])
            image_numpy_list.append(image)
            image_tensor_list.append(torch.from_numpy((image/255)).type(torch.FloatTensor).permute(2,0,1))
            max_length = image.shape[0] if (image.shape[0] > max_length) else max_length
            max_width = image.shape[1] if (image.shape[1] > max_width) else max_width
    if(do_padding):
        max_length += (divisor - max_length % divisor) if (max_length % divisor != 0) else 0
        max_width += (divisor - max_width % divisor) if (max_width % divisor != 0) else 0
        image_tensor = torch.zeros(len(image_tensor_list), 3, max_length, max_width)
        for i in range(len(image_tensor_list)):
            image = image_tensor_list[i]
            pad_length = max_length - image.shape[1]
            pad_width = max_width - image.shape[2]
            pad_up = int(pad_length/2)
            pad_down = int(pad_length/2 if (pad_length % 2 == 0) else pad_length/2 + 1)
            pad_left = int(pad_width/2)
            pad_right = int(pad_width/2 if (pad_width % 2 == 0) else pad_width/2 + 1)
            image_tensor[i] =  F.pad(input=image, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=0)
        return image_numpy_list, image_tensor
    else:
        return image_numpy_list, image_tensor_list

class Enhancer(nn.Module):
    def __init__(self, name):
        super(Enhancer, self).__init__()
        self.name = name
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, padding=5),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(in_channels=64, out_channels=196, kernel_size=5, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2)
                                    )
        self.enhancer = nn.Sequential(nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=5,stride=2,output_padding=1,padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(in_channels=196, out_channels=64, kernel_size=5, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=11, stride=2,output_padding=1,padding=5),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=11, padding=5),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=15, stride=2,output_padding=1,padding=7),
                                     nn.Sigmoid(),
                                    )

    def forward(self, x):
        x = self.encoder(x)
        x = self.enhancer(x)
        return x
    
def train_enhancer(model, input_images, target_images, batch_size=1, learning_rate=1e-3, num_epochs=5, device="cpu"):
    #move data and model to specified device
    model.to(device)

    if not os.path.exists("{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs)):
        os.makedirs("{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs))   
    
    #create dataloader for images
    dataset = data_utils.TensorDataset(input_images, target_images)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    #seeds and optimizer
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/batch_size, weight_decay=1e-5)
    
    #list that contains losses and outputs
    losses = []
    
    start_time = time.time()#time stamp
    

    # Zero the parameter gradients
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(dataloader, 0):
            img, target = data
            img = img.to(device)
            target = target.to(device)
            
            # Forward pass, backward pass, and optimize
            recon = model(img)
            loss = criterion(recon, target)
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        #empty cache to save up ram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        total_loss /= (i+1)
        
        #save outputs and stats for this epoch
        print('Epoch: {}, Loss: {:.4f}, Time: {}min {}sec'.format(epoch+1, float(total_loss), int((time.time()-start_time)/60), int((time.time()-start_time)%60)))
        losses.append(total_loss)
        torch.save(model.state_dict(), "{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1))

        
    #save stats into file
    np.savetxt("{}_train_losses.csv".format("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1)), losses)
    
    return

