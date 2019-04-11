#importing neccesary things
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from os import listdir

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=5, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SimpleEnhancer(nn.Module):
    def __init__(self):
        super(SimpleEnhancer, self).__init__()
        self.tconv2d = nn.ConvTranspose2d(in_channels=10,
                           out_channels=3,
                           kernel_size=5,
                           stride=2,
                           output_padding=1, # needed because stride=2
                           padding=2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.tconv2d(x)
        x = self.activation(x)
        return x
    
def load_sample(path, num_image):
    #load sample data for debugging
    sample_numpy = []
    sample_tensor = []
    for i in range(num_image):
            image = plt.imread("{}/{:03d}.jpg".format(path, i+1))
            sample_numpy.append(image)
            sample_tensor.append(torch.from_numpy((image/255)).type(torch.FloatTensor).permute(2,0,1))
    return sample_numpy, sample_tensor


def train_autoencoder(model, input_images, num_epochs=5, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    outputs = []
    for epoch in range(num_epochs):
        for k in range(len(input_images)):
            out = []
            img = input_images[k].reshape(1, input_images[k].shape[0], input_images[k].shape[1], input_images[k].shape[2])
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            out.append(recon[0])
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append(out)
    return outputs

def train_enhancer(model_autoencoder, model_enhancer, input_images, output_images, num_epochs=5, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_enhancer.parameters(), lr=learning_rate, weight_decay=1e-5)
    outputs = []
    for epoch in range(num_epochs):
        for k in range(len(input_images)):
            out = []
            img = model_autoencoder.encoder(input_images[k].reshape(1, input_images[k].shape[0], input_images[k].shape[1], input_images[k].shape[2]))
            recon = model_enhancer(img.detach())
            loss = criterion(recon, output_images[k].reshape(1, output_images[k].shape[0], output_images[k].shape[1], output_images[k].shape[2]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            out.append(recon.reshape(output_images[k].shape[0], output_images[k].shape[1], output_images[k].shape[2]))
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append(out)
    return outputs
