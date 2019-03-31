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
import timea
import os
from os import listdir
from os.path import isfile, join
import torchvision
import torchvision.transforms as transforms

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
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.discriminator(x)
        x = F.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))
        x = x.view(-1)
        x = self.activation(x)
        return x
def train_enhancer(model, discriminator, path, mse_ratio=0.5, train_model_per_batch=2, batch_size=1, learning_rate=1e-3, num_epochs=5, device="cpu", cont=True, prev_model_path = "", prev_discriminator_path = ""):
    
    #move data and model to specified device, and activate parallel computation where possible
    if torch.cuda.device_count() > 1:
        device_ids = [i for i in range(torch.cuda.device_count())]
        
        name = model.name
        model = nn.DataParallel(model, device_ids)
        model.name = name
        
        name = discriminator.name
        discriminator = nn.DataParallel(discriminator, device_ids)
        discriminator.name = name
    model.to(device)
    discriminator.to(device)
    if(prev_model_path != ""):
        model.load_state_dict(torch.load(prev_model_path))
    if(prev_discriminator_path != ""):
        discriminator.load_state_dict(torch.load(prev_discriminator_path))
    
    if not os.path.exists("{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs)):
        os.makedirs("{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs))   
    
    #create dataloader for images
    transform=transforms.Compose([transforms.ToTensor()])
    dataset=torchvision.datasets.ImageFolder(path, transform=transform)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    real_image = next(iter(dataloader))[0][0]
    
    #seeds and optimizer
    torch.manual_seed(42)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    m_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    
    #list that contains losses and outputs
    losses = []
    
    start_time = time.time()#time stamp
    

    found_latest_state_dict = False
    for epoch in range(num_epochs):
        
        #continue from the newest save
        if(cont and not found_latest_state_dict):
            if(os.path.exists("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1))):
                continue
            elif(epoch == 0):
                found_latest_state_dict = True
            else: 
                found_latest_state_dict = True
                model.load_state_dict(torch.load("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch)))
                discriminator.load_state_dict(torch.load("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, discriminator.name, batch_size, learning_rate, epoch)))
                print("Found latest save of model of Epoch {:0>3d}.".format(epoch))
        
        #switch the models back to train mode
        model.train()
        discriminator.train()
        
        #intiailize total losses
        total_loss = 0
        total_m_loss = 0
        total_d_loss = 0
        
        #initialize progress prompt
        sys.stdout.write('Epoch: {:0>3d} 000%'.format(epoch+1))
        
        for i, (img) in enumerate(dataloader, 0):
            #generate low and high resolution images
            img = img[0]
            low_img = F.interpolate(img, scale_factor=0.5).contiguous()
            img = img.to(device)
            low_img = low_img.to(device)
            
            #train discriminator
            discriminator.zero_grad()#zero grad
            #forward
            fake_img = model(low_img)
            d_out = discriminator(torch.cat([img, fake_img]))
            d_loss = bce_criterion(d_out, torch.cat([torch.zeros(img.size(0)),
                                                 torch.ones(img.size(0))]).to(device))
            #backward
            d_loss.backward()
            d_optimizer.step()
            #add total loss of discrimator
            total_d_loss += d_loss.item()
            
            #train model
            for j in range(train_model_per_batch):
                model.zero_grad()
                #forward
                fake_img = model(low_img)
                recon = discriminator(fake_img)
                #calculate loss
                loss = mse_criterion(fake_img, img)
                bce_loss = bce_criterion(recon, torch.zeros(img.size(0)).to(device))
                #backward
                m_loss = mse_ratio * loss + (1-mse_ratio) * bce_loss
                m_loss.backward() 
                m_optimizer.step()
                #add total loss of model
                total_loss += loss.item()
                total_m_loss += bce_loss.item()
            
            
            # updates print to notify user progress of this epoch
            sys.stdout.write('\r\r\r\r\r\r\r\r\r\r\r\r\r\r\rEpoch: {:0>3d} {:0>3d}%'.format(epoch+1, int((i+1)/(len(dataloader.dataset)/batch_size)*100)))
            sys.stdout.flush()
        
            #empty cache to save up ram
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        total_loss /= (i+1)*train_model_per_batch
        total_m_loss /= (i+1)*train_model_per_batch
        total_d_loss /= (i+1)
        
        #save outputs and stats for this epoch
        losses.append(total_loss)
        torch.save(model.state_dict(), "{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1))
        torch.save(discriminator.state_dict(), "{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, discriminator.name, batch_size, learning_rate, epoch+1))
        
        #evaluate this epoch
        print(' MSE Loss: {:.4f}, Enhancer Loss: {:.4f}, Discriminator Loss: {:.4f}, Time: {}min {}sec'.format(float(total_loss), float(total_m_loss),  float(total_d_loss), int((time.time()-start_time)/60), int((time.time()-start_time)%60)))
        model.eval()
        discriminator.eval()
        test_image = model(F.interpolate(torch.stack([real_image]), scale_factor=0.5).to(device))
        test_image = test_image[0].cpu().permute(1,2,0).detach().numpy()
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.subplot(1, 2, 2)
        plt.imshow(real_image.cpu().permute(1,2,0).detach().numpy())
        plt.show()
        plt.imsave("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}_test_image.jpg".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1), test_image)
        
        #empty cache to save up ram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
    #save stats into file
    np.savetxt("{}_train_losses.csv".format("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1)), losses)
    
    return

def image_gen (model, path,prev_model_path, dataset, batch_size=1):
    
    with torch.no_grad():    

        print("Creating fake images tensor for later uses.")
        sys.stdout.write('000%')
        sys.stdout.flush()
        model.eval()
        batch_size=64
        dataloader_fake = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        if torch.cuda.device_count() > 1:
            device_ids = [i for i in range(torch.cuda.device_count())]
            torch.cuda.empty_cache()
            name = model.name
            model = nn.DataParallel(model, device_ids)
            model.name = name
        
        model.load_state_dict(torch.load(prev_model_path,) )#map_location=torch.device('cpu')))       
        model.to(device)
        fake_images = torch.zeros(len(dataset)+(batch_size-len(dataset)%batch_size), 3, 512, 512)

        fake_images_count = 0
        for i, (images,targets) in enumerate(dataloader_fake, 0):
  
            data = images
            data = data.to(device)
            puffer = (model(F.interpolate(data, scale_factor=0.5)));

            fake_images[fake_images_count:fake_images_count + puffer.shape[0]] = puffer[0:puffer.shape[0]]

            fake_images_count += puffer.shape[0]
            sys.stdout.write('\r\r\r\r{:0>3d}%'.format(int((fake_images_count)/(len(dataset))*100)))
            sys.stdout.flush()

            puffer= None
            del puffer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if fake_images_count >= len(dataset):
                print('true')
                break
            
    print("Done")
    torch.set_grad_enabled(True)
    del model
    fake_images = fake_images.cpu()
    torch.cuda.empty_cache()
    return (fake_images)
	
def train_enhancer_only(model, path, batch_size=1, learning_rate=1e-3, num_epochs=5, device="cpu", cont=True, prev_model_path = ""):
    
    #move data and model to specified device, and activate parallel computation where possible
    if torch.cuda.device_count() > 1:
        device_ids = [i for i in range(torch.cuda.device_count())]
        name = model.name
        model = nn.DataParallel(model, device_ids)
        model.name = name
    model.to(device)
    if(prev_model_path != ""):
        model.load_state_dict(torch.load(prev_model_path))
    
    if not os.path.exists("{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs)):
        os.makedirs("{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs))   
    
    #create dataloader for images
    transform=transforms.Compose([transforms.ToTensor()])
    dataset=torchvision.datasets.ImageFolder(path, transform=transform)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    #seeds and optimizer
    torch.manual_seed(42)
    mse_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    
    #list that contains losses and outputs
    losses = []
    
    start_time = time.time()#time stamp
    

    found_latest_state_dict = False
    for epoch in range(num_epochs):
        
        #continue from the newest save
        if(cont and not found_latest_state_dict):
            if(os.path.exists("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1))):
                continue
            elif(epoch == 0):
                found_latest_state_dict = True
            else: 
                found_latest_state_dict = True
                model.load_state_dict(torch.load("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch)))            
                print("Found latest save of model of Epoch {:0>3d}.".format(epoch))
        
        #switch the models back to train mode
        model.train()
        
        #intiailize total losses
        total_loss = 0
        
        #initialize progress prompt
        sys.stdout.write('Epoch: {:0>3d} 000%'.format(epoch+1))
        
        for i, (img) in enumerate(dataloader, 0):
            #generate low and high resolution images
            img = img[0]
            low_img = F.interpolate(img, scale_factor=0.5).contiguous()
            img = img.to(device)
            low_img = low_img.to(device)
            
            #train model
            optimizer.zero_grad()
            
            #forward
            fake_img = model(low_img)
            
            #backward
            loss = mse_criterion(fake_img, img)
            loss.backward() 
            optimizer.step()
            
            #add total loss of model
            total_loss += loss.item()
            
            
            # updates print to notify user progress of this epoch
            sys.stdout.write('\r\r\r\r\r\r\r\r\r\r\r\r\r\r\rEpoch: {:0>3d} {:0>3d}%'.format(epoch+1, int((i+1)/(len(dataloader.dataset)/batch_size)*100)))
            sys.stdout.flush()
        
            #empty cache to save up ram
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        total_loss /= (i+1)
        
        #save outputs and stats for this epoch
        losses.append(total_loss)
        torch.save(model.state_dict(), "{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1))      
        
        #evaluate this epoch
        print(' MSE Loss: {:.4f}, Time: {}min {}sec'.format(float(total_loss), int((time.time()-start_time)/60), int((time.time()-start_time)%60)))
        model.eval()
        real_image = next(iter(dataloader))[0][0]
        test_image = model(F.interpolate(torch.stack([real_image]), scale_factor=0.5).to(device))
        test_image = test_image[0].cpu().permute(1,2,0).detach().numpy()
        plt.figure(figsize=(7, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image)
        plt.subplot(1, 2, 2)
        plt.imshow(real_image.cpu().permute(1,2,0).detach().numpy())
        plt.show()
        plt.imsave("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}_test_image.jpg".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1), test_image)
        
        #empty cache to save up ram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
    #save stats into file
    np.savetxt("{}_train_losses.csv".format("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1)), losses)
    
    return

def train_discriminator_only(model, discriminator, path, batch_size=1, learning_rate=1e-3, num_epochs=5, device="cpu", cont=True, prev_model_path = "", prev_discriminator_path = ""):
    
    #move data and model to specified device, and activate parallel computation where possible
    if torch.cuda.device_count() > 1:
        device_ids = [i for i in range(torch.cuda.device_count())]
        
        name = model.name
        model = nn.DataParallel(model, device_ids)
        model.name = name
        
        name = discriminator.name
        discriminator = nn.DataParallel(discriminator, device_ids)
        discriminator.name = name
    model.to(device)
    discriminator.to(device)
    if(prev_model_path != ""):
        model.load_state_dict(torch.load(prev_model_path))
    if(prev_discriminator_path != ""):
        discriminator.load_state_dict(torch.load(prev_discriminator_path))
    
    if not os.path.exists("{}_bs{}_lr{}_epoch{}".format(discriminator.name, batch_size, learning_rate, num_epochs)):
        os.makedirs("{}_bs{}_lr{}_epoch{}".format(discriminator.name, batch_size, learning_rate, num_epochs))   
     
    #create fake images for later uses
    print("Creating fake images for later uses.")
    sys.stdout.write('000%')
    sys.stdout.flush()
    torch.no_grad()
    model.eval()
    #dataset = data_utils.TensorDataset(input_images)
    
    transform=transforms.Compose([transforms.ToTensor()])
    dataset=torchvision.datasets.ImageFolder(path, transform=transform)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    fake_images = torch.zeros(len(dataset)+(batch_size-len(dataset)%batch_size), 3, 512, 512)
    
    fake_images_count = 0
    for i, buffer in enumerate(dataloader, 0):
        buffer = buffer[0]
        buffer = Variable(model(F.interpolate(buffer, scale_factor=0.5).contiguous().to(device)), volatile=True);
        buffer = buffer.cpu()
        fake_images[fake_images_count:fake_images_count + buffer.shape[0]] = buffer[0:buffer.shape[0]]
        fake_images_count += buffer.shape[0]
        sys.stdout.write('\r\r\r\r{:0>3d}%'.format(int((i+1)/(len(dataset)/batch_size)*100)))
        sys.stdout.flush()
        #empty cache to save up ram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
    print(" Done")
    torch.set_grad_enabled(True)
    
    #create dataloader for images
    #dataset = None
    dataset_fake = data_utils.TensorDataset(fake_images)
    dataloader_fake = data_utils.DataLoader(dataset_fake, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    #seeds and optimizer
    torch.manual_seed(42)
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    #list that contains losses and outputs
    losses = []
    
    start_time = time.time()#time stamp
    

    found_latest_state_dict = False
    for epoch in range(num_epochs):
        
        #continue from the newest save
        if(cont and not found_latest_state_dict):
            if(os.path.exists("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(discriminator.name, batch_size, learning_rate, num_epochs, discriminator.name, batch_size, learning_rate, epoch+1))):
                continue
            elif(epoch == 0):
                found_latest_state_dict = True
            else: 
                found_latest_state_dict = True
                discriminator.load_state_dict(torch.load("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(discriminator.name, batch_size, learning_rate, num_epochs, discriminator.name, batch_size, learning_rate, epoch)))
                print("Found latest save of model of Epoch {:0>3d}.".format(epoch))
        
        #switch the models back to train mode
        discriminator.train()
        
        #intiailize total losses
        total_d_loss = 0
        
        #initialize progress prompt
        sys.stdout.write('Epoch: {:0>3d} 000%'.format(epoch+1))
        iterator_fake = iter(dataloader_fake)
        for i, img in enumerate(dataloader, 0):
            discriminator.zero_grad()#zero grad
            
            #bring inputs to device
            img = img[0]
            fake_img = next(iterator_fake)
            fake_img = fake_img[0]
            
            #forward
            out = discriminator(torch.cat([img, fake_img]).to(device))
            loss = bce_criterion(out, torch.cat([torch.zeros(img.shape[0]), torch.ones(fake_img.shape[0])]).to(device))
            
            #backward
            loss.backward()
            optimizer.step()
            #add total loss of discrimator
            total_d_loss += loss.item()
            
            # updates print to notify user progress of this epoch
            sys.stdout.write('\r\r\r\r\r\r\r\r\r\r\r\r\r\r\rEpoch: {:0>3d} {:0>3d}%'.format(epoch+1, int((i+1)/(len(dataloader.dataset)/batch_size)*100)))
            sys.stdout.flush()
        
            #empty cache to save up ram
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        total_d_loss /= (i+1)
        
        #save outputs and stats for this epoch
        losses.append(total_d_loss)
        #torch.save(model.state_dict(), "{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(model.name, batch_size, learning_rate, num_epochs, model.name, batch_size, learning_rate, epoch+1))
        torch.save(discriminator.state_dict(), "{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(discriminator.name, batch_size, learning_rate, num_epochs, discriminator.name, batch_size, learning_rate, epoch+1))
        
        #evaluate this epoch
        print(' Discriminator Loss: {:.4f}, Time: {}min {}sec'.format(float(total_d_loss), int((time.time()-start_time)/60), int((time.time()-start_time)%60)))
        #empty cache to save up ram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
    #save stats into file
    np.savetxt("{}_train_losses.csv".format("{}_bs{}_lr{}_epoch{}/{}_bs{}_lr{}_epoch{}".format(discriminator.name, batch_size, learning_rate, num_epochs, discriminator.name, batch_size, learning_rate, epoch+1)), losses)
    model.train()
    return

def show_fake_images(model, state_dict_path, input_path, output_path, batch_size = 16):
    #move data and model to specified device, and activate parallel computation where possible
    if torch.cuda.device_count() > 1:
        device_ids = [i for i in range(torch.cuda.device_count())] 
        name = model.name
        model = nn.DataParallel(model, device_ids)
        model.name = name
    model.to(device)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)   
    sys.stdout.write('000%')
    sys.stdout.flush()
    transform=transforms.Compose([transforms.ToTensor()])
    dataset=torchvision.datasets.ImageFolder(input_path, transform=transform)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    for i, img in enumerate(dataloader, 0):
        img = img[0]
        img = Variable(model(F.interpolate(img, scale_factor=0.5).contiguous().to(device)), volatile=True);
        img = img.cpu()
        for j in range(batch_size):
            plt.imsave("{}/{:0>5d}.jpg".format(output_path, i*batch_size+j), img[j].detach().permute(1,2,0).numpy())
        sys.stdout.write("\r\r\r\r{:0>3d}%".format(int((i+1)/(len(dataset)/batch_size)*100)))
        sys.stdout.flush()  
    model.train()
    return