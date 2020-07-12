import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from RealNVP.script.RealNVP_2D import *

import time
import datetime
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

import pickle

import random

## Masks used to define the number and the type of affine coupling layers
## In each mask, 1 means that the variable at the correspoding position is
## kept fixed in the affine couling layer
masks = [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],         
         [0.0, 1.0],
         [1.0, 0.0],
         [0.0, 1.0]]

## dimenstion of hidden units used in scale and translation transformation
hidden_dim = 128

## construct the RealNVP_2D object
realNVP = RealNVP_2D(masks, hidden_dim)
if torch.cuda.device_count():
    realNVP = realNVP.cuda()
device = next(realNVP.parameters()).device

optimizer = optim.Adam(realNVP.parameters(), lr = 0.0001)
num_steps = 102

data_path = "code_dict_latent_10_tot_100000_2020_07_12_16_40_32_pca_"
axis = (0,1)
data_path = "./data/" + data_path + str(axis)
code = pickle.load(open(data_path, 'rb'))

## the following loop learns the RealNVP_2D model by data
## in each loop, data is dynamically sampled from the scipy moon dataset
for i in code.keys():
    print("Transforming digit %i..." % i)
    X = code[i]
    for idx_step in range(num_steps):
        ## sample data from the scipy moon dataset
        random.shuffle(X)
        X = torch.Tensor(X).to(device = device)/10.0

        ## transform data X to latent space Z
        z, logdet = realNVP.inverse(X)
    
        ## calculate the negative loglikelihood of X
        loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    
        if (idx_step + 1) % 100 == 0:
            print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")
            
    ## after learning, we can test if the model can transform
    ## the moon data distribution into the normal distribution
    print("Training finished. Testing...")
    X = torch.Tensor(X[:500]).to(device = device)
    z, logdet_jacobian = realNVP.inverse(X)
    z = z.cpu().detach().numpy()
    
    print("Generating images...")
    X = X.cpu().detach().numpy()
    fig = plt.figure(2, figsize = (12.8, 4.8))
    fig.clf()
    plt.subplot(1,2,1)
    plt.plot(X[:, 0], X[:, 1], ".")
    plt.title("X sampled from Moon dataset")
    plt.xlabel(r"$x_%i$" % axis[0])
    plt.ylabel(r"$x_%i$" % axis[1])
    
    plt.subplot(1,2,2)
    plt.plot(z[:, 0], z[:, 1], ".")
    plt.title("Z transformed from X")
    plt.xlabel(r"$z_%i$" % axis[0])
    plt.ylabel(r"$z_%i$" % axis[1])
    plt.savefig("./img/" + data_path + "_%i_transformed_to_z.png" % i)
    
    ## after learning, we can also test if the model can transform
    ## the normal distribution into the moon data distribution 
    z = torch.normal(0, 1, size = (1000, 2)).to(device = device)
    X, _ = realNVP(z)
    X = X.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    
    fig = plt.figure(2, figsize = (12.8, 4.8))
    fig.clf()
    plt.subplot(1,2,1)
    plt.plot(z[:,0], z[:,1], ".")
    plt.title("Z sampled from normal distribution")
    plt.xlabel(r"$z_%i$" % axis[0])
    plt.ylabel(r"$z_%i$" % axis[1])
    
    plt.subplot(1,2,2)
    plt.plot(X[:,0], X[:,1], ".")
    plt.title("X transformed from Z")
    plt.xlabel(r"$x_%i$" % axis[0])
    plt.ylabel(r"$x_%i$" % axis[1])
    plt.savefig("./img/" + data_path + "_%i_transformed_from_z.png" % i)

