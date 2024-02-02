'''
Source code for training SynthDistill (IJCB 2023):
SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data
'''
import os,sys
sys.path.append(os.getcwd())

import torch
import random
import numpy as np
from tqdm import tqdm
import math
import argparse

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)


parser = argparse.ArgumentParser(description='SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data')
parser.add_argument('--model', metavar='<model>', type= str, default='TinyFaR_A',
                    help='TinyFaR_A,TinyFaR_B,TinyFaR_C')
parser.add_argument('--resampling_coef', metavar='<resampling_coef>', type= float, default=1.0,
                    help='resampling coefficient')                    

args = parser.parse_args()
resampling_coef = args.resampling_coef


batch_size=2*64
num_epochs=21
iterations_per_epoch_train = int (1e6/batch_size)
iterations_per_epoch_val   = int (1e4/batch_size)


# Save the results in the following folder.
results_path = 'results'
os.makedirs(results_path, exist_ok= True)
with open(results_path + '/log_train.txt','w') as f:
    pass
with open(results_path + '/log.csv', 'w') as f:
    f.write(f"epoch, loss_test_MSE, loss_test_cos, loss_test\n")
#========================================================

#=================== import Network =====================
from src.Network import LightNetwork
if args.model=='TinyFaR_A':
    light_model = LightNetwork(model_name='tinynet_a')
elif args.model=='TinyFaR_B':
    light_model = LightNetwork(model_name='tinynet_b')
elif args.model=='TinyFaR_C':
    light_model = LightNetwork(model_name='tinynet_c')
print('light_model # params', sum(p.numel() for p in light_model.parameters()))
light_model.to(device)

from src.ArcFace import get_FaceRecognition_transformer
large_model= get_FaceRecognition_transformer(device)

#=================== StyleGAN
sys.path.append('./stylegan3') # git clone https://github.com/NVlabs/stylegan3

import pickle
import torch_utils

path_stylegan = './stylegan2-ffhq-256x256.pkl'
with open(path_stylegan, 'rb') as f:
    StyleGAN = pickle.load(f)['G_ema']
    # StyleGAN.to(device)
    # StyleGAN.eval()
    StyleGAN_synthesis = StyleGAN.synthesis
    StyleGAN_mapping   = StyleGAN.mapping
    StyleGAN_synthesis.eval()
    StyleGAN_mapping.eval()
    StyleGAN_synthesis.to(device)
    StyleGAN_mapping.to(device)
z_dim_StyleGAN = StyleGAN.z_dim
from src.Crop import Crop_and_resize
#========================================================

#=================== Optimizers =========================
MSELoss = torch.nn.MSELoss(reduction ='mean')
#========================================================



#=================== Optimizers =========================
lr = 0.001                
optimizer = torch.optim.Adam(light_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#========================================================



######################################################################

print('learning rate = ', str(lr))

for epoch in tqdm(range(num_epochs)):
    light_model.train()
    # large_model.eval()
    for itr in range(iterations_per_epoch_train):

        with torch.no_grad():
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, z_dim_StyleGAN, device=device)
            # generate w from noise(z)
            w = StyleGAN_mapping(z=noise, c=None, truncation_psi=1.0).detach()
            # # syntheise images from w
            img = StyleGAN_synthesis(w).detach()
            # img = StyleGAN(z=noise, c=None, truncation_psi=1.0).detach()
            # clamp values for generated images prior to feature extractor
            img = torch.clamp(img, min=-1, max=1)
            img = (img + 1) / 2.0 # range: (0,1)
            img = Crop_and_resize(img)

        # ================== forward  =================
        emb_light = light_model((img- 0.5) / 0.5)
        with torch.no_grad():
            emb_large = large_model.transform(img*255.)

        MSE = MSELoss(emb_light, emb_large)
        cos = torch.nn.CosineSimilarity()(emb_light, emb_large).mean()

        loss = MSE  
        # ================== backward =================
        optimizer.zero_grad()
        loss.backward()#(retain_graph=True)
        optimizer.step()
        #==============================================

        cos_sim =  torch.nn.CosineSimilarity()(emb_light, emb_large)
        coef = (cos_sim + 1)/2.0
        w_=w[:,0,:]
        w_ = w_ + coef.unsqueeze(1) * resampling_coef * torch.randn(batch_size, StyleGAN.w_dim, device=device)
        w = w_.unsqueeze(1).repeat([1, StyleGAN.num_ws, 1])


        with torch.no_grad():
            # Generate batch of latent vectors
            # noise = noise[indx_wrost_sim] + 0.1 * torch.randn(batch_size, z_dim_StyleGAN, device=device)
            # generate w from noise(z)
            # w = StyleGAN_mapping(z=noise, c=None, truncation_psi=1.0).detach()
            # # syntheise images from w
            img = StyleGAN_synthesis(w).detach()
            # img = StyleGAN(z=noise, c=None, truncation_psi=1.0).detach()
            # clamp values for generated images prior to feature extractor
            img = torch.clamp(img, min=-1, max=1)
            img = (img + 1) / 2.0 # range: (0,1)
            img = Crop_and_resize(img)

        # ================== forward  =================
        emb_light = light_model((img- 0.5) / 0.5)
        with torch.no_grad():
            emb_large = large_model.transform(img*255.)

        MSE = MSELoss(emb_light, emb_large)
        cos = torch.nn.CosineSimilarity()(emb_light, emb_large).mean()

        loss = MSE  
        # ================== backward =================
        optimizer.zero_grad()
        loss.backward()#(retain_graph=True)
        optimizer.step()
        #==============================================
        
        if itr%500==0:
            print('epoch = '+ str(epoch) + ', iteration = ' + str(itr) + ', loss = ' + str(loss.item()), flush=True)
            with open(results_path + '/log_train.txt','a') as f:
                f.write('epoch = '+ str(epoch) + ', iteration = ' + str(itr) + ', loss = ' + str(loss.item()) + '\n')

    light_model.eval()
    # large_model.eval()

    torch.save(light_model.state_dict(),  results_path + '/light_model_' +str(epoch)+'.pt')   
    
    loss_test = loss_test_cos = loss_test_MSE = 0
    for itr in range(iterations_per_epoch_val):
        with torch.no_grad():
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, z_dim_StyleGAN, device=device)
            # generate w from noise(z)
            w = StyleGAN_mapping(z=noise, c=None, truncation_psi=1.0).detach()
            # syntheise images from w
            img = StyleGAN_synthesis(w).detach()
            # img = StyleGAN(z=noise, c=None, truncation_psi=1.0).detach()
            # clamp values for generated images prior to feature extractor0
            img = torch.clamp(img, min=-1, max=1)
            img = (img + 1) / 2.0 # range: (0,1)
            img = Crop_and_resize(img)

            # ================== forward  =================
            emb_light = light_model((img- 0.5) / 0.5)
            emb_large = large_model.transform(img*255.)
            MSE = MSELoss(emb_light, emb_large)
            cos = torch.nn.CosineSimilarity()(emb_light, emb_large).mean()

            loss = MSE  

            loss_test_MSE  += MSE.item()
            loss_test_cos += cos.item()
            loss_test     += loss.item()

    with open(results_path + '/log.csv', 'a') as f:
        f.write(f"{epoch}, {loss_test_MSE/iterations_per_epoch_val}, {loss_test_cos/iterations_per_epoch_val}, {loss_test/iterations_per_epoch_val}\n")

    # Update schedulers
    scheduler.step()