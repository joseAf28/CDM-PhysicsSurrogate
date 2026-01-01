import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import math 
from torch.utils.data import Dataset, DataLoader

import src.models as models
import src.dataset as ds
import src.data_preparation as data_prep



def train_model(seed, device, config):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    
    X_scaled, y_scaled = data_dict['train']
    X_val_scaled, y_val_scaled = data_dict['val']
    X_test_scaled, y_test_scaled = data_dict['test']
    
    batch_size = config['training']['batch_size']

    dataset = ds.LTPDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ds.LTPDataset(X_val_scaled, y_val_scaled)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ds.LTPDataset(X_test_scaled, y_test_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ###* define the PCDAE architecture
    x_dim = config['model']['x_dim']
    y_dim = config['model']['y_dim']
    hidden_dim = config['model']['hidden_dim']
    
    s_noise = float(config['model']['s_noise'])
    sigma_max = float(config['model']['sigma_max'])
    
    pcdae = models.PCDAE(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim).to(device)
    
    ###* train the model
    
    lr = float(config['training']['lr'])
    num_epochs_max = config['training']['n_epochs_max']
    num_epochs_min = config['training']['n_epochs_min']
    noise_kernel = config['training']['noise_kernel']
    noise_dist = config['training']['noise_dist']
    
    
    optimizer = optim.Adam(pcdae.parameters(), lr=lr)
    stopper_pcdae = models.EarlyStopping(patience=20, min_delta=1e-4, min_epochs=num_epochs_min)

    for epoch in range(1, num_epochs_max+1):
        pcdae.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            
            u = torch.rand(x.size(0), 1)

            if noise_dist == 'sine':
                noise_level = sigma_max * (torch.sin((u + s_noise)/(1.0 + s_noise) * torch.pi / 2.0)).to(device)
            elif noise_dist == 'linear':
                noise_level = (sigma_max * u + s_noise).to(device)
            else:
                raise ValueError("noise dist is not defined")
                
            
            if noise_kernel == "VE":
                y_noisy = y + torch.randn_like(y) * noise_level
            elif noise_kernel == "VP":
                y_noisy = (1.0 - noise_level**2).sqrt() * y + torch.randn_like(y) * noise_level
            
            recon, _ = pcdae(x, y_noisy, noise_level)
            truth = torch.cat([x,y], dim=-1)
            
            loss = nn.MSELoss()(recon, truth)
            
            loss.backward()
            optimizer.step()
            
        
        pcdae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_dataloader:
                
                xv = xv.to(device)
                yv = yv.to(device)
                
                u = torch.rand(xv.size(0), 1)
                
                if noise_dist == 'sine':
                    noise_level = sigma_max * (torch.sin((u + s_noise)/(1.0 + s_noise) * torch.pi / 2.0)).to(device)
                elif noise_dist == 'linear':
                    noise_level = (sigma_max * u + s_noise).to(device)
                else:
                    raise ValueError("noise dist is not defined")
                
                if noise_kernel == "VE":
                    y_noisy = yv + torch.randn_like(yv) * noise_level
                elif noise_kernel == "VP":
                    y_noisy = (1.0 - noise_level**2).sqrt() * yv + torch.randn_like(yv) * noise_level
                
                recon, _ = pcdae(xv, y_noisy, noise_level)
                truth = torch.cat([xv,yv], dim=-1)
                
                val_loss += nn.MSELoss()(recon, truth).item()
                
            val_loss /= len(val_dataloader)
            
        
        if epoch % 400 == 0:
            print("Epoch: ", epoch, " Val_loss: ", val_loss)
        
        if stopper_pcdae.step(val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("seed: ", seed, " done")
    return pcdae



def train_EBM_model(seed, device, config):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    
    X_scaled, y_scaled = data_dict['train']
    X_val_scaled, y_val_scaled = data_dict['val']
    X_test_scaled, y_test_scaled = data_dict['test']
    
    batch_size = config['training']['batch_size']

    dataset = ds.LTPDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ds.LTPDataset(X_val_scaled, y_val_scaled)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ds.LTPDataset(X_test_scaled, y_test_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    ###* define the PCDAE architecture
    x_dim = config['model']['x_dim']
    y_dim = config['model']['y_dim']
    hidden_dim = config['model']['hidden_dim']
    
    s_noise = float(config['model']['s_noise'])
    sigma_max = float(config['model']['sigma_max'])
    
    pcdae_ebm = models.PCDAE_EBM(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim).to(device)
    
    ###* train the model
    
    lr = float(config['training']['lr'])
    num_epochs_max = config['training']['n_epochs_max']
    num_epochs_min = config['training']['n_epochs_min']
    noise_kernel = config['training']['noise_kernel']
    noise_dist = config['training']['noise_dist']
    
    optimizer = optim.Adam(pcdae_ebm.parameters(), lr=lr)
    stopper_pcdae = models.EarlyStopping(patience=20, min_delta=1e-4, min_epochs=num_epochs_min)
    
    for epoch in range(1, num_epochs_max+1):
        pcdae_ebm.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            
            u = torch.rand(x.size(0), 1)
            if noise_dist == 'sine':
                noise_level = (sigma_max * torch.sin((u + s_noise)/(1.0 + s_noise) * torch.pi / 2.0)).to(device)
            elif noise_dist == 'linear':
                noise_level = (sigma_max * u + s_noise).to(device)
            else:
                raise ValueError("noise dist is not defined")
                
            if noise_kernel == "VE":
                y_noisy = y + torch.randn_like(y) * noise_level
            elif noise_kernel == "VP":
                y_noisy = (1.0 - noise_level**2).sqrt() * y + torch.randn_like(y) * noise_level
            
            recon, _ = pcdae_ebm(x, y_noisy)
            truth = torch.cat([x,y], dim=-1)
            
            loss = nn.MSELoss()(recon, truth)
            
            loss.backward()
            optimizer.step()
            
        
        pcdae_ebm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_dataloader:
                
                xv = xv.to(device)
                yv = yv.to(device)
                
                u = torch.rand(xv.size(0), 1)
                    
                if noise_dist == 'sine':
                    noise_level = (sigma_max * torch.sin((u + s_noise)/(1.0 + s_noise) * torch.pi / 2.0)).to(device)
                elif noise_dist == 'linear':
                    noise_level = (sigma_max * u + s_noise).to(device)
                else:
                    raise ValueError("noise dist is not defined")
                
                if noise_kernel == "VE":
                    y_noisy = yv + torch.randn_like(yv) * noise_level
                elif noise_kernel == "VP":
                    y_noisy = (1.0 - noise_level**2).sqrt() * yv + torch.randn_like(yv) * noise_level
                
                recon, _ = pcdae_ebm(xv, y_noisy)
                truth = torch.cat([xv,yv], dim=-1)
                
                val_loss += nn.MSELoss()(recon, truth).item()
                
            val_loss /= len(val_dataloader)
            
        
        if epoch % 400 == 0:
            print("Epoch: ", epoch, " Val_loss: ", val_loss)
        
        if stopper_pcdae.step(val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("seed: ", seed, " done")
    return pcdae_ebm