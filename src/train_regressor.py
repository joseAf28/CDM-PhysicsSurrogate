import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import math 
from torch.utils.data import Dataset, DataLoader

import src.models as models
import src.dataset as ds
import src.data_preparation as data_prep
import src.inference_pcdae as constraints


def train_model(seed, device, config):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    
    X_scaled, y_scaled = data_dict['train']
    X_val_scaled, y_val_scaled = data_dict['val']
    X_test_scaled, y_test_scaled = data_dict['test']
    scaler_X, scaler_Y = data_dict['scalers']
    
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
    hidden_sizes = config['model']['hidden_size']
    
    regressor = models.Regressor(input_size=x_dim, output_size=y_dim, hidden_sizes=hidden_sizes).to(device)
    
    ###* train the model
    
    lr = float(config['training']['lr'])
    num_epochs_max = config['training']['n_epochs_max']
    num_epochs_min = config['training']['n_epochs_min']
    weight_physics = config['training']['weights_physics']
    
    optimizer = optim.Adam(regressor.parameters(), lr=lr)
    stopper_pcdae = models.EarlyStopping(patience=20, min_delta=1e-4, min_epochs=num_epochs_min)

    for epoch in range(1, num_epochs_max+1):
        regressor.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            
            pred = regressor(x)
            loss = nn.MSELoss()(pred, y)
            loss_physics = (constraints.constraints_func_loss(x,pred, scaler_X, scaler_Y)**2).mean()
            
            total_loss = (1 - weight_physics) * loss + weight_physics * loss
            total_loss.backward()
            optimizer.step()
            
        
        regressor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xv, yv in val_dataloader:
                
                xv = xv.to(device)
                yv = yv.to(device)
                
                pred = regressor(xv)
                loss = nn.MSELoss()(pred, yv)
                loss_physics = (constraints.constraints_func_loss(xv,pred, scaler_X, scaler_Y)**2).mean()
                
                total_loss = (1 - weight_physics) * loss + weight_physics * loss
                
                val_loss += total_loss.item()
                
            val_loss /= len(val_dataloader)
            
        
        if epoch % 100 == 0:
            print("Epoch: ", epoch, " Val_loss: ", val_loss)

        if stopper_pcdae.step(val_loss, epoch):
            break
    
    print("seed: ", seed, " done")
    return regressor

