import numpy as np
import yaml
import torch 
import torch.nn as nn
import math 
import h5py
import pickle
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import src.models as models
import src.dataset as ds
import src.data_preparation as data_prep
import src.inference_pcdae as inference
import src.train_pcdae as train



def procedure_one(config, device, condition):
    
    seed, ratio = condition
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config['training']['ratio_test_val_train'] = ratio
    
    pcdae = train.train_model(seed, device, config)
    
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    X_test_scaled, y_test_scaled = data_dict['test']
    scaler_X, scaler_Y = data_dict['scalers']
    
    batch_size = 128

    test_dataset = ds.LTPDataset(X_test_scaled, y_test_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    s_noise = float(config['model']['s_noise'])
    sigma_max = float(config['model']['sigma_max'])
    noise_dist = config['training']['noise_dist']
    
    T1_T, T1_K = int(config['inference']['T1_T']), int(config['inference']['T1_K'])
    T2_T, T2_K = int(config['inference']['T2_T']), int(config['inference']['T2_K'])

    T1_vec = torch.linspace(1, 0, T1_T)
    T2_vec = torch.linspace(1, 0, T2_T)

    T1_schedule = sigma_max * torch.sin((T1_vec + s_noise)/(1.0 + s_noise) * torch.pi / 2.0).to(device)
    T2_schedule = sigma_max * torch.sin((T1_vec + s_noise)/(1.0 + s_noise) * torch.pi / 2.0).to(device)
    
    ###! EBM sampling

    test_init_loss = 0
    test_T1_loss = 0
    test_T2_loss = 0
    
    constant_init_loss = 0
    constant_T_loss = 0
    constant_Tadapt_loss = 0
    
    
    for i, (x, y) in enumerate(test_dataloader):
        
        y_init = torch.randn_like(y)
        y_T1 = inference.inference_scheduled_ODE(pcdae, x, y_init, 
                            noise_schedule=T1_schedule, steps_per_level=T1_K, eps_clip=None)
        
        y_T2 = inference.inference_scheduled_ODE(pcdae, x, y_init, 
                            noise_schedule=T2_schedule, steps_per_level=T2_K, eps_clip=None)
        
        constant_init_loss += inference.get_constraints_scalar(x, y_init, scaler_X, scaler_Y).pow(2).mean()
        constant_T_loss += inference.get_constraints_scalar(x, y_T1, scaler_X, scaler_Y).pow(2).mean()
        constant_Tadapt_loss += inference.get_constraints_scalar(x, y_T2, scaler_X, scaler_Y).pow(2).mean()
        
        loss_init = nn.MSELoss()(y_init, y)
        loss_T1 = nn.MSELoss()(y_T1, y)
        loss_T2 = nn.MSELoss()(y_T2, y)
        
        test_init_loss += loss_init.item()
        test_T1_loss += loss_T1.item()
        test_T2_loss += loss_T2.item()
        
    test_init_loss /= len(test_dataloader)
    test_T1_loss /= len(test_dataloader)
    test_T2_loss /= len(test_dataloader)
    
    constant_init_loss /= len(test_dataloader)
    constant_T_loss /= len(test_dataloader)
    constant_Tadapt_loss /= len(test_dataloader)
    
    return (seed, ratio, test_init_loss, test_T1_loss, test_T2_loss, constant_init_loss, constant_T_loss, constant_Tadapt_loss)





if __name__ == "__main__":
    
    device = torch.device('cpu')
    config_file = "config_pcdae.yaml"
    output_file = "results/scaling_training_pcdae_sine_V3.h5"
    
    seed_vec = np.arange(1, 400, 40, dtype=int)
    test_val_ratios = [0.75, 0.6, 0.45, 0.3, 0.15]
    
    
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The file 'config.yaml' was not found.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit()
    
    
    condition_vec = []
    for hidden in test_val_ratios:
        for seed in seed_vec:
            condition_vec.append([seed, hidden])
    
        
    results = Parallel(n_jobs=-1, backend="loky")(delayed(procedure_one)(config, device, condition) for condition in condition_vec)
    seed_vec, hidden_vec, loss_init_vec, loss_refine1_vec, loss_refine2_vec, constant_init_loss, constant_T_loss, constant_Tadapt_loss = zip(*results)
    
    pickled_config = pickle.dumps(config)
    
    fileh5 = h5py.File(output_file, 'w')
    group = fileh5.require_group("results")
    group.create_dataset("seed", data=seed_vec)
    group.create_dataset("ratio", data=hidden_vec)
    group.create_dataset("loss_init", data=loss_init_vec)
    group.create_dataset("loss_T1", data=loss_refine1_vec)
    group.create_dataset("loss_T2", data=loss_refine2_vec)
    group.create_dataset("constant_init", data=constant_init_loss)
    group.create_dataset("constant_T", data=constant_T_loss)
    group.create_dataset("constant_Tadapt", data=constant_Tadapt_loss)
    group.create_dataset("config", data=np.void(pickled_config))
    fileh5.close()