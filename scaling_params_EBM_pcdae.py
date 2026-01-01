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
    
    seed, hidden_dim = condition
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config['model']['hidden_dim'] = hidden_dim
    
    pcdae = train.train_EBM_model(seed, device, config)
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    X_test_scaled, y_test_scaled = data_dict['test']
    scaler_X, scaler_Y = data_dict['scalers']
    
    batch_size = 128
    test_dataset = ds.LTPDataset(X_test_scaled, y_test_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    EBM_T = config['inference']['EBM_T']
    
    test_init_loss = 0.0
    test_T_loss = 0
    test_Tadapt_loss = 0
    
    constant_init_loss = 0
    constant_T_loss = 0
    constant_Tadapt_loss = 0
    
    for i, (x, y) in enumerate(test_dataloader):
        
        y_init = torch.randn_like(y)
        y_T = inference.inference_blind_adaptive_ODE(pcdae, x, y_init, eta=0.1, max_steps=EBM_T, eps_conv=1e-5, eps_clip=None)
        y_Tadapt = inference.inference_blind_adaptive_ODE(pcdae, x, y_init, eta=1.0, max_steps=EBM_T, eps_conv=1e-5, eps_clip=None, adaptative=True)
        
        constant_init_loss += inference.get_constraints_scalar(x, y_init, scaler_X, scaler_Y).pow(2).mean()
        constant_T_loss += inference.get_constraints_scalar(x, y_T, scaler_X, scaler_Y).pow(2).mean()
        constant_Tadapt_loss += inference.get_constraints_scalar(x, y_Tadapt, scaler_X, scaler_Y).pow(2).mean()
        
        loss_init = nn.MSELoss()(y_init, y)
        loss_T = nn.MSELoss()(y_T, y)
        loss_Tadapt = nn.MSELoss()(y_Tadapt, y)
        
        test_init_loss += loss_init.item()
        test_T_loss += loss_T.item()
        test_Tadapt_loss += loss_Tadapt.item()
    
    
    test_init_loss /= len(test_dataloader)
    test_T_loss /= len(test_dataloader)
    test_Tadapt_loss /= len(test_dataloader)
    
    constant_init_loss /= len(test_dataloader)
    constant_T_loss /= len(test_dataloader)
    constant_Tadapt_loss /= len(test_dataloader)
    
    print("hidden_dim, seed, RMSE: ", hidden_dim, seed, np.sqrt(test_T_loss), np.sqrt(test_Tadapt_loss))
        
    return (seed, hidden_dim, test_init_loss, test_T_loss, test_Tadapt_loss, constant_init_loss, constant_T_loss, constant_Tadapt_loss)


if __name__ == "__main__":
    
    device = torch.device('cpu')
    config_file = "config_ebm_pcdae.yaml"
    output_file = "results/scaling_params_ebm_pcdae_sine_V3.h5"
    
    seed_vec = np.arange(1, 400, 40, dtype=int)
    hidden_sizes = [25, 35, 47, 58, 68, 85]
    
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
    for hidden in hidden_sizes:
        for seed in seed_vec:
            condition_vec.append([seed, hidden])
    
    results = Parallel(n_jobs=-1, backend="loky")(delayed(procedure_one)(config, device, condition) for condition in condition_vec)
    seed_vec, hidden_vec, loss_init_vec, loss_T_vec, loss_Tadapt_vec, constant_init_loss, constant_T_loss, constant_Tadapt_loss = zip(*results)
    
    pickled_config = pickle.dumps(config)
    
    fileh5 = h5py.File(output_file, 'w')
    group = fileh5.require_group("results")
    group.create_dataset("seed", data=seed_vec)
    group.create_dataset("hidden", data=hidden_vec)
    group.create_dataset("loss_init", data=loss_init_vec)
    group.create_dataset("loss_T", data=loss_T_vec)
    group.create_dataset("loss_Tadapt", data=loss_Tadapt_vec)
    group.create_dataset("constant_init", data=constant_init_loss)
    group.create_dataset("constant_T", data=constant_T_loss)
    group.create_dataset("constant_Tadapt", data=constant_Tadapt_loss)
    group.create_dataset("config", data=np.void(pickled_config))
    fileh5.close()