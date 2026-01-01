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



def procedure_one(config, device, seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    pcdae = train.train_EBM_model(seed, device, config)
    
    ### save model
    torch.save(pcdae.state_dict(), f"models_ebm_pcdae_saved/ebm_pcdae_cosine_inference_{seed}.pth")



if __name__ == "__main__":
    
    device = torch.device('cpu')
    config_file = "config_ebm_pcdae.yaml"
    
    seed_vec = np.arange(1, 400, 40, dtype=int)

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
    for seed in seed_vec:
        condition_vec.append(seed)
        
    results = Parallel(n_jobs=-1, backend="loky")(delayed(procedure_one)(config, device, condition) for condition in condition_vec)