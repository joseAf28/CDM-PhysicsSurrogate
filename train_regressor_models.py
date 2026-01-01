import numpy as np
import yaml
import torch 
import math 
import h5py
import pickle
from joblib import Parallel, delayed

import src.models as models
import src.dataset as ds
import src.data_preparation as data_prep
# import src.inference_regressor as constraints
import src.train_regressor as train



# def procedure_one(config, device, seed):
    
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     regressor = train.train_model(seed, device, config)
    
#     input_size = config['model']['x_dim']
#     output_size = config['model']['y_dim']
#     hidden_sizes = config['model']['hidden_size']
    
#     data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
#     X_test_scaled, y_test_scaled = data_dict['test']
#     scaler_X, scaler_Y = data_dict['scalers']
    
#     solver = constraints.ProjectionSolver(scaler_X, scaler_Y, x_dim=input_size, p_dim=output_size)
#     y_pred, p_pred = solver.solve_batch(regressor, X_test_scaled[:,:], scaler_Y)
    
#     loss_net_pred = ((y_pred - y_test_scaled)**2).mean()
#     loss_proj_pred = ((p_pred - y_test_scaled)**2).mean()
    
#     return (seed, ratio, loss_net_pred, loss_proj_pred)


def procedure_one(config, device, seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    regressor = train.train_model(seed, device, config)
    ### save model
    torch.save(regressor.state_dict(), f"models_regressor_saved/regressor_inference_{seed}.pth")



if __name__ == "__main__":
    
    device = torch.device('cpu')
    config_file = "config_regressor.yaml"

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