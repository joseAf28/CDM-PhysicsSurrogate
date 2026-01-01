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
import src.inference_regressor as constraints
import src.inference_pcdae as inference
import src.train_regressor as train



def procedure_one(config, device, condition):
    
    seed, hidden_size = condition
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config['model']['hidden_size'] = [hidden_size, hidden_size]
    
    regressor = train.train_model(seed, device, config)
    
    input_size = config['model']['x_dim']
    output_size = config['model']['y_dim']
    hidden_sizes = config['model']['hidden_size']
    
    data_dict = data_prep.prepare_data_model(seed, config['data']["data_path"], config['training']["ratio_test_val_train"])
    X_test_scaled, y_test_scaled = data_dict['test']
    scaler_X, scaler_Y = data_dict['scalers']
    
    solver = constraints.ProjectionSolver(scaler_X, scaler_Y, x_dim=input_size, p_dim=output_size)
    y_pred, p_pred = solver.solve_batch(regressor, X_test_scaled[:,:], scaler_Y)
    
    loss_net_pred = ((y_pred - y_test_scaled)**2).mean()
    loss_proj_pred = ((p_pred - y_test_scaled)**2).mean()
    
    constant_net = inference.get_constraints_scalar(torch.from_numpy(X_test_scaled), torch.from_numpy(y_pred), scaler_X, scaler_Y).pow(2).mean()
    constant_proj = inference.get_constraints_scalar(torch.from_numpy(X_test_scaled), torch.from_numpy(p_pred), scaler_X, scaler_Y).pow(2).mean()
    
    return (seed, hidden_size, loss_net_pred, loss_proj_pred, constant_net, constant_proj)


if __name__ == "__main__":
    
    device = torch.device('cpu')
    config_file = "config_regressor.yaml"
    output_file = "results/scaling_params_regressorV3.h5"
    
    seed_vec = np.arange(1, 400, 40, dtype=int)
    hidden_sizes = [50, 65, 80, 95, 110, 135, 160, 180, 200]
    
    # seed_vec = [1, 20]
    # hidden_sizes = [85, 100]
    
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
    seed_vec, hidden_vec, loss_net_vec, loss_proj_vec, constant_net, constant_proj = zip(*results)
    
    pickled_config = pickle.dumps(config)
    
    fileh5 = h5py.File(output_file, 'w')
    group = fileh5.require_group("results")
    group.create_dataset("seed", data=seed_vec)
    group.create_dataset("hidden", data=hidden_vec)
    group.create_dataset("loss_net", data=loss_net_vec)
    group.create_dataset("loss_proj", data=loss_proj_vec)
    group.create_dataset("constant_net", data=constant_net)
    group.create_dataset("constant_proj", data=constant_proj)
    group.create_dataset("config", data=np.void(pickled_config))
    fileh5.close()