import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from torch.func import jacrev, vmap



def inference_PCDAE_base_evolution(model, x, y_init, noise_schedule, steps_vec, \
                        step_size=1e-3, eps_conv=1e-3, eps_clip=5e-2):
    
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    y_evolution = torch.zeros((len(noise_schedule), y.size(0), y.size(1)))
    
    for idx, noise_level in enumerate(noise_schedule):
        
        noise_tensor = torch.full((batch_size, 1), noise_level, device=x.device)
        for _ in range(steps_vec[idx]):
            with torch.no_grad():
                recon, _ = model(x, y, noise_tensor)
                
            v_vec = recon[:,x_dim:] - y
            if torch.norm(v_vec) < eps_conv:
                break
            
            torch.clip(v_vec, -eps_clip, eps_clip, out=v_vec)
            y = y + step_size * v_vec
            
        y_evolution[idx, :,:] = y
        
    return y, y_evolution



def inference_EBM_PCDAE(model, x, y_init, steps_vec, \
                        step_size=1e-3, eps_conv=1e-3, eps_clip=5e-2):
    
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    for m in range(steps_vec):
        with torch.no_grad():
            recon, _ = model(x, y)
                
        v_vec = recon[:,x_dim:] - y
        if torch.norm(v_vec) < eps_conv:
            break
            
        torch.clip(v_vec, -eps_clip, eps_clip, out=v_vec)
        y = y + step_size * v_vec
    
    return y
    # h_vec = constraints_func(x, y.clone().detach())
    # return y, h_vec


def inference_PCDAE_base(model, x, y_init, noise_schedule, steps_vec, \
                        step_size=1e-3, eps_conv=1e-3, eps_clip=5e-2):
    
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    for idx, noise_level in enumerate(noise_schedule):
        
        noise_tensor = torch.full((batch_size, 1), noise_level, device=x.device)
        for _ in range(steps_vec[idx]):
            with torch.no_grad():
                recon, _ = model(x, y, noise_tensor)
                
            v_vec = recon[:,x_dim:] - y
            if torch.norm(v_vec) < eps_conv:
                break
            
            torch.clip(v_vec, -eps_clip, eps_clip, out=v_vec)
            y = y + step_size * v_vec
    
    return y



def constraints_func(x, y):
        P = x[:, 0] 
        I = x[:, 1]
        R = x[:, 2]
        # pressure
        Tg  = y[:, 11]                    # gas temperature
        kb  = 1.380649e-23
        conc = y[:, :11].sum(dim=1)       # sum of species 0…10
        P_calc = conc * Tg * kb           # shape (B,)
        
        # electron density
        ne_model = y[:, 16]
        ne_calc  = y[:, 4] + y[:, 7] - y[:, 8]
        
        # current
        vd = y[:, 14]
        e  = 1.602176634e-19
        I_calc = e * ne_model * vd * torch.pi * R*R
        
        # stack residuals
        h = torch.stack([
            -P_calc + P,
            -I_calc + I,
            -ne_calc + ne_model
        ], dim=1)  # (B, 3)
        
        return h


def grad_constraints_func(x, y, scaler_X, scaler_Y):
    
    y_scaled = torch.tensor(scaler_Y.scale_ * y.cpu().numpy() + scaler_Y.mean_, 
                            dtype=y.dtype, device=y.device, requires_grad=True)
    x_scaled = torch.tensor(scaler_X.scale_ * x.cpu().numpy() + scaler_X.mean_, 
                            dtype=x.dtype, device=x.device)
    
    def single_sample_func(x_single, y_single):
        return constraints_func(x_single.unsqueeze(0), y_single.unsqueeze(0)).squeeze(0)

    jacobian_func = vmap(jacrev(single_sample_func, argnums=1), in_dims=(0, 0))
    jacobian = jacobian_func(x_scaled, y_scaled)

    return jacobian




def inference_PCDAE_dual(model, x, y_init, noise_schedule, steps_vec, scaler_X, scaler_Y, \
                        step_size=1e-3, eps_conv=1e-3, eps_clip=5e-2, alpha_dual=1e-3, beta_dual=1.0):
    
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    h_vec = constraints_func(x, y)
    lambda_dual = torch.zeros_like(h_vec, device=h_vec.device)
    
    for idx, noise_level in enumerate(noise_schedule):
        noise_tensor = torch.full((batch_size, 1), noise_level)
        
        for _ in range(steps_vec[idx]):
            with torch.no_grad():
                recon, _ = model(x, y, noise_tensor)
                v_vec = recon[:, x_dim:] - y
            
            
            torch.clip(v_vec, -eps_clip, eps_clip, out=v_vec)
            
            grad_h_vec = grad_constraints_func(x, y, scaler_X, scaler_Y)
            h_vec = constraints_func(x, y).to(y.device)
            
            constraint_term = (grad_h_vec.detach() * (beta_dual * h_vec.unsqueeze(-1) + lambda_dual.unsqueeze(-1))).sum(dim=1)
            
            y = y + step_size * v_vec + alpha_dual * constraint_term
            lambda_dual = lambda_dual + alpha_dual * (constraints_func(x,y).to(y.device))
    
    return y



def inference_EBM_PCDAE_dual(model, x, y_init, steps_vec, scaler_X, scaler_Y, \
                        step_size=1e-3, eps_conv=1e-3, eps_clip=5e-2, alpha_dual=1e-3, beta_dual=1.0):
    
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    h_vec = constraints_func(x, y.clone().detach())
    lambda_dual = torch.zeros_like(h_vec, device=h_vec.device)
    
    
    for idx in range(steps_vec):

        with torch.no_grad():
            recon, _ = model(x, y)
            v_vec = recon[:, x_dim:] - y
        

        if torch.norm(v_vec) < eps_conv:
            break
        
        
        torch.clip(v_vec, -eps_clip, eps_clip, out=v_vec)
        y = y + step_size * v_vec 
        
        if torch.norm(v_vec) < 0.5: 
            grad_h_vec = grad_constraints_func(x, y.clone().detach(), scaler_X, scaler_Y)
            h_vec = constraints_func(x, y.clone().detach()).to(y.device)
            constraint_term = (grad_h_vec.detach() * h_vec.unsqueeze(-1)).sum(dim=1)
            
            y -= 1e-3 * constraint_term

    return y, h_vec