import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from torch.func import jacrev, vmap


def inference_blind_adaptive_ODE(model, x, y_init, eta=0.1, max_steps=1000, 
                                 eps_conv=1e-3, eps_clip=None, adaptative=False):
    """
    Implements the EBM ODE Flow (Time-Independent).
    
    Formalism:
        y_{k+1} = (1 - eta) * y_k + eta * g(y_k)
        
    Args:
        model: Function g_phi(x, y) -> returns concatenated [x_recon, y_recon]
        x: Condition (Batch, Dim_X)
        y_init: Initial Noisy Guess (Batch, Dim_Y)
        eta: The constant 'rate' or step size (equivalent to geometric decay).
        max_steps: Safety limit to prevent infinite loops.
        eps_conv: Convergence threshold for the residual.
    """
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    # Track which samples in the batch have converged
    # (False = active, True = converged)
    converged_mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    
    for step in range(max_steps):
        # Check if whole batch is done
        if converged_mask.all():
            print("steps: ", step)
            break
            
        # Model Prediction
        with torch.no_grad():
            # we compute all and mask updates below
            recon, _ = model(x, y) 
            
        y_hat = recon[:, x_dim:] # Extract predicted y
        
        # Calculate Residual (Direction Vector)
        # Vector pointing from y_current -> y_clean
        residual = y_hat - y 
        
        # Check Convergence (Batch-wise)
        # Compute L2 norm per sample (dim=1)
        norms = torch.norm(residual, p=2, dim=1)
        
        # Update convergence mask
        newly_converged = norms < eps_conv
        converged_mask = converged_mask | newly_converged
        
        # Optional: Clipping (Safety Rail)
        if eps_clip is not None:
            # We clip the residual magnitude, preserving direction
            factor = torch.clamp(eps_clip / (norms + 1e-8), max=1.0).unsqueeze(1)
            residual = residual * factor
            
        # Update Rule (Fixed Point Iteration)
        # y_new = y_old + eta * (y_hat - y_old)
        # Only update samples that haven't converged yet
        active_indices = ~converged_mask
        if active_indices.any():
            
            if not adaptative:
                y[active_indices] += eta * residual[active_indices]
            else:
                eta_adaptative = eta * torch.norm(residual[active_indices], p=1, dim=1) / (torch.norm(y[active_indices], p=1, dim=1) + 1e-8)
                y[active_indices] += eta_adaptative.reshape(-1, 1) * residual[active_indices]
            
    return y



def inference_scheduled_ODE(model, x, y_init, noise_schedule, 
                            steps_per_level=1, eps_clip=None):
    """
    Implements the Time-Dependent Probability Flow ODE.
    
    Formalism:
        y_{i-1} = y_i + (sigma_{i-1} - sigma_i) * (y_i - y_hat) / sigma_i
    
    Args:
        noise_schedule: List of sigmas [sigma_max, ..., sigma_min]
        steps_per_level: Usually 1 for standard Euler ODE. 
                         If >1, it behaves like a predictor-corrector or repeated denoising.
    """
    y = y_init.clone().detach()
    batch_size = x.size(0)
    x_dim = x.size(1)
    
    counter_iter = 0
    
    # Iterate through the schedule (High Sigma -> Low Sigma)
    for i in range(len(noise_schedule) - 1):
        sigma_curr = noise_schedule[i]
        sigma_next = noise_schedule[i+1] # The target sigma for this step
        
        # Calculate the ODE step size derived from the formalism
        # eta = (sigma_curr - sigma_next) / sigma_curr
        # Note: If sigma_next is 0, this simplifies to eta = 1 (jump to solution)
        if sigma_curr == 0:
            step_rate = 1.0
        else:
            step_rate = (sigma_curr - sigma_next) / sigma_curr
        
        # Create noise tensor for the model (Batch, 1)
        noise_tensor = torch.full((batch_size, 1), sigma_curr, device=x.device)
        
        for _ in range(steps_per_level):
            with torch.no_grad():
                recon, _ = model(x, y, noise_tensor)
            
            y_hat = recon[:, x_dim:]
            residual = y_hat - y
            
            # Optional Clipping
            if eps_clip is not None:
                norms = torch.norm(residual, p=2, dim=1, keepdim=True)
                factor = torch.clamp(eps_clip / (norms + 1e-8), max=1.0)
                residual = residual * factor

            # ODE Update Step
            # y_{t-1} = y_t - step_rate * (y_t - y_hat)
            # which is: y_{t-1} = y_t + step_rate * (y_hat - y_t)
            y = y + step_rate * residual
            
            counter_iter += 1
    
    
    # print("counter_iter: ", counter_iter)
    
    return y




####! samplers with constraints

def constraints_func(x, y, scaler_X, scaler_Y):
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
            (-P_calc + P)/scaler_X.scale_[0],
            (-I_calc + I)/scaler_X.scale_[1],
            (-ne_calc + ne_model)/scaler_Y.scale_[4],
        ], dim=1)  # (B, 3)
        
        return h


def constraints_func_loss(x, y, scaler_X, scaler_Y):
        P = x[:, 0] * scaler_X.scale_[0] + scaler_X.mean_[0]
        I = x[:, 1] * scaler_X.scale_[1] + scaler_X.mean_[1]
        R = x[:, 2] * scaler_X.scale_[2] + scaler_X.mean_[2]
        # pressure
        Tg  = y[:, 11] * scaler_Y.scale_[11] + scaler_Y.mean_[11]                   # gas temperature
        kb  = 1.380649e-23
        
        conc_sum = y[:, 0] * scaler_Y.scale_[0] + scaler_Y.mean_[0]
        for i in range(1, 11):
            conc_sum += y[:, i] * scaler_Y.scale_[i] + scaler_Y.mean_[i]
        
        P_calc = conc_sum * Tg * kb           # shape (B,)
        
        # electron density
        ne_model = y[:, 16] * scaler_Y.scale_[16] + scaler_Y.mean_[16]
        ne_calc  = y[:, 4] * scaler_Y.scale_[4] + scaler_Y.mean_[4] + y[:, 7]  * scaler_Y.scale_[7] + scaler_Y.mean_[7] - y[:,8]  * scaler_Y.scale_[8] - scaler_Y.mean_[8]
        
        # current
        vd = y[:, 14] * scaler_Y.scale_[14] + scaler_Y.mean_[14]
        e  = 1.602176634e-19
        I_calc = e * ne_model * vd * torch.pi * R*R
        
        # stack residuals
        h = torch.stack([
            (-P_calc + P)/scaler_X.scale_[0],
            (-I_calc + I)/scaler_X.scale_[1],
            (-ne_calc + ne_model)/scaler_Y.scale_[4],
        ], dim=1)  # (B, 3)
        
        return h



def get_constraints_scalar(x, y, scaler_X, scaler_Y, grad_flag=False):
    
    y_scaled = torch.tensor(scaler_Y.scale_ * y.cpu().numpy() + scaler_Y.mean_, 
                            dtype=y.dtype, device=y.device, requires_grad=grad_flag)
    x_scaled = torch.tensor(scaler_X.scale_ * x.cpu().numpy() + scaler_X.mean_, 
                            dtype=x.dtype, device=x.device)
    
    return constraints_func(x_scaled, y_scaled, scaler_X, scaler_Y)





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

