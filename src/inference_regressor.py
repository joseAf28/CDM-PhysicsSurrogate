import torch
import torch.nn as nn
import casadi as ca
import numpy as np
import yaml

import src.models as models
import src.data_preparation as data_prep

# This helper function remains exactly the same
def setup_pytorch_model(model_path, input_size, ouput_size, hidden_sizes):
    model_loaded = models.Regressor(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)
    model_loaded.load_state_dict(torch.load(model_path, weights_only=False))
    model_loaded.eval() # Set to evaluation mode
    print("Pre-trained PyTorch model loaded successfully.\n")
    return model_loaded


# This constraint function also remains exactly the same
def define_g_constraint(scaler_X, scaler_Y, x_dim=3, p_dim=17):
    # Using SX is common with the nlpsol interface
    x_sym = ca.SX.sym('x', x_dim)
    p_sym = ca.SX.sym('p', p_dim)
    
    P = x_sym[0] * scaler_X.scale_[0] + scaler_X.mean_[0]
    I = x_sym[1] * scaler_X.scale_[1] + scaler_X.mean_[1]
    R = x_sym[2] * scaler_X.scale_[2] + scaler_X.mean_[2]
    kb, e = 1.380649e-23, 1.602176634e-19
    
    Tg = p_sym[11] * scaler_Y.scale_[11] + scaler_Y.mean_[11]
    
    conc_scaled_sum = 0
    for i in range(11):
        conc_scaled_sum += (p_sym[i] * scaler_Y.scale_[i] + scaler_Y.mean_[i])
        
    P_calc = conc_scaled_sum * Tg * kb
    
    ne_model = p_sym[16]* scaler_Y.scale_[16] + scaler_Y.mean_[16]
    ne_calc = p_sym[4] * scaler_Y.scale_[4] + scaler_Y.mean_[4] + p_sym[7] * scaler_Y.scale_[7] + scaler_Y.mean_[7] - (p_sym[8] * scaler_Y.scale_[8] + scaler_Y.mean_[8])
    
    vd = p_sym[14] * scaler_Y.scale_[14] + scaler_Y.mean_[14]
    
    I_calc = e * ne_model * vd * ca.pi * R**2
    
    P_law = (P_calc - P) / scaler_X.scale_[0]
    I_law = (I_calc - I) / scaler_X.scale_[1]
    ne_law = (ne_calc - ne_model) / scaler_Y.scale_[4]
    
    h = ca.vertcat(P_law, I_law, ne_law)
    return ca.Function('g_casadi', [x_sym, p_sym], [h])


class ProjectionSolver:
    def __init__(self, scaler_X, scaler_Y, x_dim=3, p_dim=17):
        
        self.p_dim = p_dim
        g_casadi = define_g_constraint(scaler_X, scaler_Y, x_dim, p_dim)
        
        # 'p_var' is our main decision variable (x in the NLP sense)
        p_var = ca.SX.sym('p_var', p_dim)
        
        # Define symbolic placeholders for data that changes on each iteration.
        # These are the "parameters" of the NLP.
        x_param = ca.SX.sym('x_param', x_dim)
        y_target_param = ca.SX.sym('y_target_param', p_dim)

        objective = ca.sumsqr(p_var - y_target_param)
        constraints = g_casadi(x_param, p_var)
        
        # 'x': decision variables
        # 'f': objective function
        # 'g': constraint functions
        # 'p': parameters (all fixed data that changes between solves)
        nlp_problem = {
            'x': p_var,
            'f': objective,
            'g': constraints,
            'p': ca.vcat([x_param, y_target_param])
        }
        
        options = {'ipopt.print_level' : 0, 'ipopt.sb' : "no", 'ipopt.tol' : 1e-10, 'print_time': 0,
                    'ipopt.max_iter' : 500, 'ipopt.acceptable_tol' : 1e-10}
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_problem, options)
        
        # Define the constant bounds for the equality constraints g(x,p) = 0
        self.lbg = [0] * constraints.shape[0]
        self.ubg = [0] * constraints.shape[0]

    
    def solve_batch(self, pytorch_model, x_fixed_batch, scaler_Y):
        B = x_fixed_batch.shape[0]
        
        x_tensor = torch.from_numpy(x_fixed_batch).float()
        with torch.no_grad():
            y_target_scaled_batch = pytorch_model(x_tensor).numpy()
            
        p_optimal_scaled_batch = np.zeros((B, self.p_dim))
        
        for i in range(B):
            current_x_scaled = x_fixed_batch[i, :]
            current_y_target_scaled = y_target_scaled_batch[i, :]
            
            # Concatenate all parameter values into a single vector
            # in the same order as they were defined in 'p' in the NLP problem.
            param_values = np.concatenate([current_x_scaled, current_y_target_scaled])
            
            try:
                # Solve the problem for the current item
                # x0: initial guess for the decision variable 'x' (p_var)
                # p: numerical values for the parameters 'p'
                # lbg/ubg: lower/upper bounds for the constraints 'g'
                sol = self.solver(x0=current_y_target_scaled, 
                                p=param_values,
                                lbg=self.lbg, 
                                ubg=self.ubg)
                
                p_optimal_scaled_batch[i, :] = sol['x'].toarray().flatten()
                
                # # sol['g'] contains the values of the constraints at the solution
                # g_values = sol['g'].toarray().flatten()
                # total_violation = np.sum(np.abs(g_values))
                # print(f"Item {i}: Constraint Violation (Sum Abs) = {total_violation:.2e}")

            except Exception as e:
                print(f"Solver failed for item {i}: {e}")
                p_optimal_scaled_batch[i, :] = np.full(self.p_dim, np.nan)
        
        return y_target_scaled_batch, p_optimal_scaled_batch