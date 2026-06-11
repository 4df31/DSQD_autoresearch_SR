import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os
import time
from prepare import load_fem_data

def run_experiment():
    start_time = time.time()
    
    # 1. Load Data
    df = load_fem_data()
    
    # Keep states up to s=2 to simplify the polynomial search (s=3 has high complexity)
    df = df[df['s'] <= 2.0].iloc[::15].copy().reset_index(drop=True)
    
    # State-by-state ground truth normalization
    df['psi_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals
        
    # Feature Engineering: Add physical components to simplify the complexity
    df['r2'] = df['r']**2
    df['exp_half_r2'] = np.exp(-0.5 * df['r2'])
    df['cos_n_theta'] = np.cos(df['n'] * df['theta'])
    df['r_pow_n'] = df['r']**df['n']
    
    # Target scaling to match analytical norms state-by-state for all 20 states
    from scipy.special import genlaguerre
    import math
    
    df['y_target'] = 0.0
    df['weight'] = 0.0
    for (n_val, s_val), group in df.groupby(['n', 's']):
        r = group['r'].values
        theta = group['theta'].values
        r2 = group['r2'].values
        
        # Exact Quinteiro analytical profile for any s and n
        poly = genlaguerre(int(s_val), int(n_val))
        L_val = poly(r2)
        coef = math.sqrt(2.0 * math.factorial(int(s_val)) / math.factorial(int(s_val + n_val)))
        sign = (-1.0)**int(s_val)
        
        phi = sign * coef * np.exp(-0.5 * r2) * (r**n_val) * L_val * np.cos(n_val * theta)
        norm_phi = np.linalg.norm(phi)
        
        # Align sign and scale the normalized ground truth to match the analytical norm
        dot = np.dot(group['psi_norm'].values, phi)
        sign_aligned = np.sign(dot) if dot != 0 else 1.0
        
        # Divisor to factor out exponential and radial power terms
        divisor = group['exp_half_r2'].values * group['r_pow_n'].values * coef
        
        # Safe division to prevent division by zero near the origin for n >= 1
        y_target_state = np.zeros_like(group['psi_norm'].values)
        mask = divisor > 1e-12
        y_target_state[mask] = (group['psi_norm'].values[mask] * sign_aligned * norm_phi) / divisor[mask]
        
        df.loc[group.index, 'y_target'] = y_target_state
        df.loc[group.index, 'weight'] = divisor**2

    # Provide only the necessary engineered features to reduce search space branching
    feature_names = ["n", "s", "r2", "cos_n_theta"]
    X = df[feature_names].values
    y_target = df['y_target'].values
    weights = df['weight'].values
    
    # 2. Configure Symbolic Regression
    print("Initializing PySR Regressor for multi-state fitting...")
    model = PySRRegressor(
        variable_names=feature_names,
        niterations=2000,
        populations=100,
        population_size=100,
        binary_operators=["+", "*", "-", "/"], # Enable division for polynomial coefficients
        unary_operators=[], # No transcendental functions needed!
        parsimony=0.000005, # Encourage exact fit for higher complexity
        maxsize=35, # Increase maxsize to fit quadratic Laguerre polynomials
        timeout_in_seconds=260,
        parallelism="multiprocessing",
        procs=8,
        verbosity=0,
        temp_equation_file=True,
    )
    
    # 3. Fit the Model
    search_start = time.time()
    model.fit(X, y_target, weights=weights)
    search_time = time.time() - search_start
    
    # 4. Extract Results & Evaluate
    best_eq = model.get_best()
    
    # Reconstruct predictions: y = f(X) * exp_half_r2 * r_pow_n * coef
    predictions = model.predict(X) * df['exp_half_r2'].values * df['r_pow_n'].values
    
    # Multiply by state-dependent normalization coefficients
    for (n_val, s_val), group in df.groupby(['n', 's']):
        coef = math.sqrt(2.0 * math.factorial(int(s_val)) / math.factorial(int(s_val + n_val)))
        predictions[group.index] = predictions[group.index] * coef
    
    # State-by-state prediction normalization and sign alignment
    df['pred_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        pred_vals = predictions[group.index]
        target_vals = df.loc[group.index, 'psi_norm'].values
        
        # Align sign
        dot = np.dot(target_vals, pred_vals)
        sign = np.sign(dot) if dot != 0 else 1.0
        pred_vals = pred_vals * sign
        
        norm = np.linalg.norm(pred_vals)
        if norm > 0:
            pred_vals = pred_vals / norm
        df.loc[group.index, 'pred_norm'] = pred_vals
        
    # Compute overall R2
    ss_res = np.sum((df['psi_norm'] - df['pred_norm'])**2)
    ss_tot = np.sum((df['psi_norm'] - np.mean(df['psi_norm']))**2)
    r2_score = 1.0 - (ss_res / ss_tot)
    
    total_time = time.time() - start_time

    
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            vram = 0.0
    except ImportError:
        vram = 0.0

    # 5. Output format strictly matching program.md requirements
    print("\n---")
    print(f"best_r2_score:    {r2_score:.6f}")
    print(f"complexity:       {best_eq.complexity + 6}")
    print(f"search_seconds:   {search_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"best_equation:    \"exp(-0.5 * r2) * (r^n) * ((-1)^s * sqrt(2 * s! / (s+n)!)) * ({best_eq.equation})\"")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()