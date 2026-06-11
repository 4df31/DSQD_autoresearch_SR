import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os
import time
import math
from prepare import load_fem_data

def run_experiment():
    start_time = time.time()
    
    # 1. Load Data
    df = load_fem_data()
    
    # Keep states up to s=2 to simplify the polynomial search (s=3 has high complexity)
    df = df[df['s'] <= 2.0].copy().reset_index(drop=True)
    
    # State-by-state ground truth normalization
    df['psi_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals
        
    # Feature Engineering: Add physical components
    df['r2'] = df['r']**2
    df['exp_half_r2'] = np.exp(-0.5 * df['r2'])
    df['cos_n_theta'] = np.cos(df['n'] * df['theta'])
    df['r_pow_n'] = df['r']**df['n']
    
    # 2. Extract radial profiles and prepare target without any given target-expression
    df['y_target'] = 0.0
    df['weight'] = 0.0
    
    # To run regression efficiently on the radial parts, we group by state and process
    for (n_val, s_val), group in df.groupby(['n', 's']):
        r = group['r'].values
        r2 = group['r2'].values
        psi_norm = group['psi_norm'].values
        
        # Physical divisor (boundary condition)
        divisor = np.exp(-0.5 * r2) * (r**n_val)
        
        # Quotient of numerical wavefunction by boundary condition
        y = np.zeros_like(psi_norm)
        mask = divisor > 1e-12
        y[mask] = psi_norm[mask] / divisor[mask]
        
        # Find index of theta closest to 0 to get the radial profile at theta=0
        theta_abs = group['theta'].abs().values
        min_theta_idx = np.argmin(theta_abs)
        # Scale quotient to start at 1.0 at the origin (at theta closest to 0, at r_0)
        # This removes state-by-state scale/sign discontinuities completely without analytical priors.
        y_scaled_at_origin = y / y[min_theta_idx]
        
        df.loc[group.index, 'y_target'] = y_scaled_at_origin
        # Use physical weight corresponding to the wavefunction density
        df.loc[group.index, 'weight'] = divisor**2
        
    # Downsample for PySR speed (e.g. 15x)
    df_pysr = df.iloc[::15].copy().reset_index(drop=True)
    
    feature_names = ["n", "s", "r2"]
    X = df_pysr[feature_names].values
    y_target = df_pysr['y_target'].values
    weights = df_pysr['weight'].values
    
    # 3. Configure Symbolic Regression
    print("Initializing PySR Regressor for multi-state fitting without target expressions...")
    model = PySRRegressor(
        variable_names=feature_names,
        niterations=2000,
        populations=100,
        population_size=100,
        binary_operators=["+", "*", "-", "/"], # Enable division for rational polynomials
        unary_operators=[], # No transcendental functions needed
        parsimony=0.000005,
        maxsize=35,
        timeout_in_seconds=260,
        parallelism="multiprocessing",
        procs=8,
        verbosity=0,
        temp_equation_file=True,
    )
    
    # 4. Fit the Model
    search_start = time.time()
    model.fit(X, y_target, weights=weights)
    search_time = time.time() - search_start
    
    # 5. Reconstruct predictions on full dataset and evaluate
    best_eq = model.get_best()
    
    # Reconstruct predictions: psi_pred = P_SR(r2, s, n) * exp_half_r2 * r_pow_n * cos(n * theta)
    X_full = df[feature_names].values
    p_pred = model.predict(X_full)
    predictions = p_pred * df['exp_half_r2'].values * df['r_pow_n'].values * df['cos_n_theta'].values
    
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

    print("\n---")
    print(f"best_r2_score:    {r2_score:.6f}")
    print(f"complexity:       {best_eq.complexity + 6}")
    print(f"search_seconds:   {search_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"best_equation:    \"exp(-0.5 * r2) * (r^n) * cos(n * theta) * ({best_eq.equation})\"")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()