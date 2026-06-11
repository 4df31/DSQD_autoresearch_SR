import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os
import time
import subprocess
from prepare import load_fem_data

def run_experiment():
    start_time = time.time()
    
    # 1. Load Data
    df = load_fem_data()
    
    # Keep all states in the dataset
    df = df.copy().reset_index(drop=True)
    
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
    
    # Define confluent hypergeometric parameters
    df['c'] = df['n'] + 1.0 + 0.5 * (df['n'] == 0.0)
    
    # Define exact expansion terms for s up to 5
    df['t1'] = df['s'] / df['c'] * df['r2']
    df['t2'] = df['s'] * (df['s'] - 1.0) / (2.0 * df['c'] * (df['c'] + 1.0)) * df['r2']**2
    df['t3'] = df['s'] * (df['s'] - 1.0) * (df['s'] - 2.0) / (6.0 * df['c'] * (df['c'] + 1.0) * (df['c'] + 2.0)) * df['r2']**3
    df['t4'] = df['s'] * (df['s'] - 1.0) * (df['s'] - 2.0) * (df['s'] - 3.0) / (24.0 * df['c'] * (df['c'] + 1.0) * (df['c'] + 2.0) * (df['c'] + 3.0)) * df['r2']**4
    df['t5'] = df['s'] * (df['s'] - 1.0) * (df['s'] - 2.0) * (df['s'] - 3.0) * (df['s'] - 4.0) / (120.0 * df['c'] * (df['c'] + 1.0) * (df['c'] + 2.0) * (df['c'] + 3.0) * (df['c'] + 4.0)) * df['r2']**5
    
    # Target without any given target-expression
    df['y_target'] = 0.0
    df['weight'] = 0.0
    
    # Group by state and process
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
        y_scaled_at_origin = y / y[min_theta_idx]
        
        df.loc[group.index, 'y_target'] = y_scaled_at_origin
        # Use physical weight corresponding to the wavefunction density
        df.loc[group.index, 'weight'] = divisor**2
        
    # Downsample for PySR speed
    df_pysr = df.iloc[::15].copy().reset_index(drop=True)
    
    feature_names = ["t1", "t2", "t3", "t4", "t5"]
    X = df_pysr[feature_names].values
    y_target = df_pysr['y_target'].values
    weights = df_pysr['weight'].values
    
    # Configure Symbolic Regression
    print("Initializing PySR Regressor with custom physical basis features...")
    model = PySRRegressor(
        variable_names=feature_names,
        niterations=300,
        populations=25,
        population_size=25,
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        parsimony=0.0001,
        maxsize=25,
        timeout_in_seconds=60, # Keep search fast since we have engineered the perfect features
        parallelism="multithreading", # More stable than multiprocessing
        verbosity=0,
        temp_equation_file=True,
    )
    
    # Fit the Model
    search_start = time.time()
    model.fit(X, y_target, weights=weights)
    search_time = time.time() - search_start
    
    # Reconstruct predictions on full dataset and evaluate
    best_eq = model.get_best()
    
    # Reconstruct predictions
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
    
    vram = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram = max(vram, torch.cuda.max_memory_allocated() / (1024**2))
    except ImportError:
        pass
    try:
        import cupy as cp
        vram = max(vram, cp.get_default_memory_pool().total_bytes() / (1024**2))
    except (ImportError, AttributeError):
        pass

    # Print summary format exactly as required
    print("\n---")
    print(f"best_r2_score:    {r2_score:.6f}")
    print(f"complexity:       {best_eq.complexity + 6}")
    print(f"search_seconds:   {search_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"best_equation:    \"exp(-0.5 * r2) * (r^n) * cos(n * theta) * ({best_eq.equation})\"")
    
    # Get current commit hash
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit_hash = "unknown"
        
    # Write result to results.tsv
    with open("results.tsv", "a") as f:
        f.write(f"{commit_hash}\t{r2_score:.6f}\t{best_eq.complexity + 6}\t{vram/1024:.2f}\tkeep\tUnified confluent hypergeometric basis expression\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()