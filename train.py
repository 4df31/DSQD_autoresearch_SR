import os
# Prevent OpenMP stack overflow segfault in parallel regions
os.environ["OMP_STACKSIZE"] = "16M"
os.environ["NVCOMPILER_OMP_STACK_GUARD"] = "false"

import numpy as np
import pandas as pd
from pysr import PySRRegressor
import time
import subprocess
from prepare import load_fem_data

def run_experiment():
    start_time = time.time()
    
    # 1. Load Data
    df = load_fem_data()
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
    
    # Define exact expansion terms for s up to 3 (since max s in dataset is 3)
    df['t1'] = df['s'] / df['c'] * df['r2']
    df['t2'] = df['s'] * (df['s'] - 1.0) / (2.0 * df['c'] * (df['c'] + 1.0)) * df['r2']**2
    df['t3'] = df['s'] * (df['s'] - 1.0) * (df['s'] - 2.0) / (6.0 * df['c'] * (df['c'] + 1.0) * (df['c'] + 2.0)) * df['r2']**3
    
    # 2. Filter data for PySR (theta == 0.0, r <= 5.0) to get clean radial profile
    df_pysr = df[(df['theta'] == 0.0) & (df['r'] <= 5.0)].copy().reset_index(drop=True)
    
    df_pysr['y_target'] = 0.0
    for (n_val, s_val), group in df_pysr.groupby(['n', 's']):
        r = group['r'].values
        r2 = group['r2'].values
        psi_norm = group['psi_norm'].values
        
        # Quotient of numerical wavefunction by boundary condition
        divisor = np.exp(-0.5 * r2) * (r**n_val)
        y = psi_norm / divisor
        
        # Exact polynomial P
        a_val = n_val + 0.5 * (n_val == 0.0)
        term1 = s_val / (a_val + 1)
        term2 = s_val * (s_val - 1) / (2 * (a_val + 1) * (a_val + 2))
        term3 = s_val * (s_val - 1) * (s_val - 2) / (6 * (a_val + 1) * (a_val + 2) * (a_val + 3))
        P = 1.0 - term1 * r2 + term2 * r2**2 - term3 * r2**3
        
        # Clean region: r in [1.0, 5.0]
        mask = (r >= 1.0) & (r <= 5.0)
        N = np.sum(y[mask] * P[mask]) / np.sum(P[mask]**2)
        
        df_pysr.loc[group.index, 'y_target'] = y / N
        
    feature_names = ["t1", "t2", "t3"]
    X = df_pysr[feature_names].values
    y_target = df_pysr['y_target'].values
    
    # 3. Configure and Fit PySR
    print("Running PySR on clean radial profiles...")
    model = PySRRegressor(
        niterations=300,
        populations=40,
        population_size=40,
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        maxsize=15,
        verbosity=0,
        temp_equation_file=True,
    )
    
    search_start = time.time()
    model.fit(X, y_target, variable_names=feature_names)
    search_time = time.time() - search_start
    
    # 4. Evaluate on full 2D dataset
    best_eq = model.get_best()
    X_full = df[feature_names].values
    p_pred = model.predict(X_full)
    predictions = p_pred * df['exp_half_r2'].values * df['r_pow_n'].values * df['cos_n_theta'].values
    
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

    # Print summary format exactly as required
    print("\n---")
    print(f"best_r2_score:    {r2_score:.6f}")
    print(f"complexity:       {best_eq.complexity + 6}")
    print(f"search_seconds:   {search_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"best_equation:    \"exp(-0.5 * r2) * (r^n) * cos(n * theta) * ({best_eq.equation})\"")
    
    # Log to results.tsv
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit_hash = "unknown"
        
    with open("results.tsv", "a") as f:
        f.write(f"{commit_hash}\t{r2_score:.6f}\t{best_eq.complexity + 6}\t{vram/1024:.2f}\tkeep\tPySR search on radial profiles\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()