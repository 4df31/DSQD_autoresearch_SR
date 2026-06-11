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
    
    # Filter strictly for the first 2 eigenfunctions varying s and n (4 states total)
    df = df[(df['s'] <= 1.0) & (df['n'] <= 1.0)].copy()
    
    # State-by-state ground truth normalization
    df['psi_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals
        
    X = df[['r', 'theta', 'n', 's']].values
    y_target = df['psi_norm'].values
    
    # 2. Configure Symbolic Regression
    print("Initializing PySR Regressor for multi-state fitting...")
    model = PySRRegressor(
        variable_names=["r", "theta", "n", "s"],
        niterations=100,
        populations=20,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp", "square", "sqrt", "cos", "sin"],
        parsimony=0.005,
        maxsize=35,
        timeout_in_seconds=280,
        parallelism="multiprocessing",
        procs=4,
        verbosity=0,
        temp_equation_file=True,
    )
    
    # 3. Fit the Model
    search_start = time.time()
    model.fit(X, y_target)
    search_time = time.time() - search_start
    
    # 4. Extract Results & Evaluate
    best_eq = model.get_best()
    predictions = model.predict(X)
    
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
    
    # 5. Output format strictly matching program.md requirements
    print("\n---")
    print(f"best_r2_score:    {r2_score:.6f}")
    print(f"complexity:       {best_eq.complexity}")
    print(f"search_seconds:   {search_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            vram = 0.0
    except ImportError:
        vram = 0.0
        
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"best_equation:    \"{best_eq.equation}\"")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()
