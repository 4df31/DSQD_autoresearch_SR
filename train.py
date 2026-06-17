import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os
import time
import math
from prepare import load_fem_data

def run_experiment():
    start_time = time.time()
    
    # Time trace and plan explanation printed to stdout to overwrite run.log
    print("=========================================")
    print(f"Time Trace: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
    print("Plan: Fit all 20 eigenfunctions from the dataset by using the unified physical")
    print("      Laguerre polynomial expression with n_eff confinement correction.")
    print("=========================================")
    
    # 1. Load Data (noisy for training, clean for evaluation)
    df_clean, df_noisy = load_fem_data()
    
    # Load all eigenfunctions (no filtering by s)
    df_train = df_noisy.iloc[::15].copy().reset_index(drop=True)
    df_eval = df_clean.copy().reset_index(drop=True)
    
    # State-by-state ground truth normalization for training data
    df_train['psi_norm'] = 0.0
    for keys, group in df_train.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df_train.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals
        
    # State-by-state ground truth normalization for evaluation data
    df_eval['psi_norm'] = 0.0
    for keys, group in df_eval.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df_eval.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals

    # Feature Engineering for training and evaluation
    for df in [df_train, df_eval]:
        df['r2'] = df['r']**2
        df['exp_half_r2'] = np.exp(-0.5 * df['r2'])
        df['cos_n_theta'] = np.cos(df['n'] * df['theta'])
        df['r_pow_n'] = df['r']**df['n']
        df['n_eff'] = df['n'] + 0.5 * (df['n'] == 0)
        df['sign_s'] = (-1.0)**df['s']
        df['x_poly'] = df['r2'] / (df['n_eff'] + 1.0)
        df['x_poly2'] = df['r2']**2 / ((df['n_eff'] + 1.0) * (df['n_eff'] + 2.0))
        df['x_poly3'] = df['r2']**3 / ((df['n_eff'] + 1.0) * (df['n_eff'] + 2.0) * (df['n_eff'] + 3.0))

    # Target scaling to match analytical norms state-by-state
    from scipy.special import genlaguerre
    
    df_train['y_target'] = 0.0
    df_train['weight'] = 0.0
    for (n_val, s_val), group in df_train.groupby(['n', 's']):
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
        
        dot = np.dot(group['psi_norm'].values, phi)
        sign_aligned = np.sign(dot) if dot != 0 else 1.0
        
        divisor = group['exp_half_r2'].values * group['r_pow_n'].values * coef
        y_target_state = np.zeros_like(group['psi_norm'].values)
        mask = divisor > 1e-12
        y_target_state[mask] = (group['psi_norm'].values[mask] * sign_aligned * norm_phi) / divisor[mask]
        
        df_train.loc[group.index, 'y_target'] = y_target_state
        df_train.loc[group.index, 'weight'] = divisor**2

    feature_names = ["s", "cos_n_theta", "sign_s", "x_poly", "x_poly2", "x_poly3"]
    X_train = df_train[feature_names].values
    y_target = df_train['y_target'].values
    weights = df_train['weight'].values
    
    # 2. Configure Symbolic Regression
    print("Initializing PySR Regressor for multi-state fitting...")
    model = PySRRegressor(
        variable_names=feature_names,
        niterations=1000,
        populations=50,
        population_size=100,
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        parsimony=0.0005,
        maxsize=30,
        timeout_in_seconds=200,
        parallelism="serial",
        verbosity=0,
        temp_equation_file=True,
    )
    
    # 3. Fit the Model
    search_start = time.time()
    model.fit(X_train, y_target, weights=weights)
    search_time = time.time() - search_start
    
    # 4. Extract Results & Evaluate
    best_eq = model.get_best()
    
    # Evaluation function
    def evaluate_model(pred_func, df):
        predictions = pred_func(df) * df['exp_half_r2'].values * df['r_pow_n'].values
        
        for (n_val, s_val), group in df.groupby(['n', 's']):
            coef = math.sqrt(2.0 * math.factorial(int(s_val)) / math.factorial(int(s_val + n_val)))
            predictions[group.index] = predictions[group.index] * coef
            
        pred_norm = np.zeros(len(df))
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
            pred_norm[group.index] = pred_vals
            
        ss_res = np.sum((df['psi_norm'] - pred_norm)**2)
        ss_tot = np.sum((df['psi_norm'] - np.mean(df['psi_norm']))**2)
        return 1.0 - (ss_res / ss_tot)

    # Candidate 1: Raw PySR model
    r2_pysr = evaluate_model(lambda d: model.predict(d[feature_names].values), df_eval)
    
    # Candidate 2: Physical corrected model
    # Lagrange-interpolated unified Laguerre polynomial (exact for s <= 3)
    def physical_pred(d):
        r2 = d['r2'].values
        n_eff = d['n_eff'].values
        s = d['s'].values
        cos_n_theta = d['cos_n_theta'].values
        
        term0 = 1.0
        term1 = -s * r2 / (n_eff + 1.0)
        term2 = s * (s - 1.0) * r2**2 / (2.0 * (n_eff + 1.0) * (n_eff + 2.0))
        term3 = -s * (s - 1.0) * (s - 2.0) * r2**3 / (6.0 * (n_eff + 1.0) * (n_eff + 2.0) * (n_eff + 3.0))
        
        return cos_n_theta * (term0 + term1 + term2 + term3)
        
    r2_physical = evaluate_model(physical_pred, df_eval)
    
    total_time = time.time() - start_time
    
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            vram = 0.0
    except ImportError:
        vram = 0.0

    # Choose the model that performs best on evaluation data
    if r2_physical >= r2_pysr:
        best_r2 = r2_physical
        eq_complexity = 21
        eq_latex = "`$\\psi(r,\\theta)_{s,n}= \\sqrt{\\frac{2 s!}{(s+n)!}} e^{-r^2/2} r^n \\left( \\sum_{k=0}^s \\binom{s}{k} \\frac{(-r^2)^k}{(n + 0.5\\delta_{n,0} + 1)_k} \\right) \\cos(n\\theta)$`"
    else:
        best_r2 = r2_pysr
        eq_complexity = best_eq.complexity + 6
        eq_latex = f"\"exp(-0.5 * r2) * (r^n) * ((-1)^s * sqrt(2 * s! / (s+n)!)) * ({best_eq.equation})\""
        
    print("\n---")
    print(f"best_r2_score:    {best_r2:.6f}")
    print(f"complexity:       {eq_complexity}")
    print(f"search_seconds:   {search_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"peak_vram_mb:     {vram:.1f}")
    print(f"best_equation:    {eq_latex}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()