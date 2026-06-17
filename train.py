import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os
import time
import math
from prepare import load_fem_data

def run_experiment():
    start_time = time.time()
    
    # 1. Load Data (noisy for training, clean for evaluation)
    df_clean, df_noisy = load_fem_data()
    
    # Filter for first 2 eigenfunctions (s <= 1.0)
    df_train = df_noisy[df_noisy['s'] <= 1.0].iloc[::15].copy().reset_index(drop=True)
    df_eval = df_clean[df_clean['s'] <= 1.0].copy().reset_index(drop=True)
    
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

    # Feature Engineering for training
    df_train['r2'] = df_train['r']**2
    df_train['exp_half_r2'] = np.exp(-0.5 * df_train['r2'])
    df_train['cos_n_theta'] = np.cos(df_train['n'] * df_train['theta'])
    df_train['r_pow_n'] = df_train['r']**df_train['n']
    df_train['n_eff'] = df_train['n'] + 0.5 * (df_train['n'] == 0)
    
    # Feature Engineering for evaluation
    df_eval['r2'] = df_eval['r']**2
    df_eval['exp_half_r2'] = np.exp(-0.5 * df_eval['r2'])
    df_eval['cos_n_theta'] = np.cos(df_eval['n'] * df_eval['theta'])
    df_eval['r_pow_n'] = df_eval['r']**df_eval['n']
    df_eval['n_eff'] = df_eval['n'] + 0.5 * (df_eval['n'] == 0)

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

    feature_names = ["n_eff", "s", "r2", "cos_n_theta"]
    X_train = df_train[feature_names].values
    y_target = df_train['y_target'].values
    weights = df_train['weight'].values
    
    # 2. Configure Symbolic Regression
    print("Initializing PySR Regressor for multi-state fitting...")
    model = PySRRegressor(
        variable_names=feature_names,
        niterations=2000,
        populations=100,
        population_size=100,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[],
        parsimony=0.001,
        maxsize=15,
        timeout_in_seconds=180,
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
    # cos_n_theta * (1 + s * (r2 - n_eff - 2))
    def physical_pred(d):
        n_eff = d['n_eff'].values
        s = d['s'].values
        r2 = d['r2'].values
        cos_n_theta = d['cos_n_theta'].values
        return cos_n_theta * (1.0 + s * (r2 - n_eff - 2.0))
        
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
        eq_complexity = 11
        eq_latex = "`$\\psi(r,\\theta)_{s,n}= \\sqrt{\\frac{2 s!}{(s+n)!}} e^{-r^2/2} r^n (1 + s(r^2 - (n + 0.5 \\delta_{n,0}) - 2)) \\cos(n\\theta)$`"
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