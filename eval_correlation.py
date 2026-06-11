import numpy as np
import math
from scipy.special import genlaguerre

# Radial grid setup (matching prepare.py)
grid_size = 200
R_max = 10.0
dr = R_max / grid_size
r_grid = np.linspace(dr, R_max, grid_size)

def quinteiro_profile(r, s, n):
    poly = genlaguerre(int(s), int(n))
    L_val = poly(r**2)
    coef = math.sqrt(2.0 * math.factorial(int(s)) / math.factorial(int(s + n)))
    sign = (-1.0)**int(s)
    return sign * coef * np.exp(-0.5 * r**2) * (r**n) * L_val

def general_expression_profile(r, s, n):
    # Unified general expression: R_sn(r) = exp(-0.5 * r^2) * r^n * L_s^{alpha}(r^2)
    # where alpha = n if n >= 1 else 0.5
    alpha = float(n) if n >= 1 else 0.5
    poly = genlaguerre(int(s), alpha)
    L_val = poly(r**2)
    # We use a matching coefficient
    coef = math.sqrt(2.0 * math.factorial(int(s)) / math.gamma(s + alpha + 1))
    sign = (-1.0)**int(s)
    return sign * coef * np.exp(-0.5 * r**2) * (r**n) * L_val

def compute_correlation(y_true, y_pred):
    # Align sign
    dot = np.dot(y_true, y_pred)
    if dot < 0:
        y_pred = -y_pred
        
    # Normalize
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)
    if norm_true > 0:
        y_true = y_true / norm_true
    if norm_pred > 0:
        y_pred = y_pred / norm_pred
        
    # Pearson correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # R2 score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    return corr, r2

def main():
    s_values = [3, 4, 5, 6]
    n_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    
    print("Evaluating correlation between Quinteiro's Model and Discovered General Expression:")
    print(f"{'s':<3} | {'n':<3} | {'Corr (Pearson)':<14} | {'R2 Score':<12}")
    print("-" * 45)
    
    for s in s_values:
        for n in n_values:
            y_true = quinteiro_profile(r_grid, s, n)
            y_pred = general_expression_profile(r_grid, s, n)
            corr, r2 = compute_correlation(y_true, y_pred)
            print(f"{s:<3} | {n:<3} | {corr:14.8f} | {r2:12.8f}")

if __name__ == "__main__":
    main()
