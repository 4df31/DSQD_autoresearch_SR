import numpy as np
from pysr import PySRRegressor
import os
import time
from prepare import generate_fem_data

def run_experiment():
    start_time = time.time()
    
    # 1. Load Data
    r, psi = generate_fem_data()
    X = r.reshape(-1, 1)
    y = psi
    
    # Normalization
    y_max = np.max(np.abs(y))
    y_norm = y / y_max
    
    # 2. Configure Symbolic Regression
    print("Initializing PySR Regressor...")
    model = PySRRegressor(
        variable_names=["r"],
        niterations=100,
        populations=20,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp", "square", "sqrt", "cos", "sin"],
        parsimony=0.005,
        maxsize=35,
        timeout_in_seconds=280,
        procs=os.cpu_count() or 4,
        verbosity=0,
        temp_equation_file=True,
    )
    
    # 3. Fit the Model
    search_start = time.time()
    model.fit(X, y_norm)
    search_time = time.time() - search_start
    
    # 4. Extract Results & Evaluate
    best_eq = model.get_best()
    predictions = model.predict(X)
    
    # Calculate R2
    # Ground Truth: psi \propto exp(-r^2 / 2)
    true_psi = np.exp(-0.5 * r**2)
    true_psi = true_psi / np.linalg.norm(true_psi) # Normalize
    
    # Normalize predictions for correct R2 comparison
    pred_norm = predictions / np.linalg.norm(predictions) if np.linalg.norm(predictions) > 0 else predictions
    
    ss_res = np.sum((true_psi - pred_norm)**2)
    ss_tot = np.sum((true_psi - np.mean(true_psi))**2)
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
