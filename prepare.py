import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pysr import PySRRegressor
import pandas as pd
import subprocess
import os

def generate_fem_data(grid_size=200, R_max=10.0, omega=1.0):
    """Step 1: Generate synthetic FEM data for 2D IHO (n=0, l=0) using GPU acceleration if available"""
    print("-> Solving FEM 2D IHO...")
    dr = R_max / grid_size
    r = np.linspace(dr, R_max, grid_size)
    
    diag = -2.0 / (dr**2)
    off_diag = 1.0 / (dr**2)
    laplacian = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(grid_size, grid_size))
    potential = 0.5 * omega**2 * r**2
    
    H_sparse = -0.5 * laplacian + sp.diags(potential)
    H_dense = H_sparse.toarray()
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Solving on GPU/CPU via PyTorch (device: {device})...")
        H_tensor = torch.tensor(H_dense, dtype=torch.float64, device=device)
        vals, vecs = torch.linalg.eigh(H_tensor)
        psi = vecs[:, 0].cpu().numpy()
    except Exception as e:
        print(f"   PyTorch/GPU failed or not available ({e}). Falling back to SciPy CPU...")
        vals, vecs = eigsh(H_sparse, k=1, which='SM')
        psi = vecs[:, 0]
        
    # Normalize the FEM wavefunction
    psi = psi / np.linalg.norm(psi)
    return r, psi

def save_fem_data(r, psi):
    """Save generated FEM data to repository folder."""
    csv_path = os.path.join(os.path.dirname(__file__), "fem_dsqd_data.csv")
    df = pd.DataFrame({"r": r, "psi": psi})
    df.to_csv(csv_path, index=False)
    print(f"-> Saved FEM data to repo: {csv_path}")

def load_fem_data():
    """Load pre-generated FEM data from repo, or generate it if missing."""
    csv_path = os.path.join(os.path.dirname(__file__), "fem_dsqd_data.csv")
    if not os.path.exists(csv_path):
        print("-> FEM data file not found in repo. Generating fresh data...")
        r, psi = generate_fem_data()
        save_fem_data(r, psi)
    else:
        print(f"-> Loading FEM data from repo: {csv_path}")
    df = pd.read_csv(csv_path)
    return df['r'].values, df['psi'].values

def run_symbolic_regression(r, psi):
    """Step 2: Run PySR to find the analytical equation"""
    print("-> Running Symbolic Regression (PySR)...")
    X = r.reshape(-1, 1)
    y = psi
    
    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-"],
        unary_operators=["exp", "square"],
        extra_sympy_mappings={"square": lambda x: x**2},
        verbosity=0 
    )
    model.fit(X, y)
    best_eq = model.get_best()
    
    print(f"\nBest Equation Found: {best_eq.equation}")
    return model

def evaluate_and_commit(r, psi, model):
    """Step 3: Evaluate against ground truth and commit if successful"""
    print("-> Evaluating Mathematical Accuracy...")
    
    # Predict using the symbolic model
    pred_psi = model.predict(r.reshape(-1, 1))
    
    # Ground Truth: psi \propto exp(-r^2 / 2)
    true_psi = np.exp(-0.5 * r**2)
    true_psi = true_psi / np.linalg.norm(true_psi) # Normalize
    
    # Calculate R^2
    ss_res = np.sum((true_psi - pred_psi)**2)
    ss_tot = np.sum((true_psi - np.mean(true_psi))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Evaluation R^2 Score: {r2:.6f}")
    
    if r2 > 0.999:
        print("Success! Threshold met. Committing to Git...")
        with open("discovered_model.txt", "a") as f:
            f.write(f"Equation: {model.sympy()}\nR^2: {r2}\n\n")
            
        subprocess.run(["git", "add", "discovered_model.txt", "train.py"])
        subprocess.run(["git", "commit", "-m", f"Automated Discovery: R^2={r2:.4f}"])
    else:
        print("Threshold not met. Further optimization required.")

if __name__ == "__main__":
    # Load and cache
    r, psi = load_fem_data()
    model = run_symbolic_regression(r, psi)
    evaluate_and_commit(r, psi, model)
