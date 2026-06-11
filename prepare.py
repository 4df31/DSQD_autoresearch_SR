import numpy as np
import scipy.sparse as sp
import pandas as pd
import os

def generate_fem_data(grid_size=200, R_max=10.0, omega=1.0):
    """Generate numerical envelope wavefunctions and eigenvalues for the first 20 states on GPU/CPU"""
    dr = R_max / grid_size
    r = np.linspace(dr, R_max, grid_size)
    
    diag = -2.0 / (dr**2)
    off_diag = 1.0 / (dr**2)
    laplacian = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(grid_size, grid_size))
    
    try:
        import cupy as cp
        # Verify that we can access a GPU device
        cp.array([1.0])
        device = 'cuda'
        print("-> Generating FEM data on GPU using CuPy...")
    except Exception:
        device = 'cpu'
        print("-> CuPy or GPU not found. Generating FEM data on CPU using SciPy...")
        
    # First 20 states ordered by energy E = 2*s + n + 1 (for s >= 0, n >= 0)
    states = []
    for s_val in range(10):
        for n_val in range(10):
            energy = 2 * s_val + n_val + 1
            states.append((energy, s_val, n_val))
    states.sort()
    selected_states = states[:20]  # List of (energy, s, n)
    
    # Group states by n to avoid solving the Hamiltonian multiple times for the same n
    from collections import defaultdict
    n_to_s = defaultdict(list)
    for _, s_val, n_val in selected_states:
        n_to_s[n_val].append(s_val)
        
    data_rows = []
    theta_grid = np.linspace(0, 2 * np.pi, 50)
    
    for n_val in sorted(n_to_s.keys()):
        # Effective potential (centrifugal term is omitted for n=0 to avoid singularity)
        if n_val == 0:
            potential = 0.5 * omega**2 * r**2
        else:
            potential = 0.5 * omega**2 * r**2 + (n_val**2 - 0.25) / (2.0 * r**2)
            
        H_sparse = -0.5 * laplacian + sp.diags(potential)
        H_dense = H_sparse.toarray()
        
        if device == 'cuda':
            H_gpu = cp.array(H_dense, dtype=cp.float64)
            vals_gpu, vecs_gpu = cp.linalg.eigh(H_gpu)
            vals = vals_gpu.get()
            vecs = vecs_gpu.get()
        else:
            vals, vecs = sp.linalg.eigh(H_dense)
            
        for s_val in n_to_s[n_val]:
            u = vecs[:, s_val]
            # Convert to envelope wavefunction R(r)
            if n_val == 0:
                R = u / r
            else:
                R = u / np.linalg.norm(u) / np.sqrt(r)
                
            # Normalize R
            R = R / np.linalg.norm(R)
            
            # Align sign to start positive near the origin
            if np.sum(R[:10]) < 0:
                R = -R
                
            eigenvalue = vals[s_val]
            
            # Populate grid
            for i, r_val in enumerate(r):
                R_val = R[i]
                for theta_val in theta_grid:
                    psi_val = R_val * np.cos(n_val * theta_val)
                    data_rows.append({
                        "r": r_val,
                        "theta": theta_val,
                        "n": float(n_val),
                        "s": float(s_val),
                        "psi": psi_val,
                        "eigenvalue": eigenvalue
                    })
                    
    df = pd.DataFrame(data_rows)
    return df

def save_fem_data(df):
    """Save generated FEM dataframe to repository folder."""
    csv_path = os.path.join(os.path.dirname(__file__), "fem_dsqd_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"-> Saved FEM data to repo: {csv_path}")

def load_fem_data():
    """Load pre-generated FEM data from repo, or generate it if missing."""
    csv_path = os.path.join(os.path.dirname(__file__), "fem_dsqd_data.csv")
    if not os.path.exists(csv_path):
        print("-> FEM data file not found in repo. Generating fresh data...")
        df = generate_fem_data()
        save_fem_data(df)
    else:
        print(f"-> Loading FEM data from repo: {csv_path}")
        df = pd.read_csv(csv_path)
    return df

def evaluate_r2(predictions, y_true):
    """Immutable evaluation harness."""
    ss_res = np.sum((y_true - predictions) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot)

