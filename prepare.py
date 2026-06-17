import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import torch

def generate_fem_data(grid_size=200, R_max=10.0, omega=1.0):
    """
    Generate numerical envelope wavefunctions and eigenvalues for the first 20 states.
    Strictly uses GPU computation to leverage high-performance cards (e.g., RTX 5090).
    Outputs both clean and gaussian-noise augmented data sets.
    """
    # Enforce GPU computation
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for execution. No GPU detected, aborting to avoid CPU fallback.")
        
    device = torch.device('cuda')
    print(f"-> Generating FEM data strictly on GPU device: {device}...")

    dr = R_max / grid_size
    r = np.linspace(dr, R_max, grid_size)
    
    diag = -2.0 / (dr**2)
    off_diag = 1.0 / (dr**2)
    laplacian = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(grid_size, grid_size))
    
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
        
    data_rows_clean = []
    data_rows_noisy = []
    theta_grid = np.linspace(0, 2 * np.pi, 50)
    
    for n_val in sorted(n_to_s.keys()):
        # Effective potential
        if n_val == 0:
            potential = 0.5 * omega**2 * r**2
        else:
            potential = 0.5 * omega**2 * r**2 + (n_val**2 - 0.25) / (2.0 * r**2)
            
        H_sparse = -0.5 * laplacian + sp.diags(potential)
        H_dense = H_sparse.toarray()
        
        # GPU computation for Eigenvalues/Eigenvectors
        H_tensor = torch.tensor(H_dense, dtype=torch.float64, device=device)
        vals_tensor, vecs_tensor = torch.linalg.eigh(H_tensor)
        vals = vals_tensor.cpu().numpy()
        vecs = vecs_tensor.cpu().numpy()
            
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
            
            # --- Noise Calculations ---
            # Max amplitude limit (10% of peak amplitude)
            max_amp = np.max(np.abs(R))
            psi_noise_limit = 0.10 * max_amp 
            
            # Create a proportional "altered" eigenvalue (max 10% variance)
            eigen_noise_limit = 0.10 * np.abs(eigenvalue)
            # Use 3-sigma distribution logic to keep the bulk of noise smooth but strictly clip at the maximum threshold
            eigen_noise = np.clip(np.random.normal(0, eigen_noise_limit / 3.0), -eigen_noise_limit, eigen_noise_limit)
            eigenvalue_noisy = eigenvalue + eigen_noise
            
            # Populate grids for clean and noisy streams
            for i, r_val in enumerate(r):
                R_val = R[i]
                for theta_val in theta_grid:
                    psi_val = R_val * np.cos(n_val * theta_val)
                    
                    # Generate random bounded Gaussian noise for the wavefunction coordinate
                    psi_noise = np.clip(np.random.normal(0, psi_noise_limit / 3.0), -psi_noise_limit, psi_noise_limit)
                    psi_noisy_val = psi_val + psi_noise
                    
                    # Clean data row
                    data_rows_clean.append({
                        "r": r_val,
                        "theta": theta_val,
                        "n": float(n_val),
                        "s": float(s_val),
                        "psi": psi_val,
                        "eigenvalue": eigenvalue
                    })
                    
                    # Noisy data row
                    data_rows_noisy.append({
                        "r": r_val,
                        "theta": theta_val,
                        "n": float(n_val),
                        "s": float(s_val),
                        "psi": psi_noisy_val,
                        "eigenvalue": eigenvalue_noisy
                    })
                    
    df_clean = pd.DataFrame(data_rows_clean)
    df_noisy = pd.DataFrame(data_rows_noisy)
    return df_clean, df_noisy

def save_fem_data(df_clean, df_noisy):
    """Save both the pristine and noisy FEM dataframes to the repository."""
    base_dir = os.path.dirname(__file__)
    clean_path = os.path.join(base_dir, "fem_dsqd_data.csv")
    noisy_path = os.path.join(base_dir, "fem_dsqd_data_noisy.csv")
    
    df_clean.to_csv(clean_path, index=False)
    df_noisy.to_csv(noisy_path, index=False)
    
    print(f"-> Saved NOISELESS FEM data to repo: {clean_path}")
    print(f"-> Saved NOISY FEM data to repo: {noisy_path}")

def load_fem_data():
    """Load pre-generated FEM data from repo, or generate both if missing."""
    base_dir = os.path.dirname(__file__)
    clean_path = os.path.join(base_dir, "fem_dsqd_data.csv")
    noisy_path = os.path.join(base_dir, "fem_dsqd_data_noisy.csv")
    
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        print("-> FEM data files not found in repo. Generating fresh dataset pairs via GPU...")
        df_clean, df_noisy = generate_fem_data()
        save_fem_data(df_clean, df_noisy)
    else:
        print(f"-> Loading FEM datasets (clean & noisy) from repo...")
        df_clean = pd.read_csv(clean_path)
        df_noisy = pd.read_csv(noisy_path)
        
    return df_clean, df_noisy

def evaluate_r2(predictions, y_true):
    """Immutable evaluation harness."""
    ss_res = np.sum((y_true - predictions) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot)
