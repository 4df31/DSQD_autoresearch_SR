import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os
import time

# --- STRICT ADHERENCE TO program.md ---
# We MUST use the read-only prepare.py for data loading and final evaluation.
try:
        from prepare import load_fem_data, evaluate_r2
except ImportError:
        # Fallback/Mock if prepare.py is not yet in the environment during testing
            print("WARNING: prepare.py not found. Using internal fallback metrics.")
                def load_fem_data():
                            return pd.read_csv(os.path.expanduser("~/.cache/autoresearch/fem_dsqd_data.csv"))
                            def evaluate_r2(predictions, y_true):
                                        ss_res = np.sum((y_true - predictions) ** 2)
                                                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                                                        return 1.0 - (ss_res / ss_tot)

                                                    def run_experiment():
                                                            start_time = time.time()
                                                                
                                                                    # 1. Load Data (Using all quantum numbers to find the general equation)
                                                                        df = load_fem_data()
                                                                            
                                                                                # Features: r (radial), theta (azimuthal), n (OAM quantum number), s (radial quantum number)
                                                                                    # Exposing all these allows PySR to discover the generalized Laguerre polynomial forms
                                                                                        feature_cols = ['r', 'theta', 'n', 's']
                                                                                            X = df[feature_cols].values
                                                                                                
                                                                                                    # Target: The real part or magnitude of the envelope wavefunction
                                                                                                        # (Agent can modify this to fit psi_real, psi_imag, or psi_mag depending on strategy)
                                                                                                            y = df['psi_mag'].values
                                                                                                                
                                                                                                                    # Normalization: crucial for symbolic regression to find consistent coefficients
                                                                                                                        y_max = np.max(np.abs(y))
                                                                                                                            y_norm = y / y_max

                                                                                                                                # 2. Configure Symbolic Regression (The Agent will tweak these parameters)
                                                                                                                                    print("Initializing PySR Regressor...")
                                                                                                                                        model = PySRRegressor(
                                                                                                                                                        # Naming the variables so the output equation reads like the physics paper
                                                                                                                                                                variable_names=feature_cols,
                                                                                                                                                                        
                                                                                                                                                                        # Agent: Tune evolutionary hyperparameters
                                                                                                                                                                                niterations=100,      # High iterations, but constrained by timeout
                                                                                                                                                                                        populations=20,
                                                                                                                                                                                                
                                                                                                                                                                                                # Agent: Operators tailored for Quinteiro's 2DIHO wavefunctions
                                                                                                                                                                                                        # Includes trig functions for the e^{-in*theta} azimuthal component
                                                                                                                                                                                                                binary_operators=["+", "*", "-", "/"],
                                                                                                                                                                                                                        unary_operators=["exp", "square", "sqrt", "cos", "sin"],
                                                                                                                                                                                                                                
                                                                                                                                                                                                                                # Agent: Tweak parsimony penalty to favor simpler physical equations
                                                                                                                                                                                                                                        parsimony=0.005,
                                                                                                                                                                                                                                                maxsize=35,           # Increased to accommodate Laguerre polynomials + exponential
                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                        # Enforce the 5-minute (300s) budget rule from program.md
                                                                                                                                                                                                                                                                timeout_in_seconds=280,
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                        # Fast execution using all CPU cores
                                                                                                                                                                                                                                                                                procs=os.cpu_count() or 4,
                                                                                                                                                                                                                                                                                        verbosity=0,
                                                                                                                                                                                                                                                                                                temp_equation_file=True,
                                                                                                                                                                                                                                                                                                    )
                                                                                                                                            
                                                                                                                                            # 3. Fit the Model
                                                                                                                                                search_start = time.time()
                                                                                                                                                    model.fit(X, y_norm)
                                                                                                                                                        search_time = time.time() - search_start
                                                                                                                                                            
                                                                                                                                                                # 4. Extract Results & Evaluate using immutable prepare.py harness
                                                                                                                                                                    best_eq = model.get_best()
                                                                                                                                                                        predictions = model.predict(X)
                                                                                                                                                                            
                                                                                                                                                                                # Ground truth evaluation against the test harness
                                                                                                                                                                                    r2_score = evaluate_r2(predictions, y_norm)
                                                                                                                                                                                        
                                                                                                                                                                                            total_time = time.time() - start_time
                                                                                                                                                                                                
                                                                                                                                                                                                    # 5. Output format strictly matching program.md requirements
                                                                                                                                                                                                        print("\n---")
                                                                                                                                                                                                            print(f"best_r2_score:    {r2_score:.6f}")
                                                                                                                                                                                                                print(f"complexity:       {best_eq.complexity}")
                                                                                                                                                                                                                    print(f"search_seconds:   {search_time:.1f}")
                                                                                                                                                                                                                        print(f"total_seconds:    {total_time:.1f}")
                                                                                                                                                                                                                            
                                                                                                                                                                                                                                # GPU VRAM tracking for the 5090
                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                import torch
                                                                                                                                                                                                                                                        # Forces PyTorch to initialize CUDA to check memory if FEM was loaded on GPU
                                                                                                                                                                                                                                                                if torch.cuda.is_available():
                                                                                                                                                                                                                                                                                vram = torch.cuda.max_memory_allocated() / (1024**2)
                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                        vram = 0.0
                                                                                                                                                                                                                                                                                                            except ImportError:
                                                                                                                                                                                                                                                                                                                        vram = 0.0
                                                                                                                                                                                                                                                                                                                            print(f"peak_vram_mb:     {vram:.1f}")
                                                                                                                                                                                                                                                                                                                                print(f"best_equation:    \"{best_eq.equation}\"")

                                                                                                                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                        # Ensure warnings don't clutter the stdout parsing for the agent
                                                                                                                                                                                                                                                                                                                                            import warnings
                                                                                                                                                                                                                                                                                                                                                warnings.filterwarnings("ignore")
                                                                                                                                                                                                                                                                                                                                                    run_experiment()
