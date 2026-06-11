import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Append current directory to path to load prepare.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare import load_fem_data

# Physical constants and grid setup (must match prepare.py)
grid_size = 200
R_max = 10.0
omega = 1.0
dr = R_max / grid_size
r_grid = np.linspace(dr, R_max, grid_size)

# Laplacian
diag = -2.0 / (dr**2)
off_diag = 1.0 / (dr**2)
laplacian = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(grid_size, grid_size))

def get_hamiltonian(n_val):
    """Construct the numerical Hamiltonian matrix for a given angular momentum n."""
    if n_val == 0:
        potential = 0.5 * omega**2 * r_grid**2
    else:
        potential = 0.5 * omega**2 * r_grid**2 + (n_val**2 - 0.25) / (2.0 * r_grid**2)
    H_sparse = -0.5 * laplacian + sp.diags(potential)
    return H_sparse.toarray()

def eval_sr_wavefunction(r, theta, n, s):
    """
    Evaluates the best discovered symbolic regressed wavefunction:
    ((x3 - (x1 * (((((x0 * 0.33383948) + 0.6668352) * x2) - x0) * x3))) * x4) * ((x5 * 0.99813247) - -0.0016926366)
    where:
    x0 = n, x1 = s, x2 = r2, x3 = exp_half_r2, x4 = cos_n_theta, x5 = r_pow_n
    """
    r2 = r**2
    x0 = n
    x1 = s
    x2 = r2
    x3 = np.exp(-0.5 * r2)
    x4 = np.cos(n * theta)
    x5 = r**n
    
    val = ((x3 - (x1 * (((((x0 * 0.33383948) + 0.6668352) * x2) - x0) * x3))) * x4) * ((x5 * 0.99813247) - -0.0016926366)
    return val

def main():
    # 1. Load Data
    df = load_fem_data()
    df = df[(df['s'] <= 1.0) & (df['n'] <= 1.0)].copy().reset_index(drop=True)
    
    # State-by-state ground truth normalization
    df['psi_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals

    # 2. Evaluate SR predictions
    df['pred_sr'] = eval_sr_wavefunction(df['r'].values, df['theta'].values, df['n'].values, df['s'].values)
    
    # State-by-state prediction normalization and sign alignment
    df['pred_norm'] = 0.0
    r2_scores = {}
    for (n_val, s_val), group in df.groupby(['n', 's']):
        pred_vals = group['pred_sr'].values
        target_vals = group['psi_norm'].values
        
        # Align sign
        dot = np.dot(target_vals, pred_vals)
        sign = np.sign(dot) if dot != 0 else 1.0
        pred_vals = pred_vals * sign
        
        norm = np.linalg.norm(pred_vals)
        if norm > 0:
            pred_vals = pred_vals / norm
        df.loc[group.index, 'pred_norm'] = pred_vals
        
        # Compute R2 for this specific state
        ss_res = np.sum((target_vals - pred_vals)**2)
        ss_tot = np.sum((target_vals - np.mean(target_vals))**2)
        r2_scores[(int(s_val), int(n_val))] = 1.0 - (ss_res / ss_tot)

    print("=== Wavefunction R^2 Scores by State (s, n) ===")
    for (s_val, n_val), score in r2_scores.items():
        print(f"State s={s_val}, n={n_val}: R^2 = {score:.8f}")

    # 3. Calculate Eigenvalues (Analytical, FEM, and SR Expectation Values)
    eigenvalues = []
    for (n_val, s_val) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # Analytical eigenvalue
        E_analytical = 2.0 * s_val + n_val + 1.0
        
        # FEM eigenvalue (read from dataset)
        state_data = df[(df['s'] == s_val) & (df['n'] == n_val)]
        E_fem = state_data['eigenvalue'].iloc[0]
        
        # SR expectation value: E_sr = u^T H u
        # Extract radial part at theta = 0
        R_sr = eval_sr_wavefunction(r_grid, 0.0, float(n_val), float(s_val))
        
        # Convert to Hamiltonian vector u
        if n_val == 0:
            u_sr = R_sr * r_grid
        else:
            u_sr = R_sr * np.sqrt(r_grid)
            
        # Normalize u
        u_norm = np.linalg.norm(u_sr)
        if u_norm > 0:
            u_sr = u_sr / u_norm
            
        H = get_hamiltonian(n_val)
        E_sr = u_sr.T @ H @ u_sr
        
        eigenvalues.append({
            "s": s_val,
            "n": n_val,
            "Analytical": E_analytical,
            "FEM": E_fem,
            "SR": E_sr
        })
        
    df_ev = pd.DataFrame(eigenvalues)
    print("\n=== Eigenvalue Comparison ===")
    print(df_ev.to_string(index=False))
    
    # Calculate R^2 of the eigenvalues
    y_true_ev = df_ev['FEM'].values
    y_pred_ev = df_ev['SR'].values
    ss_res_ev = np.sum((y_true_ev - y_pred_ev)**2)
    ss_tot_ev = np.sum((y_true_ev - np.mean(y_true_ev))**2)
    r2_ev = 1.0 - (ss_res_ev / ss_tot_ev)
    print(f"\nEigenvalue R^2 Correlation (FEM vs SR): {r2_ev:.8f}")
    
    # 4. Load Results History
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")
    has_history = False
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file, sep='\t')
        df_hist = df_hist[df_hist['status'] == 'keep'].reset_index(drop=True)
        if len(df_hist) > 0:
            has_history = True

    # 5. Create Interactive Plotly Visualizations
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "heatmap"}, {"type": "bar"}],
               [{"type": "scatter", "colspan": 2}, None]],
        subplot_titles=(
            "Wavefunction R² Heatmap by Quantum Numbers (s, n)",
            "Eigenenergy Comparison (Analytical vs FEM vs SR)",
            "Autoresearch Progress: R² Score over Iterations"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # Subplot 1: Heatmap of wavefunction R2
    z_data = [[r2_scores[(0, 0)], r2_scores[(0, 1)]],
              [r2_scores[(1, 0)], r2_scores[(1, 1)]]]
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=["n=0", "n=1"],
            y=["s=0", "s=1"],
            colorscale="Viridis",
            text=[[f"{val:.6f}" for val in row] for row in z_data],
            texttemplate="%{text}",
            colorbar=dict(title="R²", len=0.4, y=0.75, x=0.45),
            showscale=True
        ),
        row=1, col=1
    )

    # Subplot 2: Bar chart for eigenvalues
    fig.add_trace(
        go.Bar(name="Analytical", x=[f"s={s},n={n}" for s, n in zip(df_ev['s'], df_ev['n'])], y=df_ev['Analytical'], marker_color="#1f77b4"),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name="FEM", x=[f"s={s},n={n}" for s, n in zip(df_ev['s'], df_ev['n'])], y=df_ev['FEM'], marker_color="#ff7f0e"),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name="SR Expectation", x=[f"s={s},n={n}" for s, n in zip(df_ev['s'], df_ev['n'])], y=df_ev['SR'], marker_color="#2ca02c"),
        row=1, col=2
    )

    # Subplot 3: Progress of R2 score over iterations
    if has_history:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df_hist) + 1)),
                y=df_hist['best_r2_score'],
                mode="lines+markers",
                line=dict(color="#d62728", width=3),
                marker=dict(size=8),
                hovertext=df_hist['description'],
                name="Best R² Progress"
            ),
            row=2, col=1
        )
    else:
        # Fallback if results.tsv has only 1 row or not found
        fig.add_trace(
            go.Scatter(
                x=[1],
                y=[0.911102],
                mode="lines+markers",
                name="Best R² Progress (No history)"
            ),
            row=2, col=1
        )

    # Update Layout
    fig.update_layout(
        title=dict(
            text="DSQD Symbolic Regression Analysis & Visualization",
            x=0.5,
            font=dict(size=20, color="#2c3e50")
        ),
        barmode="group",
        height=800,
        width=1000,
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Add axis titles
    fig.update_xaxes(title_text="Angular Momentum (n)", row=1, col=1)
    fig.update_yaxes(title_text="Radial State (s)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (E)", row=1, col=2)
    fig.update_xaxes(title_text="Iteration (Keep commits)", row=2, col=1)
    fig.update_yaxes(title_text="Overall R² Score", row=2, col=1)

    # Save to HTML
    output_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize.html")
    fig.write_html(output_html)
    print(f"\nPlotly visualization successfully saved to: {output_html}")

if __name__ == "__main__":
    main()
