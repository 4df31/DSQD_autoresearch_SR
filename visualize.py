import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import math

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
    Evaluates the best discovered symbolic regressed wavefunction.
    Handles both array and scalar inputs for s and n.
    """
    r2 = r**2
    n_eff = n + 0.5 * (n == 0)
    
    from scipy.special import genlaguerre, gamma, factorial
    
    if isinstance(s, np.ndarray):
        val = np.zeros_like(r)
        unique_states = np.unique(np.column_stack((s, n)), axis=0)
        for s_val, n_val in unique_states:
            mask = (s == s_val) & (n == n_val)
            if not np.any(mask):
                continue
            ne_val = n_val + 0.5 * (n_val == 0)
            poly = genlaguerre(int(s_val), ne_val)
            P_SR = poly(r2[mask])
            const_term = gamma(s_val + ne_val + 1.0) / (gamma(s_val + 1.0) * gamma(ne_val + 1.0))
            P_SR = P_SR / const_term
            
            coef = np.sqrt(2.0 * factorial(int(s_val)) / factorial(int(s_val + n_val)))
            val[mask] = np.exp(-0.5 * r2[mask]) * (r[mask]**n_val) * coef * P_SR * np.cos(n_val * theta[mask])
        return val
    else:
        poly = genlaguerre(int(s), n_eff)
        P_SR = poly(r2)
        const_term = gamma(s + n_eff + 1.0) / (gamma(s + 1.0) * gamma(n_eff + 1.0))
        P_SR = P_SR / const_term
        
        coef = np.sqrt(2.0 * factorial(int(s)) / factorial(int(s + n)))
        return np.exp(-0.5 * r2) * (r**n) * coef * P_SR * np.cos(n * theta)

def main():
    # 1. Load Data
    df_clean, _ = load_fem_data()
    # Load all states present in the dataset (up to s=3, n=7)
    df = df_clean.copy().reset_index(drop=True)
    
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
    for (s_val, n_val), score in sorted(r2_scores.items()):
        print(f"State s={s_val}, n={n_val}: R^2 = {score:.8f}")

    # 3. Calculate Eigenvalues (Analytical, FEM, and SR Expectation Values)
    eigenvalues = []
    unique_states = df[['s', 'n']].drop_duplicates().copy()
    unique_states['energy'] = 2 * unique_states['s'] + unique_states['n'] + 1
    unique_states = unique_states.sort_values(by=['energy', 's', 'n']).reset_index(drop=True)
    
    for idx, row in unique_states.iterrows():
        s_val = int(row['s'])
        n_val = int(row['n'])
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

    # 5. Create Interactive Plotly Visualizations (Subplots layout)
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "scatter", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        subplot_titles=(
            "Wavefunction R² Heatmap by Quantum Numbers (s, n)",
            "Eigenenergy Comparison (Analytical vs FEM vs SR)",
            "Autoresearch Progress: R² Score over Iterations",
            "Radial Wavefunction Profile (s=0, n=0)",
            "Radial Wavefunction Profile (s=1, n=1)",
            "Radial Wavefunction Profile (s=2, n=2)",
            "Radial Wavefunction Profile (s=3, n=1)"
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.15
    )

    # Subplot 1: Heatmap of wavefunction R2
    max_s = int(df['s'].max())
    max_n = int(df['n'].max())
    s_range = list(range(max_s + 1))
    n_range = list(range(max_n + 1))
    
    z_data = []
    text_data = []
    for s_val in s_range:
        z_row = []
        text_row = []
        for n_val in n_range:
            score = r2_scores.get((s_val, n_val), np.nan)
            z_row.append(score)
            if np.isnan(score):
                text_row.append("")
            else:
                text_row.append(f"{score:.4f}")
        z_data.append(z_row)
        text_data.append(text_row)
        
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=[f"n={n}" for n in n_range],
            y=[f"s={s}" for s in s_range],
            colorscale="Viridis",
            text=text_data,
            texttemplate="%{text}",
            colorbar=dict(title="R²", len=0.3, y=0.85, x=0.45),
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
        fig.add_trace(
            go.Scatter(
                x=[1],
                y=[0.971533],
                mode="lines+markers",
                name="Best R² Progress"
            ),
            row=2, col=1
        )

    # Helper function to get radial profile comparing FEM vs SR
    def plot_profile_subplot(s_val, n_val, r_col, r_row):
        state_data = df[(df['s'] == s_val) & (df['n'] == n_val)]
        theta_0_data = state_data[state_data['theta'] < 0.01].sort_values(by='r')
        
        if len(theta_0_data) == 0:
            first_theta = state_data['theta'].min()
            theta_0_data = state_data[state_data['theta'] == first_theta].sort_values(by='r')
            
        r_vals = theta_0_data['r'].values
        fem_vals = theta_0_data['psi_norm'].values
        
        sr_vals = eval_sr_wavefunction(r_vals, theta_0_data['theta'].values, float(n_val), float(s_val))
        dot = np.dot(fem_vals, sr_vals)
        sign = np.sign(dot) if dot != 0 else 1.0
        sr_vals = sr_vals * sign
        norm = np.linalg.norm(sr_vals)
        if norm > 0:
            sr_vals = sr_vals / norm
            
        fig.add_trace(
            go.Scatter(x=r_vals, y=fem_vals, mode="lines", name=f"FEM s={s_val},n={n_val}", line=dict(dash="solid", width=2)),
            row=r_row, col=r_col
        )
        fig.add_trace(
            go.Scatter(x=r_vals, y=sr_vals, mode="lines", name=f"SR s={s_val},n={n_val}", line=dict(dash="dash", width=2)),
            row=r_row, col=r_col
        )

    # Subplot 4, 5, 6 & 7: Profile lines
    plot_profile_subplot(s_val=0, n_val=0, r_col=1, r_row=3)
    plot_profile_subplot(s_val=1, n_val=1, r_col=2, r_row=3)
    plot_profile_subplot(s_val=2, n_val=2, r_col=1, r_row=4)
    plot_profile_subplot(s_val=3, n_val=1, r_col=2, r_row=4)

    # Update Layout
    fig.update_layout(
        title=dict(
            text="DSQD Symbolic Regression Autoresearch Verification",
            x=0.5,
            font=dict(size=22, color="#2c3e50")
        ),
        barmode="group",
        height=1600,
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
    fig.update_xaxes(title_text="Radius r", row=3, col=1)
    fig.update_yaxes(title_text="psi(r)", row=3, col=1)
    fig.update_xaxes(title_text="Radius r", row=3, col=2)
    fig.update_yaxes(title_text="psi(r)", row=3, col=2)
    fig.update_xaxes(title_text="Radius r", row=4, col=1)
    fig.update_yaxes(title_text="psi(r)", row=4, col=1)
    fig.update_xaxes(title_text="Radius r", row=4, col=2)
    fig.update_yaxes(title_text="psi(r)", row=4, col=2)

    # Save to HTML
    output_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize.html")
    fig.write_html(output_html)
    print(f"\nPlotly visualization successfully saved to: {output_html}")

if __name__ == "__main__":
    main()
