import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # 1. Load data
    df = pd.read_csv('fem_dsqd_data.csv')
    
    # Ground truth normalization
    df['psi_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        psi_vals = group['psi'].values
        norm = np.linalg.norm(psi_vals)
        df.loc[group.index, 'psi_norm'] = psi_vals / norm if norm > 0 else psi_vals
        
    # Analytical/SR wavefunction formula
    df['r2'] = df['r']**2
    df['exp_half_r2'] = np.exp(-0.5 * df['r2'])
    df['cos_n_theta'] = np.cos(df['n'] * df['theta'])
    df['r_pow_n'] = df['r']**df['n']
    df['c'] = df['n'] + 1.0 + 0.5 * (df['n'] == 0.0)
    
    # Compute features
    df['t1'] = df['s'] / df['c'] * df['r2']
    df['t2'] = df['s'] * (df['s'] - 1.0) / (2.0 * df['c'] * (df['c'] + 1.0)) * df['r2']**2
    df['t3'] = df['s'] * (df['s'] - 1.0) * (df['s'] - 2.0) / (6.0 * df['c'] * (df['c'] + 1.0) * (df['c'] + 2.0)) * df['r2']**3
    
    # Polynomial component P = 1 - t1 + t2 - t3
    df['p_pred'] = 1.0 - df['t1'] + df['t2'] - df['t3']
    predictions = df['p_pred'] * df['exp_half_r2'] * df['r_pow_n'] * df['cos_n_theta']
    
    # State-by-state sign alignment and normalization
    df['pred_norm'] = 0.0
    for keys, group in df.groupby(['n', 's']):
        pred_vals = predictions[group.index].values
        target_vals = group['psi_norm'].values
        dot = np.dot(target_vals, pred_vals)
        sign = np.sign(dot) if dot != 0 else 1.0
        pred_vals = pred_vals * sign
        norm = np.linalg.norm(pred_vals)
        if norm > 0:
            pred_vals = pred_vals / norm
        df.loc[group.index, 'pred_norm'] = pred_vals
        
    # Compute state-by-state wavefunction R2 scores
    r2_matrix = {}
    states = sorted(list(df.groupby(['n', 's']).groups.keys()))
    for n_val, s_val in states:
        group = df[(df['n'] == n_val) & (df['s'] == s_val)]
        target_vals = group['psi_norm'].values
        pred_vals = group['pred_norm'].values
        ss_res = np.sum((target_vals - pred_vals)**2)
        ss_tot = np.sum((target_vals - np.mean(target_vals))**2)
        r2_val = 1.0 - (ss_res / ss_tot)
        if n_val not in r2_matrix:
            r2_matrix[n_val] = {}
        r2_matrix[n_val][s_val] = r2_val
        
    # Convert to DataFrame for heatmap
    n_unique = sorted(df['n'].unique())
    s_unique = sorted(df['s'].unique())
    heatmap_data = np.zeros((len(s_unique), len(n_unique)))
    for i, s_val in enumerate(s_unique):
        for j, n_val in enumerate(n_unique):
            if n_val in r2_matrix and s_val in r2_matrix[n_val]:
                heatmap_data[i, j] = r2_matrix[n_val][s_val]
            else:
                heatmap_data[i, j] = np.nan
                
    # Compute expectation values of Hamiltonian numerically
    fem_energies = []
    analytical_energies = []
    sr_energies = []
    state_labels = []
    
    for n_val, s_val in states:
        state_df = df[(df['n'] == n_val) & (df['s'] == s_val)].copy()
        state_df = state_df.sort_values(by=['r', 'theta']).reset_index(drop=True)
        r_uniq = state_df['r'].unique()
        theta_uniq = state_df['theta'].unique()
        Nr, Ntheta = len(r_uniq), len(theta_uniq)
        
        r_grid = state_df['r'].values.reshape(Nr, Ntheta).copy()
        psi_grid = state_df['pred_norm'].values.reshape(Nr, Ntheta).copy()
        
        dr = r_uniq[1] - r_uniq[0]
        dtheta = theta_uniq[1] - theta_uniq[0]
        
        # Normalize psi on polar grid
        dA = r_grid * dr * dtheta
        norm_const = np.sqrt(np.sum(psi_grid**2 * dA))
        psi_grid /= norm_const
        
        # Derivatives
        dpsi_dr = np.gradient(psi_grid, dr, axis=0)
        d2psi_dr2 = np.gradient(dpsi_dr, dr, axis=0)
        dpsi_dtheta = np.gradient(psi_grid, dtheta, axis=1)
        d2psi_dtheta2 = np.gradient(dpsi_dtheta, dtheta, axis=1)
        
        # Laplacian (with effective centrifugal term for n=0)
        if n_val == 0.0:
            laplacian = d2psi_dr2 + (1.0 / r_grid) * dpsi_dr + (1.0 / r_grid**2) * d2psi_dtheta2 - (0.25 / r_grid**2) * psi_grid
        else:
            laplacian = d2psi_dr2 + (1.0 / r_grid) * dpsi_dr + (1.0 / r_grid**2) * d2psi_dtheta2
            
        H_psi = -0.5 * laplacian + 0.5 * r_grid**2 * psi_grid
        E_expect = np.sum(psi_grid * H_psi * dA)
        
        fem_val = state_df['eigenvalue'].iloc[0]
        E_analytical = 2.0 * s_val + n_val + 1.0 + 0.5 * (n_val == 0.0)
        
        fem_energies.append(fem_val)
        analytical_energies.append(E_analytical)
        sr_energies.append(E_expect)
        state_labels.append(f"s={int(s_val)}, n={int(n_val)}")
        
    # 2D Heatmap of Wavefunction R2 scores
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"n={int(n)}" for n in n_unique],
        y=[f"s={int(s)}" for s in s_unique],
        colorscale='Viridis',
        zmin=0.9, zmax=1.0,
        text=np.round(heatmap_data, 6),
        texttemplate="%{text}",
        hoverinfo='z'
    ))
    fig_heatmap.update_layout(
        title='Wavefunction R² Heatmap depending on s and n',
        xaxis_title='Angular Momentum (n)',
        yaxis_title='Radial Quantum Number (s)',
        width=700, height=500
    )
    fig_heatmap.write_html('wavefunction_r2_heatmap.html')
    print("Saved wavefunction_r2_heatmap.html")
    
    # Comparative group bar chart for eigenenergies
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Bar(
        x=state_labels, y=analytical_energies,
        name='Analytical', marker_color='rgb(55, 83, 109)'
    ))
    fig_energy.add_trace(go.Bar(
        x=state_labels, y=fem_energies,
        name='FEM (Numerical)', marker_color='rgb(26, 118, 141)'
    ))
    fig_energy.add_trace(go.Bar(
        x=state_labels, y=sr_energies,
        name='SR (Expectation Value)', marker_color='rgb(235, 120, 50)'
    ))
    fig_energy.update_layout(
        title='Comparison of Eigenenergies: Analytical vs FEM vs SR',
        xaxis_tickangle=-45,
        xaxis_title='States (s, n)',
        yaxis_title='Energy',
        bargroupgap=0.15,
        width=1000, height=600
    )
    fig_energy.write_html('eigenenergy_comparison.html')
    print("Saved eigenenergy_comparison.html")
    
    # Progress line chart of R2
    try:
        results_df = pd.read_csv('results.tsv', sep='\t')
        fig_progress = go.Figure()
        fig_progress.add_trace(go.Scatter(
            x=list(range(len(results_df))),
            y=results_df['best_r2_score'],
            mode='lines+markers',
            name='R² Score',
            line=dict(color='firebrick', width=3)
        ))
        fig_progress.update_layout(
            title='Progress of R² Score across Iterations',
            xaxis_title='Iteration',
            yaxis_title='Best R² Score',
            width=700, height=450
        )
        fig_progress.write_html('r2_progress.html')
        print("Saved r2_progress.html")
    except Exception as e:
        print("Failed to plot progress:", e)
        
    # Subplots contrasting radial wavefunctions
    fig_radial = make_subplots(
        rows=4, cols=5,
        subplot_titles=[f"s={int(s_val)}, n={int(n_val)}" for n_val, s_val in states],
        horizontal_spacing=0.04, vertical_spacing=0.08
    )
    
    for idx, (n_val, s_val) in enumerate(states):
        row = (idx // 5) + 1
        col = (idx % 5) + 1
        
        state_df = df[(df['n'] == n_val) & (df['s'] == s_val) & (df['theta'] == 0.0)].copy()
        state_df = state_df.sort_values(by='r').reset_index(drop=True)
        
        r = state_df['r'].values
        target = state_df['psi_norm'].values
        pred = state_df['pred_norm'].values
        
        fig_radial.add_trace(
            go.Scatter(x=r, y=target, mode='lines', name='Numerical (FEM)', line=dict(color='blue'), showlegend=(idx==0)),
            row=row, col=col
        )
        fig_radial.add_trace(
            go.Scatter(x=r, y=pred, mode='lines', name='SR Wavefunction', line=dict(color='orange', dash='dash'), showlegend=(idx==0)),
            row=row, col=col
        )
        
    fig_radial.update_layout(
        title='Radial Wavefunctions Comparison (at theta=0): Numerical vs SR',
        width=1200, height=800,
        showlegend=True
    )
    fig_radial.write_html('radial_wavefunction_comparison.html')
    print("Saved radial_wavefunction_comparison.html")
        
    # Overall R2 score of all eigenvalues (FEM vs SR)
    fem_energies = np.array(fem_energies)
    sr_energies = np.array(sr_energies)
    ss_res_e = np.sum((fem_energies - sr_energies)**2)
    ss_tot_e = np.sum((fem_energies - np.mean(fem_energies))**2)
    r2_energy = 1.0 - (ss_res_e / ss_tot_e)
    
    print(f"\nOverall R2 correlation score of the set of all eigenvalues (FEM vs SR): {r2_energy:.6f}")

if __name__ == "__main__":
    main()
