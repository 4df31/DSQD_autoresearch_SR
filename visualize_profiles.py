import numpy as np
import pandas as pd
import os
import sys
import math
from scipy.special import genlaguerre
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Append current directory to path to load prepare.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare import load_fem_data

# Radial grid setup (must match prepare.py)
grid_size = 200
R_max = 10.0
dr = R_max / grid_size
r_grid = np.linspace(dr, R_max, grid_size)

def quinteiro_radial_profile(r, s, n):
    """
    Computes Quinteiro's analytical radial profile for quantum numbers s and n:
    R_sn(r) = (-1)^s * sqrt(2 * s! / (s+n)!) * exp(-0.5 * r^2) * r^n * L_s^n(r^2)
    """
    r2 = r**2
    # L_s^n polynomial
    poly = genlaguerre(int(s), int(n))
    L_val = poly(r2)
    
    # Factorial calculation
    coef = math.sqrt(2.0 * math.factorial(int(s)) / math.factorial(int(s + n)))
    sign = (-1.0)**int(s)
    
    R = sign * coef * np.exp(-0.5 * r2) * (r**n) * L_val
    return R

def sr_radial_profile(r, s, n):
    """
    Computes the symbolic regressed radial profile:
    exp(-0.5 * r2) * (r^n) * ((-1)^s * sqrt(2 * s! / (s+n)!)) * L_SR(r2, s, n)
    """
    r2 = r**2
    x0 = n
    x1 = s
    x2 = r2
    
    s_int = int(round(s))
    sn_int = int(round(s + n))
    coef = math.sqrt(2.0 * math.factorial(s_int) / math.factorial(sn_int))
    sign = (-1.0)**s_int
    
    P_SR = (((x2 + ((1.9050109 - x1) * 28.332409)) * ((((x1 * x0) + (1.6565776 - x2)) * ((x1 - 0.8160357) * x1)) - 0.1692691)) + 3.2703564) * -0.1592712
    
    val = np.exp(-0.5 * r2) * (r**n) * sign * coef * P_SR
    return val

def main():
    # 1. Load FEM Data
    df = load_fem_data()
    
    # Get all unique states (s, n) sorted by energy E = 2s + n + 1
    state_keys = df[['s', 'n']].drop_duplicates().copy()
    state_keys['energy'] = 2 * state_keys['s'] + state_keys['n'] + 1
    state_keys = state_keys.sort_values(by=['energy', 's', 'n']).reset_index(drop=True)
    
    # Select the first 20 states
    selected_states = state_keys.head(20)
    print("Found 20 states to plot:")
    for idx, row in selected_states.iterrows():
        print(f"Subplot {idx+1}: s={int(row['s'])}, n={int(row['n'])} (E={row['energy']})")
        
    # 2. Setup Subplot Grid (5 rows, 4 columns)
    fig = make_subplots(
        rows=5, cols=4,
        subplot_titles=[f"s={int(row['s'])}, n={int(row['n'])}" for _, row in selected_states.iterrows()],
        vertical_spacing=0.06,
        horizontal_spacing=0.06
    )
    
    # Colors for the three profiles
    color_fem = "#ff7f0e"       # Orange
    color_quint = "#1f77b4"     # Blue
    color_sr = "#2ca02c"        # Green

    # Loop through each state and populate subplots
    for idx, row in selected_states.iterrows():
        s_val = int(row['s'])
        n_val = int(row['n'])
        
        # Calculate row and col for subplot (1-indexed)
        r_idx = (idx // 4) + 1
        c_idx = (idx % 4) + 1
        
        # Extract numerical FEM radial profile (at theta = 0)
        # We find theta closest to 0
        state_df = df[(df['s'] == s_val) & (df['n'] == n_val)].copy()
        min_theta = state_df['theta'].abs().min()
        radial_df = state_df[state_df['theta'].abs() == min_theta].sort_values(by='r').reset_index(drop=True)
        
        r_fem = radial_df['r'].values
        psi_fem = radial_df['psi'].values
        
        # Normalize FEM profile
        norm_fem = np.linalg.norm(psi_fem)
        if norm_fem > 0:
            psi_fem = psi_fem / norm_fem
            
        # Align sign: ensure start of wavefunction near origin is positive
        # (similar to how prepare.py aligns signs)
        if np.sum(psi_fem[:10]) < 0:
            psi_fem = -psi_fem
            
        # Compute Quinteiro's analytical profile
        psi_quint = quinteiro_radial_profile(r_fem, s_val, n_val)
        norm_quint = np.linalg.norm(psi_quint)
        if norm_quint > 0:
            psi_quint = psi_quint / norm_quint
            
        # Align Quinteiro sign with FEM
        dot_quint = np.dot(psi_fem, psi_quint)
        if dot_quint < 0:
            psi_quint = -psi_quint
            
        # Compute Symbolic Regressed profile
        psi_sr = sr_radial_profile(r_fem, s_val, n_val)
        norm_sr = np.linalg.norm(psi_sr)
        if norm_sr > 0:
            psi_sr = psi_sr / norm_sr
            
        # Align SR sign with FEM
        dot_sr = np.dot(psi_fem, psi_sr)
        if dot_sr < 0:
            psi_sr = -psi_sr
            
        # Add traces to subplot
        # Only show legend for the first subplot to keep layout clean
        show_legend = (idx == 0)
        
        fig.add_trace(
            go.Scatter(
                x=r_fem, y=psi_fem,
                mode="lines",
                name="Numerical FEM",
                line=dict(color=color_fem, width=2.5),
                showlegend=show_legend
            ),
            row=r_idx, col=c_idx
        )
        
        fig.add_trace(
            go.Scatter(
                x=r_fem, y=psi_quint,
                mode="lines",
                name="Quinteiro Analytical",
                line=dict(color=color_quint, width=1.5, dash="dash"),
                showlegend=show_legend
            ),
            row=r_idx, col=c_idx
        )
        
        fig.add_trace(
            go.Scatter(
                x=r_fem, y=psi_sr,
                mode="lines",
                name="Symbolic Regressed",
                line=dict(color=color_sr, width=1.5, dash="dot"),
                showlegend=show_legend
            ),
            row=r_idx, col=c_idx
        )
        
        # Format axes limits for clean visualization
        fig.update_xaxes(range=[0, 6], row=r_idx, col=c_idx)
        
    # Update layout
    fig.update_layout(
        title=dict(
            text="Superposition of Radial Profiles for the First 20 States (s, n)",
            x=0.5,
            font=dict(size=22, color="#2c3e50")
        ),
        height=1200,
        width=1200,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )
    
    # Save to HTML
    output_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize_profiles.html")
    fig.write_html(output_html)
    print(f"\nInteractive radial profile superpositions successfully saved to: {output_html}")

if __name__ == "__main__":
    main()
