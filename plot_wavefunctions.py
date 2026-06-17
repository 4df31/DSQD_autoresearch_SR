import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # 1. Load the clean and noisy datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(base_dir, "fem_dsqd_data.csv")
    noisy_path = os.path.join(base_dir, "fem_dsqd_data_noisy.csv")

    # Fallback to current working directory if not found in script folder
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        clean_path = "fem_dsqd_data.csv"
        noisy_path = "fem_dsqd_data_noisy.csv"

    print(f"Loading noiseless dataset: {clean_path}")
    df_clean = pd.read_csv(clean_path)

    print(f"Loading noisy dataset: {noisy_path}")
    df_noisy = pd.read_csv(noisy_path)

    # 2. Extract radial wavefunction by filtering for theta = 0.0
    # The radial part of the wavefunction corresponds to theta=0.0 where cos(n*theta) = 1
    # hence psi(r, theta=0) = R(r).
    df_clean_r = df_clean[df_clean['theta'] == 0.0].copy()
    df_noisy_r = df_noisy[df_noisy['theta'] == 0.0].copy()

    # 3. Identify unique quantum states and sort them by energy:
    # E = 2*s + n + 1 (analytical energy level), sub-sorting by s and then n.
    unique_states = df_clean_r[['s', 'n']].drop_duplicates().copy()
    unique_states['energy'] = 2 * unique_states['s'] + unique_states['n'] + 1
    unique_states = unique_states.sort_values(by=['energy', 's', 'n']).reset_index(drop=True)

    # Select the first 9 wavefunctions (first 9 states)
    first_9_states = unique_states.head(9)
    print("\nFirst 9 states to plot:")
    for idx, row in first_9_states.iterrows():
        print(f"  Plot {idx+1}: s={int(row['s'])}, n={int(row['n'])} (Analytical Energy E={int(row['energy'])})")

    # 4. Define subplot titles for the 3x3 grid
    subplot_titles = [
        f"State (s={int(row['s'])}, n={int(row['n'])}) | E={int(row['energy'])}"
        for _, row in first_9_states.iterrows()
    ]

    # Initialize 3x3 subplots. Use vertical_spacing and horizontal_spacing to
    # prevent any overlap of labels, titles, and ticks between plots.
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.16,      # Generous space to keep titles, x-labels, and y-labels separate
        horizontal_spacing=0.12     # Generous horizontal spacing to prevent y-axis labels and titles overlap
    )

    # 5. Define style colors and markers for premium visual appeal (Modern Light theme)
    color_noiseless = "#2563eb"  # Vibrant blue for continuous line
    color_noisy = "#f43f5e"      # Rose-pink for noisy wavefunction dots

    # 6. Add traces for each state
    for idx, row in first_9_states.iterrows():
        s_val = row['s']
        n_val = row['n']
        
        # Filter data for this specific quantum state
        clean_state = df_clean_r[(df_clean_r['s'] == s_val) & (df_clean_r['n'] == n_val)].sort_values(by='r')
        noisy_state = df_noisy_r[(df_noisy_r['s'] == s_val) & (df_noisy_r['n'] == n_val)].sort_values(by='r')
        
        r_vals = clean_state['r'].values
        psi_clean = clean_state['psi'].values
        psi_noisy = noisy_state['psi'].values
        
        # Subplot coordinates (1-based index)
        row_idx = (idx // 3) + 1
        col_idx = (idx % 3) + 1
        
        # Plot noiseless wavefunction as a continuous line
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=psi_clean,
                mode='lines',
                line=dict(color=color_noiseless, width=2.5),
                name="Noiseless FEM",
                legendgroup="noiseless",
                showlegend=(idx == 0)
            ),
            row=row_idx, col=col_idx
        )
        
        # Plot noisy wavefunction as dots
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=psi_noisy,
                mode='markers',
                marker=dict(color=color_noisy, size=3, opacity=0.8),
                name="Noisy Data (Gaussian Noise)",
                legendgroup="noisy",
                showlegend=(idx == 0)
            ),
            row=row_idx, col=col_idx
        )
        
        # Update axes properties with custom spacing, labels, and ticks for light theme
        fig.update_xaxes(
            title=dict(text="Radius (r)", font=dict(size=11, color="#1e293b")),
            row=row_idx, col=col_idx,
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e0",
            tickfont=dict(size=9, color="#64748b"),
            showgrid=True,
            zeroline=True
        )
        fig.update_yaxes(
            title=dict(text="R(r)", font=dict(size=11, color="#1e293b")),
            row=row_idx, col=col_idx,
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e0",
            tickfont=dict(size=9, color="#64748b"),
            showgrid=True,
            zeroline=True
        )

    # 7. Apply a premium light layout style and ensure no overlaps
    fig.update_layout(
        title=dict(
            text="<b>Comparison of Noiseless and Noisy Radial Wavefunctions R(r)</b><br><sup>First 9 Quantum States Ordered by Energy E = 2s + n + 1</sup>",
            font=dict(size=18, family="Inter, sans-serif", color="#0f172a"),
            x=0.5,
            y=0.96,
            xanchor='center',
            yanchor='top'
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",    # Modern white background
        plot_bgcolor="#ffffff",     # Sleek white background for subplots
        font=dict(family="Inter, sans-serif", color="#334155"),
        height=1000,
        width=1200,
        margin=dict(l=80, r=40, t=110, b=80),  # Generous margins
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#334155"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0)"
        ),
        hovermode="closest"
    )

    # Clean up and style subplot annotation text to avoid any font size overlap
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12, color="#0f172a", family="Inter, sans-serif")

    # 8. Save the output plot
    output_html = os.path.join(base_dir, "wavefunctions_comparison.html")
    fig.write_html(output_html)
    print(f"\nSuccess! Interactive plot saved to: {output_html}")

if __name__ == "__main__":
    main()
