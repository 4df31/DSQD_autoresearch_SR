# Autoresearch: DSQD Symbolic Regression
This is an experiment to have an LLM autonomously conduct research. The specific objective of this run is to perform multi-state symbolic regression to rediscover the unified analytical wavefunction expression of the envelope functions for disk-shaped quantum dots (DSQD) under isotropic harmonic confinement, varying both radial quantum number $s$ and angular momentum quantum number $n$.

The ultimate goal is to find the unified multi-state expression that fits all target states  present in the experimental FEM datasets for different values of $s$ and $n$ simultaneously.

## Analytical Model (Target Expression)

For disk-shaped quantum dots (DSQDs), the real in-plane envelope wavefunction $\psi(r, \theta, n, s)$ under isotropic confinement frequency $\omega = 1$ is represented numerically in the datasets.

## Procedure

The continuous automated research loop is responsible for refining the following workflow:

- Symbolic Search: train.py loads the dataset (use extrictly the file `fem_dsqd_data_noisy.csv` ), filters for the first 2 eigenfunctions (varying $s$ and $n$ in a suitable way), normalizes the wavefunctions state-by-state, and uses PySR to propose unified mathematical models of `$\psi(r,\theta)$_{n,s}$`.

- Evaluation: Evaluate the proposed unified function against the state-by-state normalized dataset, aligning the sign of each state's prediction and computing the overall $R^2$ score.

- Iteration: Loop the symbolic regression search until the discovered expressions match the unified physical expression (Target $R^2 \approx 1.0$).

- Verify data exists: Check that `fem_dsqd_data.csv` is present in the repository. If datasets files are not present notify the human
- Initialize results.tsv: Create results.tsv with the header row.


### what if it does not matches?
be creative heed the physical constraints in order to modify the symbolic expression to match the True one. Act as an expert in physical mathematics and differential equations to improve the fitting.


# Experimentation Constraints

The script runs for a fixed budget (e.g., 5 minutes wall-clock per search) on a single RTX 5090 GPU.

What you CAN do:

- Modify train.py. This is the ONLY file you edit. You can alter PySR binary/unary operators, equation complexities, populations, parsimony penalties, and custom loss formulations.

What you CANNOT do:

- DO NOT modify prepare.py. It is read-only. It contains the fixed FEM solver, ground truth wavefunctions, and the $R^2$ validation harness.
- DO NOT modify the evaluation harness. The evaluate_r2 function in prepare.py is the immutable ground truth metric.

Simplicity Criterion: Symbolic regression relies heavily on parsimony. A small improvement in $R^2$ that adds ugly complexity (overfitting) is discarded. Weigh complexity cost against improvement magnitude.

# Output Format & Logging

Once train.py finishes, it must print a summary:

---
best_r2_score:    1.000000
complexity:       21
search_seconds:   300.1
total_seconds:    315.4
peak_vram_mb:     12040.2
best_equation:    "`$\psi(r,\theta)_{s,n}$= <LaTeX-format written equation>`"


Log the results to results.tsv:
commit \t best_r2_score \t complexity \t memory_gb \t status \t description
(status: keep, discard, or crash)

The Experiment Loop

LOOP FOREVER:

Look at the git state (current branch/commit).

Tune train.py with an experimental idea by directly hacking the PySR config.

git commit -am "experiment detail"

Run the experiment: python train.py > run.log 2>&1

Read out results: grep "^best_r2_score:\|^complexity:" run.log

If grep is empty (crash), read tail -n 50 run.log to fix the stack trace.

Record results in results.tsv.

If best_r2_score improved or complexity decreased for the same score: Keep the commit.

If performance is worse: git reset --hard HEAD~1 back to where you started.

NEVER STOP: You are autonomous. If you run out of ideas, try different basis functions (e.g., add sqrt, besselj to PySR), adjust parsimony, or regress the logarithm of the wavefunction. The loop runs until human interruption.

## Hardware and Environment

Host: Nvidia GeForce RTX 5090, Intel Core Ultra 9-258, 64GB RAM, Fedora Linux 44.

System details: Python 3.10.12, CUDA 13.2, NVIDIA-SMI 595.71.05.

Key Pip Packages Available: pysr (1.0.0), juliacall (0.9.23), torch (2.12.0+cu128), scipy (1.15.3), sympy (1.14.0), numpy (2.2.6), pandas (2.3.3).

# Autoresearch Enhancements & Directives

## 1. Host Automation (Creation & Updates)
To completely automate the setup, deployment, and updating of the Autoresearch host:
- Deploy the host environment using the pre-configured [Dockerfile](file:///home/QUCIT/adrian/ML/Dockerfile) and [start.sh](file:///home/QUCIT/adrian/ML/start.sh).
- Automate git synchronization inside the container by trusting the mounted workspace directory:
  ```bash
  git config --global safe.directory *
  ```
- Trigger automated execution upon launch by setting the entrypoint command to execute the agent:
  ```bash
  python3 /workspace/research_agent.py
  ```

## 2. GPU Enhancement in Symbolic Regression
To maximize GPU/VRAM acceleration during symbolic regression training:
- **PySR/Julia CUDA Backend**: Ensure PySR utilizes the GPU-enabled Julia packages (e.g., using `pysr.install()` with appropriate CUDA capability) when evaluating candidate equations.
- **PyTorch Integration**: Leverage GPU acceleration during the initial numerical solve in [prepare.py](file:///home/QUCIT/adrian/ML/DSQD_autoresearch_SR/prepare.py) to generate the ground truth dataset directly on CUDA.
- PySR automatically interfaces with Julia's multiprocessing and GPU-parallelized execution if CUDA-compatible packages are installed in the active Julia environment.

## 3. Wavefunction R² and Eigenenergy Visualization (plotly)
To visualize search performance and contrast numerical results:
- Run [visualize.py](file:///home/QUCIT/adrian/ML/DSQD_autoresearch_SR/visualize.py) to compute and plot:
  - An interactive 2D heatmap of the wavefunction $R^2$ scores depending on the radial quantum number $s$ and angular momentum $n$.
  - A comparative group bar chart for the eigenenergies contrasting **Analytical**, **FEM** (numerical eigenvalues), and **SR** (the expectation values of the Hamiltonian $\langle \psi_{SR} | \hat{H} | \psi_{SR} \rangle$ computed using the symbolic-regressed wavefunctions).
  - A progress line chart showing the progress of $R^2$ across the iterations.
  - The overall $R^2$ correlation score of the set of all eigenvalues (FEM vs SR).
