# Autoresearch: DSQD Symbolic Regression

This is an experiment to have an LLM autonomously conduct research. The specific objective of this run is to perform multi-state symbolic regression to discover the unified analytical wavefunction expression of the envelope functions for disk-shaped quantum dots (DSQD) under isotropic harmonic confinement, varying both radial quantum number $s$ and angular momentum quantum number $n$.

The ultimate goal is to find the unified multi-state expression that fits all target states ($s \in \{0, 5\}$ and $n \in \{0, 10\}$) simultaneously.

## Analytical Model (Target Expression)

The expression for the inplane component of the wavefunction  is $\phi_{s,n}(r,\theta)= R(r)\Theta(\theta)$ where $R(r)$ is the radial profile solution from the one dimensional schrodinger equation for the radial part and $\Theta(\theta)$ is the azymuthal contribution from the angular part with $n,s$ strictly integers. Note that $\phi_{s,n}$ has to be normalized n order to preserve the probability so the explicit form of $R(r)$ has to be written in terms of a normalization coefficcient. Look for the corresponding eigen energies for each state based on the results from de Finite Element Method computed in the file `fem_dsqd_data.csv` and obtain the corresponding analytical expression for the energies of the system ($E=E(\omega_{0},s,n)$).
Heed that the angular frequency of the harmonic oscillator is written by the expression 

$$\omega_{0}=\frac{\hbar}{2m\ell^{2}}$$

where $\ell$ is named the characteristic length of the confinement of the electrons and $m$ is the effective mass of the charge carrier (electron/hole).

## Procedure

The continuous automated research loop is responsible for refining the following workflow:

- Data Generation: prepare.py generates a dataset for the first 20 eigenfunctions (varying $s$ and $n$ quantum numbers up to energy level $E=8$) and caches them directly in the repository as `fem_dsqd_data.csv`. This ensures numerical results are allocated inside the repository instead of being regenerated each time. you cannot read or write the prepare.py file 

- Symbolic Search: train.py uses PySR to propose unified mathematical models of $R$ and $\Theta$ omega in terms of $s$, $n$, $m$ and $\ell$. you can update this file and apply physical and mathematical concepts in order to create the analytical expression. if the train.py file does not exist create one from scratch. you have a powerfull GPU (RTX5090) exploit its computing capacity.

- Evaluation: Evaluate the proposed unified function against the state-by-state normalized dataset, aligning the sign of each state's prediction and energy's prediction and computing the overall $R^2$ score. Evaluate that the proposed eigenenergy matches the one in the FEM dataset state-by-state.

- Iteration: Loop the symbolic regression search until the discovered expressions match the unified physical expression (Target $R^2 \approx 1.0$).

## what if it does not matches?
be creative heed the physical constraints in order to modify the symbolic expression to match the True one. Act as an expert in physical mathematics and differential equations to improve the fitting using special functions and different sets of basis functions , e.g., polynomials of jacobi, hermite, laguerre, hypergeometric functions, etc...

# Setup (Human-in-the-Loop)

To set up a new experiment:

1. Agree on a run tag (e.g., dsqd-mar5).
2. Create branch: git checkout -b autoresearch/<tag>.
3. Verify data exists: Check that `fem_dsqd_data.csv` is present in the repository. If not, run python prepare.py to generate it.
4. Initialize results.tsv: Create results.tsv with the header row.
5. Kick off experimentation by launching the agent.


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
best_equation:    "exp(-0.5 * r^2) * (r^n) * (1.0 + s * (n - (n+2)/3.0 * r^2)) * cos(n * theta)"


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

Hardware and Environment

Host: Nvidia GeForce RTX 5090, Intel Core Ultra 9-258, 64GB RAM, Fedora Linux 44.

System details: Python 3.10.12, CUDA 13.2, NVIDIA-SMI 595.71.05.

Key Pip Packages Available: pysr (1.0.0), juliacall (0.9.23), torch (2.12.0+cu128), scipy (1.15.3), sympy (1.14.0), numpy (2.2.6), pandas (2.3.3).

# Autoresearch Enhancements & Directives

## 1. Host Automation (Creation & Updates)
To completely automate the setup, deployment, and updating of the Autoresearch host:
- Deploy the host environment using the pre-configured [Dockerfile](file:///home/QUCIT/adrian/ML/Dockerfile).
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
- PySR automatically interfaces with Julia's multiprocessing and GPU-parallelized execution if CUDA-compatible packages are installed in the active Julia environment.

## 3. Wavefunction R² and Eigenenergy Visualization (plotly)
To visualize search performance and contrast numerical results:
- create [visualize.py](file:///home/QUCIT/adrian/ML/DSQD_autoresearch_SR/visualize.py) to compute and plot:
  - An interactive 2D heatmap of the wavefunction $R^2$ scores depending on the radial quantum number $s$ and angular momentum $n$.
  - A comparative group bar chart for the eigenenergies contrasting **Analytical** ($E = 2s + n + 1$), **FEM** (numerical eigenvalues), and **SR** (the expectation values of the Hamiltonian $\langle \psi_{SR} | \hat{H} | \psi_{SR} \rangle$ computed using the symbolic-regressed wavefunctions).
  - A progress line chart showing the progress of $R^2$ across the iterations.
  - The overall $R^2$ correlation score of the set of all eigenvalues (FEM vs SR).
