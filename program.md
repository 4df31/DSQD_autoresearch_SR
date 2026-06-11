# Autoresearch: DSQD Symbolic Regression

This is an experiment to have an LLM autonomously conduct research. The specific objective of this run is to perform multi-state symbolic regression to rediscover the unified analytical wavefunction expression of the envelope functions for disk-shaped quantum dots (DSQD) under isotropic harmonic confinement, varying both radial quantum number $s$ and angular momentum quantum number $n$.

The ultimate goal is to find the unified multi-state expression that fits all four target states ($s \in \{0, 1\}$ and $n \in \{0, 1\}$) simultaneously.

## Analytical Model (Target Expression)

For disk-shaped quantum dots (DSQDs), the real in-plane envelope wavefunction $\psi(r, \theta, n, s)$ under isotropic confinement frequency $\omega = 1$ is represented numerically.

Due to numerical discretization at $r \to 0$, the centrifugal term $-0.25/(2r^2)$ is omitted for $n = 0$, while for $n = 1$ the centrifugal potential $+0.75/(2r^2)$ is included. This modifies the $(s=1, n=0)$ radial state slightly. 

The resulting 4 wavefunctions corresponding to the first 2 eigenfunctions varying $s \in \{0, 1\}$ and $n \in \{0, 1\}$ have the following exact forms:
1. $s=0, n=0: \psi \propto e^{-0.5 r^2}$
2. $s=1, n=0: \psi \propto (1 - \frac{2}{3} r^2) e^{-0.5 r^2}$
3. $s=0, n=1: \psi \propto r e^{-0.5 r^2} \cos(\theta)$
4. $s=1, n=1: \psi \propto r (2 - r^2) e^{-0.5 r^2} \cos(\theta)$

These four states can be unified into a single analytical expression of $r$, $\theta$, $n$, and $s$:

$$\psi(r, \theta, n, s) = N_{sn} e^{-0.5 r^2} r^n \left( 1 + s \left( n - \frac{n+2}{3} r^2 \right) \right) \cos(n \theta)$$

where $N_{sn}$ is a state-dependent normalization constant.

## Procedure

The continuous automated research loop is responsible for refining the following workflow:

- Data Generation: prepare.py generates a dataset for the first 20 eigenfunctions (varying $s$ and $n$ quantum numbers up to energy level $E=8$) and caches them directly in the repository as `fem_dsqd_data.csv`. This ensures numerical results are allocated inside the repository instead of being regenerated each time.

- Symbolic Search: train.py loads the dataset, filters for the first 2 eigenfunctions (varying $s \in \{0, 1\}$ and $n \in \{0, 1\}$), normalizes the wavefunctions state-by-state, and uses PySR to propose unified mathematical models of $r$, $\theta$, $n$, and $s$.

- Evaluation: Evaluate the proposed unified function against the state-by-state normalized dataset, aligning the sign of each state's prediction and computing the overall $R^2$ score.

- Iteration: Loop the symbolic regression search until the discovered expressions match the unified physical expression (Target $R^2 \approx 1.0$).

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
