Autoresearch: DSQD Symbolic Regression

This is an experiment to have an LLM autonomously conduct research. The specific objective of this run is to perform symbolic regression to rediscover the analytical eigenfunctions and eigenenergies of the envelope functions for disk-shaped quantum dots (DSQD) using the confinement potential of the 2D isotropic harmonic oscillator (2DIHO).

The ultimate goal is to match the target analytical model of Quinteiro (radial function based on Laguerre polynomials and angular function as normal modes of the azimuthal oscillatory solution).

Analytical Model (Target Expression)

For disk-shaped quantum dots (DSQDs), the electron's wave function can be expressed as the product of three distinct components: a microscopic cell-periodic (Bloch) function, an envelope function, and a spin part $\xi$. Formally, this reads:


$$\psi(\mathbf{r}) = \phi(r, \theta) Z(z) u(\mathbf{r}) \xi \tag{1}$$


where $u(\mathbf{r})$ is the Bloch function, $\phi(r,\theta)Z(z)$ is the envelope function, and $\xi$ is the spin part.

Envelope Function: Separation and Analytical Form
The in-plane confinement potential is well approximated by a two-dimensional harmonic oscillator:


$$V_i(r) = \frac{1}{2} m_i^* \omega_{0i}^2 r^2$$


In the presence of an external magnetic field $B$ applied along the $z$-axis, the in-plane envelope function eigenstates take the explicit analytical form:


$$\phi_{isn}(r,\theta) = \frac{(-1)^s}{\sqrt{2\pi}\ell_i} \sqrt{\frac{s!}{(s+|n|)!}} e^{-r^2/(4\ell_i^2)} \left(\frac{r}{\sqrt{2}\ell_i}\right)^{|n|} L_s^{|n|}\left(\frac{r^2}{2\ell_i^2}\right) e^{-in\theta} = R_{isn}(r)e^{-in\theta} \tag{2}$$


Here, $\ell_i^2 = \hbar/(2m_i^*|\omega_i|)$ is the characteristic confinement length, $\omega_i^2 = \omega_{0i}^2 + \Omega_i^2/4$ is the effective frequency, and $L_s^{|n|}$ is a generalized Laguerre polynomial ($s$ is the radial quantum number, $n$ is the $z$-projection of the orbital angular momentum).

Procedure

The continuous automated research loop is responsible for refining the following workflow:

Data Generation: prepare.py generates a training set based on the Finite Elements Method (FEM) solution of the Schrödinger equation of the 2DIHO for the first 20 eigenfunctions.

Symbolic Search: train.py uses the Julia symbolic regression package (PySR) to propose mathematical models to match the radial and azimuthal parts of the FEM solution.

Evaluation: Evaluate proposed mathematical functions against the FEM dataset computing the $R^2$ score.

Iteration: Loop the symbolic regression search until the expressions match the theoretical physical expressions outlined above (Target $R^2 \approx 1.0$).

Setup (Human-in-the-Loop)

To set up a new experiment:

Agree on a run tag (e.g., dsqd-mar5).

Create branch: git checkout -b autoresearch/<tag>.

Verify data exists: Check that ~/.cache/autoresearch/ contains FEM tensors. If not, the human must run uv run prepare.py.

Initialize results.tsv: Create results.tsv with the header row.

Kick off experimentation by launching the agent.

Experimentation Constraints

The script runs for a fixed budget (e.g., 5 minutes wall-clock per search) on a single RTX 5090 GPU.

What you CAN do:

Modify train.py. This is the ONLY file you edit. You can alter PySR binary/unary operators, equation complexities, populations, parsimony penalties, and custom loss formulations.

What you CANNOT do:

DO NOT modify prepare.py. It is read-only. It contains the fixed FEM solver, ground truth wavefunctions, and the $R^2$ validation harness.

DO NOT modify the evaluation harness. The evaluate_r2 function in prepare.py is the immutable ground truth metric.

Simplicity Criterion: Symbolic regression relies heavily on parsimony. A small improvement in $R^2$ that adds ugly complexity (overfitting) is discarded. Weigh complexity cost against improvement magnitude.

Output Format & Logging

Once train.py finishes, it must print a summary:

---
best_r2_score:    0.987541
complexity:       14
search_seconds:   300.1
total_seconds:    315.4
peak_vram_mb:     12040.2
best_equation:    "exp(-x1^2 / 4) * L_s(x1^2 / 2) * cos(n * x2)"


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
