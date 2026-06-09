# Symbolic regression performed by AI agents

The Karpathy's autoresearch architechture tool has been used to implement the symbolic regression (SR) algorithms in order to match the literature reported analytical expression of the disk shaped quantum dots. 

## Some interesting facts

The model creates numerical data training set based on Finite Elements Method (FEM) to find the wavefunctions and their corresponding eigenvalues numerically by solving the time independent Schrödinger equation of the two-dimensional Isotropic Harmonic Oscillator (2DIHO), then the symbolic regression is being exeuted based on some hypothesis (ansatz) and the agent iterates running experiments to maximize the R² correlation coefficient between the updated ansatz modeled and the corresponding analytical solution. 

