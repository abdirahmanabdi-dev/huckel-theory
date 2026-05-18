# Mathematical Foundations of Hückel Theory and Hermitian Solvers

This document provides a rigorous mathematical and theoretical overview of the underlying chemistry assumptions and linear algebra mechanics used by the computational solver. It demonstrates why a real symmetric chemical adjacency graph maps perfectly to a Hermitian eigenvalue problem, allowing numerical libraries like JAX to optimize the solution.

---

## 1. Core Assumptions of Hückel Theory

Hückel Molecular Orbital (HMO) theory simplifies the complex quantum mechanics of conjugated polyenes by isolating the $\pi$-electrons from the underlying $\sigma$-backbone. The framework operates under four primary approximations:

### I. The $\sigma$-$\pi$ Separation
The total molecular wavefunction is decoupled. The $\sigma$-bonds (single bonds) form a rigid, localized framework in the molecular plane, while the $2p_z$ atomic orbitals ($\phi$) mix independently to form delocalized $\pi$ molecular orbitals ($\psi$):

$$\psi = \sum_{i=1}^{N} c_i \phi_i = c_1\phi_1 + c_2\phi_2 + \dots + c_N\phi_N$$

### II. Zero Differential Overlap (ZDO)
The overlap integral ($S_{ij}$) between different atomic orbitals is assumed to be orthogonal and normalized, completely dropping to zero unless evaluating the exact same atom:

$$S_{ij} = \int \phi_i^* \phi_j \, d\tau = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### III. Constant Coulomb Integral ($\alpha$)
The energy of an electron resting entirely in an isolated carbon $2p_z$ orbital is assumed to be identical across all carbon positions. In our numerical matrix, this serves as the main diagonal baseline:

$$H_{ii} = \int \phi_i^* \hat{H} \phi_i \, d\tau = \alpha \xrightarrow{\text{scaled}} 0.0$$

### IV. Nearest-Neighbor Resonance Integral ($\beta$)
Electrons can only delocalize (or tunnel) between atoms that are directly adjacent/bonded in the molecular graph. All non-bonded long-range interactions drop to zero:

$$H_{ij} = \int \phi_i^* \hat{H} \phi_j \, d\tau = \begin{cases} \beta & \text{if atom } i \text{ is bonded to } j \xrightarrow{\text{scaled}} 1.0 \\ 0 & \text{if atom } i \text{ is NOT bonded to } j \end{cases}$$

---

## 2. The Linear Algebra Proof: Real Symmetric vs. Hermitian

When setting up the matrix array for the JAX solver, the system is passed to `eigh`, a solver optimized specifically for **Hermitian** systems. Below is the proof demonstrating why a real-numbered chemical graph fits this definition.

### Step A: The Definition of a Hermitian Matrix
A matrix $H$ is mathematically defined as Hermitian if it is equal to its own conjugate transpose ($H^\dagger$ or $H^H$):

$$H^H = (H^*)^T = H$$

Where:
* $H^*$ denotes the complex conjugate (flipping the sign of imaginary numbers, $i \to -i$).
* $T$ denotes the standard matrix transpose (reflecting rows and columns across the main diagonal).

### Step B: Eliminating the Complex Conjugate
Because the chemical connectivity matrix only tracks the absence (`0.0`) or presence (`1.0`) of a physical $\pi$-interaction, every entry in the matrix is a **purely real number** ($H_{ij} \in \mathbb{R}$). Because real numbers lack an imaginary component, taking the complex conjugate yields no change:

$$H^* = H$$

### Step C: Enforcing Transpose Symmetry
Chemical bonding is fundamentally a mutual relationship. If Atom $i$ shares a $\pi$-bond with Atom $j$, then Atom $j$ must share that exact same bond back with Atom $i$. Therefore:

$$H_{ij} = H_{ji}$$

By definition, a matrix where $H_{ij} = H_{ji}$ is perfectly symmetric and equal to its standard transpose:

$$H^T = H$$

### Final Derivation
Combining these properties yields the mathematical bridge:

$$H^H = (H^*)^T = (H)^T = H$$

Because $H = H^T = H^H$, your real symmetric chemical matrix **is completely identical to a Hermitian matrix**, making it fully compatible with Hermitian solvers.

---

## 3. Why This Symmetry Empowers `eigh` to Work

By establishing that the matrix is real and symmetric, the system unlocks **The Spectral Theorem** of linear algebra. This theorem guarantees three critical properties that numerical computers exploit for rapid calculation:

### 1. Purely Real Eigenvalues
It guarantees that all eigenvalues (the Molecular Orbital energies, $E$) are strictly real numbers ($\mathbb{R}$), ensuring the algorithm will never yield non-physical imaginary energy roots.

### 2. Mutually Perpendicular (Orthogonal) Vectors
It proves that the resulting eigenvectors (the wavefunction coefficients, $c_n$) are perfectly orthogonal to one another. Geometrically, they act as an independent, rigid coordinate framework:

$$\mathbf{c}_i \cdot \mathbf{c}_j = \delta_{ij}$$

### 3. Matrix Diagonalization Via Pure Rotation
Because of this rigid orthogonality, `eigh` completely bypasses the need to expand large, computationally intensive secular polynomials on paper. Instead, it treats the matrix as a geometric space and applies a sequence of orthogonal coordinate transformations (rotations):

$$C^T H C = \Lambda$$

Where:
* $H$ is the original molecular connectivity matrix.
* $C$ is the transformation matrix (the collected columns of your rotated coordinate axes).
* $\Lambda$ is a perfectly **diagonalized matrix** where all off-diagonal element interactions have been systematically crushed down to absolute zero:

$$\Lambda = \begin{bmatrix}
E_1 & 0 & 0 & \dots & 0 \\
0 & E_2 & 0 & \dots & 0 \\
0 & 0 & E_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & E_N
\end{bmatrix}$$

> **Conclusion:** When JAX finishes grinding away the off-diagonal entries using this iterative rotational geometry, it hands you the diagonal entries of $\Lambda$ as your **Energies**, and the columns of your rotation framework $C$ as your **Orbital Coefficients** simultaneously.
