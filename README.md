# Hückel Theory Solver

A Python solver for Hückel Molecular Orbital (HMO) theory, built using JAX. Give it a conjugated molecule's carbon count and shape — linear chain or ring — and it returns the π-orbital energies and wavefunction coefficients.

---

## What is Hückel Theory?

In organic chemistry, conjugated molecules (like benzene or butadiene) have electrons that aren't locked between two atoms — they're smeared across the whole molecule through overlapping p orbitals. These are called **π electrons**, and they're responsible for a lot of interesting chemistry: colour, stability, reactivity.

Hückel theory is a way of calculating the energies of these π electrons without having to solve the full Schrödinger equation for every electron in the molecule. It does this by making four simplifying assumptions:

**1. Ignore the σ-framework.** The single bonds holding the molecule together are treated as a rigid scaffold. We only care about the π electrons sitting above and below the molecular plane.

**2. Orbitals on different atoms don't overlap (ZDO).** In reality they do slightly, but the approximation is good enough and it massively simplifies the maths. Each atomic orbital is treated as independent.

**3. Every carbon looks the same (constant α).** The on-site energy of a π electron on any carbon atom is set to the same reference value α. In the matrix, this just becomes zeros on the diagonal.

**4. Only bonded neighbours interact (nearest-neighbour β).** A π electron can only delocalise between atoms that are directly bonded. Everything else is zero. In the matrix, a bond between atoms i and j becomes a 1 in positions (i,j) and (j,i).

These four rules turn a quantum mechanical problem into a neat matrix — a grid of 0s and 1s that maps out which atoms are connected.

---

## From Chemistry to Linear Algebra

Once you have the matrix, finding the orbital energies is an eigenvalue problem. The matrix **H** encodes the connectivity of the molecule; its eigenvalues are the orbital energies, and its eigenvectors are the orbital coefficients (how much each atom contributes to each orbital).

There's a useful mathematical shortcut here. JAX has a solver called `eigh` that's specifically optimised for **Hermitian** matrices — matrices equal to their own conjugate transpose. It's faster and more numerically stable than a general solver.

It turns out the Hückel matrix is Hermitian, and the proof is simple:

- Every entry is a real number (0 or 1), so taking the complex conjugate changes nothing.
- Bonding is mutual — if atom i bonds to atom j, then j bonds to i — so the matrix is symmetric: H[i,j] = H[j,i].

A real symmetric matrix satisfies the Hermitian condition exactly. This unlocks the **Spectral Theorem**, which guarantees that all eigenvalues are real numbers (no imaginary energies) and that the eigenvectors are orthogonal to each other. `eigh` exploits this to diagonalise the matrix efficiently, giving you energies and orbital coefficients in one shot.

---

## The Code

```python
from jax.lax.linalg import eigh
import jax.numpy as jnp

def solve_polyene(n_carbons, molecule_type):
    # Build an N×N zero matrix
    h = jnp.zeros((n_carbons, n_carbons), dtype=jnp.float32)

    # Fill in bonding interactions along the chain
    for i in range(n_carbons - 1):
        h = h.at[i, i+1].set(1.)
        h = h.at[i+1, i].set(1.)

    # Close the ring if cyclic
    if molecule_type in ("cyclic", "c", "ring"):
        h = h.at[0, n_carbons - 1].set(1.0)
        h = h.at[n_carbons - 1, 0].set(1.0)

    # Diagonalise — returns orbital coefficients and energies
    coeffs, energies = eigh(h)
    return jnp.round(coeffs, 3), jnp.round(energies, 3)
```

Run it:

```
Enter the length of your polyene (e.g. Ethene=2): 6
What type of molecule is it? cyclic/linear: cyclic
```

This gives you benzene (6 carbons, ring) — the classic test case for Hückel theory.

---

## Dependencies

```
jax
jaxlib
```
