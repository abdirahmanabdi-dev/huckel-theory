from jax.lax.linalg import eigh
import jax.numpy as jnp

"""
For a general case of Huckel theory for a conjugated polyene molecule, we know the determinant is an NxN matrix
Due to the connectivity rules we can write out a matrix for any NxN system, the diagonals are always 0. and |i-j|=1, because directly bonded
atoms in the matrix is just beta which is 1 in our case. And the rest of the matrix is 0, as we assume non-bonding atoms don't interact.
"""

det_ally_radical = jnp.array(
    [[0., 1., 0.],
    [1., 0., 1.],
    [0., 1., 0.]], dtype=jnp.float32)

def solve_polyene(n_carbons, molecule_type):
    """
    Builds and solves the Hückel matrix for a linear conjugated polyene. 
    As the matrix is symmetric and real it is equal to its own conjugate transpose (Hermitian, where A = A^T = A^H). 
    This mathematical property means that we get purely real eigenvalues (energies) and eigenvectors.
    """
    h = jnp.zeros((n_carbons, n_carbons), dtype=jnp.float32)
    
    for i in range(n_carbons-1):
        h = h.at[i, i+1].set(1.)
        h = h.at[i+1, i].set(1.)
        
    match molecule_type.lower().strip():
        case "cyclic" | "c" | "ring":
            # Add the ring closure connection
            h = h.at[0, n_carbons - 1].set(1.0)
            h = h.at[n_carbons - 1, 0].set(1.0)
            print("\n[System Info] Closed-ring boundary conditions applied.")
            
        case "linear" | "l" | "chain":
            # No changes needed, it's already a chain
            print("\n[System Info] Open-chain linear boundary conditions applied.")
            
        case _:
            # Default fallback if they type something weird
            print("\n[Warning] Unknown type entered. Defaulting to a linear chain.")

    coeffs, energies = eigh(h)
    return jnp.round(coeffs, decimals=3), jnp.round(energies, decimals=3)

if __name__ == "__main__": 
    N = int(input("Enter the length of your polyene (e.g. Ethene=2): "))
    mol_shape = input("What type of molecule is it cyclic (cyclic, c, ring) or linear (linear, l, chain)?")
    coeffs, energies = solve_polyene(N, mol_shape)
    print(f"-- Coeffeicents for polyene (N={N}) ---)")
    print(coeffs)
    print(f"-- Energies for polyene ---)")
    print(energies)