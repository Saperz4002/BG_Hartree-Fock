# BG_Hartree-Fock
# Noninteracting initialization for Bernal bilayer graphene

This package computes the single-particle eigenvalues, eigenvectors, occupations,
and initial density matrix `rho0` **before** a Hartree–Fock calculation.
It uses the four-band continuum Hamiltonian in Eq. (1) of:

E. Aguilar-Méndez, T. Neupert, and G. Wagner,
*Full, three-quarter, half and quarter Wigner crystals in Bernal bilayer graphene*,
[arXiv:2505.09685](https://arxiv.org/abs/2505.09685).

The interaction and Wigner-crystal band folding are not part of this initializer.
All four sublattice/layer bands are retained. The paper subsequently projects onto
the highest valence band for its Wigner-crystal calculation; that later projection
is a separate step.

## Start here

Use Python 3.10 or newer. From this directory:

```bash
python -m pip install -r requirements.txt
python bernal_initializer.py --D-meV=50 --density-cm2=-5e11 --temperature-K=0
```

This writes `initial_state.npz` and a readable `initial_state.json` summary.
The bundled `example_initial_state.npz` and `.json` contain an executed example
with the same settings. Open `01_noninteracting_initialization.ipynb` in Jupyter
or VS Code for the explained calculation, numerical matrices, and plots. Keep
the notebook and `bernal_initializer.py` in the same directory.

NumPy's `linalg.eigh` solves the explicit Hermitian matrices; SciPy supplies the
finite-temperature number solver. PySCF is not a dependency at this stage.
Matplotlib is used only by the notebook.

## Inputs and units

| Input | Default | Interpretation |
|---|---:|---|
| `t0_eV` | 2.61 | Intralayer hopping |
| `t1_eV` | 0.361 | Vertical dimer hopping |
| `t3_eV` | 0.283 | Skew interlayer hopping |
| `t4_eV` | 0.138 | Skew interlayer hopping |
| `delta_prime_eV` | 0.015 | Extra onsite energy of the dimer sites |
| `a_angstrom` | 2.46 | Graphene Bravais lattice constant |
| `D_eV` | 0.050 | Editable layer **energy bias**, in eV |
| `temperature_K` | 0 | Temperature in kelvin |
| `target_density_cm2` | -5e11 | Editable total carrier density, in cm^-2 |
| `shells` | 24 | Mesh resolution: 1 + 3*shells*(shells+1) points |
| `cutoff_times_a` | 0.12 | Dimensionless product k_max*a |

The hoppings and onsite shift are the paper's parameters. `D=50 meV` and
`n=-5e11 cm^-2` are example operating conditions, not unique material constants.
If your experimental control is a displacement field in V/nm, convert it to a
layer energy difference before assigning `D_eV`.

The default mesh follows the paper's **translation-preserving comparison** in
Sec. II: 1,801 triangular-lattice points in a hexagon of side/radius `0.12/a`,
with nearest-neighbor spacing `0.005/a`. It is distinct from the paper's
13-by-13 reduced-zone mesh plus nine Wigner reciprocal-lattice shifts.

The integration weights are an explicit implementation choice: triangular nodal
quadrature, with relative weights 1 inside, 1/2 on straight edges, and 1/3 at
the six corners. These weights integrate the area of the specified hexagon
exactly; the paper does not specify this boundary-weight prescription.

For numerical accuracy beyond reproducing the stated mesh, increase `shells`
at fixed cutoff. Separately increase the cutoff if occupied carrier pockets or
thermal excitations approach the boundary. A finite continuum patch does not
represent the entire graphene Brillouin zone.

## Hamiltonian and basis

The sublattice basis is always `(A1, B1, A2, B2)`.
Momentums `k_Ainv` are relative to K or K', in inverse Angstrom.
At fixed valley `tau`, the dimensionless number

\[
z_\tau(\mathbf k)=\frac{\sqrt3 a}{2}(\tau k_x+i k_y)
\]

gives `v_i*pi = t_i*z` and `v_i*pi_dagger = t_i*z.conjugate()`.
Thus no numerical value of hbar is needed in the matrix construction:

\[
h_\tau(\mathbf k)=\begin{pmatrix}
D/2&t_0z_\tau^*&-t_4z_\tau^*&-t_3z_\tau\\
t_0z_\tau&D/2+\Delta'&t_1&-t_4z_\tau^*\\
-t_4z_\tau&t_1&-D/2+\Delta'&t_0z_\tau^*\\
-t_3z_\tau^*&-t_4z_\tau&t_0z_\tau&-D/2
\end{pmatrix}.
\]

At k=0 the spectrum can be checked analytically against the sorted values
`D/2`, `-D/2`, and `delta_prime +/- sqrt(t1**2 + (D/2)**2)`.

## Density, filling, and the initial matrix

The input density is **total signed doping**, including both spins and both
valleys. Negative density means holes. Do not divide it by four before calling
`initialize`, and do not multiply the returned density by another degeneracy
factor.

One common chemical potential is determined from all energies on the grid:

\[
n_{\rm carrier}=\sum_{\tau,s,\mathbf k}w_{\mathbf k}
\left[\sum_{\nu=1}^4 f_{\tau s\nu}(\mathbf k)-2\right],
\qquad
w_{\mathbf k}=\frac{\text{quadrature area}_{\mathbf k}}{(2\pi)^2}
\times10^{16}\;\mathrm{cm}^{-2}.
\]

Here the quadrature area is in inverse Angstrom squared. The subtraction of two
counts the two filled bands per spin/valley defining neutrality in the retained
four-band space. This is density bookkeeping; the code does **not** subtract a
reference density matrix from `rho0` or add a reference Hamiltonian.

At positive temperature the solver uses Fermi–Dirac occupations
`f=1/(exp((energy-mu)/(k_B*T))+1)`, evaluated stably with `scipy.special.expit`.

At exactly zero temperature, levels are filled in global ascending energy order.
If a requested density cuts through a degenerate Fermi shell, all states in that
shell receive the same fractional occupation. This is a finite-grid ensemble
convention that matches the density without selecting an arbitrary spin or valley.
The default energy tolerance for grouping that shell is `1e-10 eV`.
Do not reconstruct the zero-temperature occupations with a strict `energy < mu`
test; use the returned `occupations`, which retains that fractional shell.
In a spectral gap, the reported chemical potential is a midgap representative.

The convention for the density matrix matches your derivation:

\[
\rho^{(0)}_{\sigma\sigma'}(\mathbf k)
=\langle c^\dagger_{\mathbf k\sigma'}c_{\mathbf k\sigma}\rangle
=\sum_\nu f_\nu(\mathbf k)
U_{\sigma\nu}(\mathbf k)U^*_{\sigma'\nu}(\mathbf k).
\]

Eigenvectors are the **columns** of `U`. Each eigenvector has norm one.
Momentum weights enter density integrals; they do not multiply the matrix
`rho0(k)` itself. A completely occupied band contributes a projector of trace one.

`rho0` is Hermitian with eigenvalues between zero and one. At T=0 it is
idempotent wherever every occupation is zero or one; a fractionally occupied
Fermi shell is an explicit exception. Finite-temperature matrices are generally
not idempotent. Eigenvector phases do not affect the constructed density matrix.

## Python usage

```python
import numpy as np
from bernal_initializer import Parameters, make_hexagonal_grid, initialize

parameters = Parameters(D_eV=0.050)
grid = make_hexagonal_grid(
    shells=24, cutoff_times_a=0.12, a_angstrom=parameters.a_angstrom
)
state = initialize(
    target_density_cm2=-5e11,
    temperature_K=0.0,
    parameters=parameters,
    grid=grid,
)

energies = state.eigenvalues_eV
U = state.eigenvectors
rho0 = state.rho0
print(state.mu_eV, state.actual_density_cm2)

# Select valley K (+1), spin up, and the grid point nearest the valley center.
ik = np.argmin(np.linalg.norm(grid.k_Ainv, axis=1))
print(energies[0, 0, ik])       # four energies, ascending order
print(U[0, 0, ik, :, 1])       # normalized eigenvector of the second band
print(rho0[0, 0, ik])          # the corresponding 4 x 4 density matrix

rho0_full = state.full_density_matrix()  # optional (Nk,16,16) representation
state.save("my_initial_state.npz")
```

| Array | Shape | Axis meanings |
|---|---|---|
| `grid.k_Ainv` | `(Nk,2)` | momentum point; x/y coordinate |
| `grid.weights_cm2` | `(Nk,)` | one-band, one-flavor density integration weight |
| `state.eigenvalues_eV` | `(2,2,Nk,4)` | valley, spin, momentum, band |
| `state.eigenvectors` | `(2,2,Nk,4,4)` | valley, spin, momentum, sublattice, band |
| `state.occupations` | `(2,2,Nk,4)` | valley, spin, momentum, band |
| `state.rho0` | `(2,2,Nk,4,4)` | valley, spin, momentum, sublattice, sublattice' |
| `state.full_density_matrix()` | `(Nk,16,16)` | momentum, combined index, combined index' |

Valley indices are `[+1,-1]`; spin indices are `[up,down]`.
The combined 16-component order is four sublattices for each block:
`(K,up)`, `(K,down)`, `(K',up)`, `(K',down)`.
The Hamiltonian here is spin independent. On the symmetric default grid the
initial carrier populations are balanced across flavors; inter-flavor coherences
are zero. No ordered phase is being assumed or solved for at this stage.

To supply a different integration grid:

```python
from bernal_initializer import MomentumGrid

# k_points has shape (Nk,2), in inverse Angstrom.
# cell_areas has shape (Nk,), in inverse Angstrom squared.
custom_grid = MomentumGrid.from_cell_areas(k_points, cell_areas)
state = initialize(-5e11, 0.0, parameters, custom_grid)
```

For a rectangular midpoint grid, each cell area is `dkx*dky`. The weights must
describe the retained integration region, not an arbitrarily normalized set.

## Verification

Run `python -m unittest -v`. The supplied checks cover:

- the analytic k=0 spectrum and time-reversal relation;
- agreement with the small-momentum limit of the lattice hopping expression;
- grid point count and exact integration of the hexagon's area;
- equal occupations of degenerate states, independent of ordering;
- the density-matrix index convention and invariance under eigenvector phases;
- target density, Hermiticity, Pauli bounds, and `[h0,rho0]=0` at T=0 and T>0;
- assembly of the 16-component matrix and saving/loading the results.

The executed default gives `mu = -24.9131001632 meV`, target density
`-5e11 cm^-2`, and eigenpair residual below `7e-16 eV` in the checked environment.
These are finite-grid results, not a claim of momentum-grid convergence.

## References

- [Aguilar-Méndez et al., model and grids, Sec. II](https://arxiv.org/abs/2505.09685).
- [Koh et al., Appendix B, density matrices and filling](https://arxiv.org/html/2306.12486v3#A2).
- [NumPy `eigh`: eigenvalues and column eigenvectors](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html).
- [PySCF SCF documentation](https://pyscf.org/user/scf.html), for the later context of self-consistent mean fields.
