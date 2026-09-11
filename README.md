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
with the same settings. Keep
the notebook and `bernal_initializer.py` in the same directory.

NumPy's `linalg.eigh` solves the explicit Hermitian matrices; SciPy supplies the
finite-temperature number solver.

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


For numerical accuracy beyond reproducing the stated mesh, increase `shells`
at fixed cutoff. Separately increase the cutoff if occupied carrier pockets or
thermal excitations approach the boundary. A finite continuum patch does not
represent the entire graphene Brillouin zone.

## References

- [Aguilar-Méndez et al., model and grids, Sec. II](https://arxiv.org/abs/2505.09685).
- [Koh et al., Appendix B, density matrices and filling](https://arxiv.org/html/2306.12486v3#A2).
- [NumPy `eigh`: eigenvalues and column eigenvectors](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html).
