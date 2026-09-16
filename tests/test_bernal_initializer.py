"""Noninteracting four-band Bernal graphene: eigenpairs and rho^(0).

Source: Aguilar-Mendez, Neupert, Wagner, arXiv:2505.09685, Eq. (1), Sec. II.
This module contains no Hartree-Fock iteration and no interaction self-energy.

Units: energy eV, momentum inverse Angstrom, temperature K, density cm^-2.
Density is signed TOTAL carrier density relative to charge neutrality:
negative = holes; positive = electrons. Both valleys and spins are included.

Run the worked default:
    python bernal_initializer.py --density-cm2=-5e11 --D-meV=50 --temperature-K=0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit

KB_EV_PER_K = 8.617333262145e-5
ANGSTROM_MINUS2_TO_CM_MINUS2 = 1.0e16
VALLEYS = (+1, -1)
SPINS = ("up", "down")
SUBLATTICES = ("A1", "B1", "A2", "B2")


@dataclass(frozen=True)
class Parameters:
    """Paper hopping parameters; D_eV is a layer ENERGY bias, not V/nm.

    D=50 meV is an editable example used in the paper's line-cut discussion.
    The lattice constant a=2.46 Angstrom is the graphene Bravais lattice
    constant, not the carbon-carbon nearest-neighbor distance.
    """

    D_eV: float = 0.050
    a_angstrom: float = 2.46
    t0_eV: float = 2.61
    t1_eV: float = 0.361
    t3_eV: float = 0.283
    t4_eV: float = 0.138
    delta_prime_eV: float = 0.015

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("All model parameters must be finite.")
        if self.a_angstrom <= 0:
            raise ValueError("a_angstrom must be positive.")


@dataclass(frozen=True)
class MomentumGrid:
    """One identical valley-relative grid used in each spin/valley sector.

    k_Ainv: (Nk, 2), in inverse Angstrom, measured FROM the valley center.
    weights_cm2: (Nk,), approximates d^2k/(2*pi)^2, converted to cm^-2.
    No spin or valley multiplicity is included in these weights.
    """

    k_Ainv: NDArray
    weights_cm2: NDArray
    description: str = "User-supplied momentum quadrature"

    def __post_init__(self) -> None:
        k = np.array(self.k_Ainv, dtype=float, copy=True)
        w = np.array(self.weights_cm2, dtype=float, copy=True)
        if k.ndim != 2 or k.shape[1] != 2 or len(k) == 0:
            raise ValueError("k_Ainv must have shape (Nk, 2), with Nk > 0.")
        if w.shape != (len(k),):
            raise ValueError("weights_cm2 must have shape (Nk,).")
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(w)):
            raise ValueError("Momentum points and weights must be finite.")
        if np.any(w <= 0):
            raise ValueError("All integration weights must be positive.")
        object.__setattr__(self, "k_Ainv", k)
        object.__setattr__(self, "weights_cm2", w)

    @classmethod
    def from_cell_areas(
        cls, k_Ainv: ArrayLike, area_weights_Ainv2: ArrayLike,
        description: str = "User-supplied momentum quadrature",
    ) -> MomentumGrid:
        """Convert quadrature area weights (inverse Angstrom squared).

        For a rectangular midpoint grid, each area weight is dkx*dky.
        Supply quadrature weights, not a normalized probability distribution.
        """
        weights = (np.asarray(area_weights_Ainv2, dtype=float)
                   * ANGSTROM_MINUS2_TO_CM_MINUS2 / (2 * np.pi) ** 2)
        return cls(np.asarray(k_Ainv), weights, description)


def make_hexagonal_grid(
    shells: int = 24,
    cutoff_times_a: float = 0.12,
    a_angstrom: float = 2.46,
) -> MomentumGrid:
    """Triangular mesh inside a hexagon, including its boundary.

    Defaults reproduce the paper's translation-preserving point grid:
    1 + 3*24*25 = 1801 points, radius/side 0.12/a, spacing 0.005/a.

    Integration choice made HERE (not specified by the paper): nodal
    piecewise-linear triangular quadrature. Relative point weights are 1
    in the interior, 1/2 on straight edges, and 1/3 at the six corners.
    Their sum integrates the specified hexagon's area exactly.
    """
    if isinstance(shells, bool) or not isinstance(shells, (int, np.integer)) or shells < 1:
        raise ValueError("shells must be an integer >= 1.")
    if not np.isfinite(cutoff_times_a) or cutoff_times_a <= 0:
        raise ValueError("cutoff_times_a must be positive and finite.")
    if not np.isfinite(a_angstrom) or a_angstrom <= 0:
        raise ValueError("a_angstrom must be positive and finite.")
    step = cutoff_times_a / (a_angstrom * shells)
    points, relative_weights = [], []
    for i in range(-shells, shells + 1):
        for j in range(-shells, shells + 1):
            radius = max(abs(i), abs(j), abs(i + j))
            if radius > shells:
                continue
            points.append((step * (i + j / 2), step * np.sqrt(3) * j / 2))
            if radius < shells:
                relative_weights.append(1.0)
            elif i == 0 or j == 0 or i + j == 0:
                relative_weights.append(1.0 / 3.0)
            else:
                relative_weights.append(0.5)
    interior_cell_area = np.sqrt(3) * step**2 / 2
    areas = interior_cell_area * np.asarray(relative_weights)
    description = (f"Triangular hexagon: {shells} shells, "
                   f"cutoff*a={cutoff_times_a:g}, step*a={cutoff_times_a/shells:g}; "
                   "boundary-weighted triangular quadrature")
    return MomentumGrid.from_cell_areas(points, areas, description)


def hamiltonian(
    k_Ainv: ArrayLike, valley: int = +1, parameters: Parameters | None = None,
) -> NDArray:
    """Return h_tau(k) in basis (A1, B1, A2, B2), in eV.

    A single k=(kx,ky) gives (4,4); an (Nk,2) array gives (Nk,4,4).
    We substitute z=sqrt(3)*a*(tau*kx+i*ky)/2, so v_i*pi=t_i*z:
    the hbar factors cancel without mixing energy and momentum units.
    """
    p = parameters if parameters is not None else Parameters()
    if valley not in VALLEYS:
        raise ValueError("valley must be +1 or -1.")
    k = np.asarray(k_Ainv, dtype=float)
    if k.ndim not in (1, 2) or k.shape[-1] != 2 or not np.all(np.isfinite(k)):
        raise ValueError("k_Ainv must be a finite (2,) or (Nk,2) array.")
    z = np.sqrt(3) * p.a_angstrom / 2 * (valley * k[..., 0] + 1j * k[..., 1])
    h = np.zeros(k.shape[:-1] + (4, 4), dtype=complex)
    h[..., 0, 0] = p.D_eV / 2
    h[..., 1, 1] = p.D_eV / 2 + p.delta_prime_eV
    h[..., 2, 2] = -p.D_eV / 2 + p.delta_prime_eV
    h[..., 3, 3] = -p.D_eV / 2
    h[..., 0, 1] = p.t0_eV * z.conj()
    h[..., 0, 2] = -p.t4_eV * z.conj()
    h[..., 0, 3] = -p.t3_eV * z
    h[..., 1, 2] = p.t1_eV
    h[..., 1, 3] = -p.t4_eV * z.conj()
    h[..., 2, 3] = p.t0_eV * z.conj()
    for row, col in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        h[..., col, row] = h[..., row, col].conj()
    return h


def fill_occupations(
    energies_eV: ArrayLike,
    state_weights_cm2: ArrayLike,
    target_occupied_density_cm2: float,
    temperature_K: float = 0.0,
    degeneracy_tolerance_eV: float = 1e-10,
) -> tuple[float, NDArray]:
    """Globally fill states at a specified weighted occupation density.

    This lower-level function receives ABSOLUTE occupied density, not doping.
    initialize() performs the conversion from signed carrier density.

    At T=0, equal-energy states in the Fermi shell receive equal occupation.
    A fractional last shell matches arbitrary density on a finite grid without
    choosing an arbitrary member of a spin/valley-degenerate multiplet.
    Consequently rho need not be idempotent at a partially filled shell.
    At T>0, a single chemical potential solves the Fermi-Dirac number equation.
    """
    energies = np.asarray(energies_eV, dtype=float)
    weights = np.broadcast_to(np.asarray(state_weights_cm2, dtype=float), energies.shape)
    if energies.size == 0 or not np.all(np.isfinite(energies)):
        raise ValueError("energies_eV must be nonempty and finite.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Every state weight must be positive and finite.")
    if not np.isfinite(temperature_K) or temperature_K < 0:
        raise ValueError("temperature_K must be finite and >= 0.")
    if not np.isfinite(degeneracy_tolerance_eV) or degeneracy_tolerance_eV < 0:
        raise ValueError("degeneracy_tolerance_eV must be finite and >= 0.")
    target = float(target_occupied_density_cm2)
    capacity = float(np.sum(weights, dtype=np.longdouble))
    density_tolerance = 64 * np.finfo(float).eps * max(capacity, 1.0)
    if not np.isfinite(target) or target < -density_tolerance or target > capacity + density_tolerance:
        raise ValueError(f"Requested occupation lies outside [0, {capacity:.8g}] cm^-2.")
    target = float(np.clip(target, 0.0, capacity))
    if target == 0:
        mu = float(energies.min() - 1.0) if temperature_K == 0 else -np.inf
        return mu, np.zeros_like(energies)
    if target == capacity:
        mu = float(energies.max() + 1.0) if temperature_K == 0 else np.inf
        return mu, np.ones_like(energies)

    if temperature_K == 0:
        order = np.argsort(energies.ravel(), kind="stable")
        sorted_e = energies.ravel()[order]
        sorted_w = weights.ravel()[order]
        cumulative = np.cumsum(sorted_w, dtype=np.longdouble)
        index = min(int(np.searchsorted(cumulative, target)), len(sorted_e) - 1)
        fermi_energy = sorted_e[index]
        lo = int(np.searchsorted(sorted_e, fermi_energy - degeneracy_tolerance_eV, side="left"))
        hi = int(np.searchsorted(sorted_e, fermi_energy + degeneracy_tolerance_eV, side="right"))
        below = cumulative[lo - 1] if lo else np.longdouble(0)
        shell_weight = np.sum(sorted_w[lo:hi], dtype=np.longdouble)
        fraction = float(np.clip((np.longdouble(target) - below) / shell_weight, 0, 1))
        sorted_f = np.zeros_like(sorted_e)
        sorted_f[:lo] = 1.0
        sorted_f[lo:hi] = fraction
        flat_f = np.empty_like(sorted_f)
        flat_f[order] = sorted_f
        # In a gap, mu is not unique. Choose the midgap representative.
        mu = float(fermi_energy)
        if fraction >= 1 - 1e-13 and hi < len(sorted_e):
            mu = float((sorted_e[hi - 1] + sorted_e[hi]) / 2)
        elif fraction <= 1e-13 and lo > 0:
            mu = float((sorted_e[lo - 1] + sorted_e[lo]) / 2)
        return mu, flat_f.reshape(energies.shape)

    kBT = KB_EV_PER_K * temperature_K

    def number_error(mu: float) -> float:
        f = expit((mu - energies) / kBT)
        return float(np.sum(weights * f, dtype=np.longdouble) - target)

    padding = max(1.0, 50 * kBT)
    lower, upper = float(energies.min() - padding), float(energies.max() + padding)
    mu = brentq(number_error, lower, upper, xtol=1e-14, rtol=1e-14)
    return float(mu), expit((mu - energies) / kBT)


def density_matrix(eigenvectors: ArrayLike, occupations: ArrayLike) -> NDArray:
    """rho[a,b] = <c_b^dagger c_a> = sum_n f_n U[a,n] U[b,n]^*.

    Eigenvectors occupy COLUMNS, as returned by numpy.linalg.eigh.
    Integration weights do not multiply rho(k); they enter momentum sums.
    """
    u = np.asarray(eigenvectors, dtype=complex)
    f = np.asarray(occupations, dtype=float)
    if u.ndim < 2 or u.shape[-1] != u.shape[-2] or f.shape != u.shape[:-2] + (u.shape[-1],):
        raise ValueError("Expected U(...,a,n) and occupations(...,n) of matching size.")
    return np.einsum("...an,...n,...bn->...ab", u, f, u.conj(), optimize=True)


@dataclass
class InitialState:
    parameters: Parameters
    grid: MomentumGrid
    temperature_K: float
    target_density_cm2: float
    mu_eV: float
    eigenvalues_eV: NDArray   # (valley, spin, k, band)
    eigenvectors: NDArray    # (valley, spin, k, sublattice, band)
    occupations: NDArray     # (valley, spin, k, band)
    rho0: NDArray            # (valley, spin, k, sublattice, sublattice')

    @property
    def density_by_flavor_cm2(self) -> NDArray:
        """Signed carrier densities, shape (2 valleys, 2 spins)."""
        excess = self.occupations.sum(axis=-1) - 2.0
        return np.sum(excess * self.grid.weights_cm2[None, None, :], axis=-1)

    @property
    def actual_density_cm2(self) -> float:
        return float(self.density_by_flavor_cm2.sum())

    def full_density_matrix(self) -> NDArray:
        """Return (Nk,16,16) in (valley,spin,sublattice) block order.

        Four blocks: (+,up), (+,down), (-,up), (-,down), each in
        (A1,B1,A2,B2). Initial inter-spin/inter-valley coherences are zero.
        """
        full = np.zeros((len(self.grid.k_Ainv), 16, 16), dtype=complex)
        for iv in range(2):
            for isp in range(2):
                start = 4 * (2 * iv + isp)
                full[:, start:start + 4, start:start + 4] = self.rho0[iv, isp]
        return full

    def summary(self) -> dict:
        return {
            "parameters": asdict(self.parameters),
            "grid": self.grid.description,
            "points_per_valley": len(self.grid.k_Ainv),
            "valleys": list(VALLEYS), "spins": list(SPINS),
            "sublattices": list(SUBLATTICES),
            "temperature_K": self.temperature_K,
            "target_density_cm2": self.target_density_cm2,
            "actual_density_cm2": self.actual_density_cm2,
            "density_error_cm2": self.actual_density_cm2 - self.target_density_cm2,
            "density_by_flavor_cm2": self.density_by_flavor_cm2.tolist(),
            "mu_eV": self.mu_eV if np.isfinite(self.mu_eV) else str(self.mu_eV),
            "fractionally_occupied_states": int(np.count_nonzero(
                (self.occupations > 1e-12) & (self.occupations < 1 - 1e-12))),
            "eigenvalues_shape": list(self.eigenvalues_eV.shape),
            "eigenvectors_shape": list(self.eigenvectors.shape),
            "rho0_shape": list(self.rho0.shape),
            "density_convention": "Total electrons minus neutrality; negative for holes",
        }

    def save(self, filename: str | Path) -> Path:
        """Save portable NumPy arrays; load with np.load(..., allow_pickle=False)."""
        filename = Path(filename)
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filename,
            k_Ainv=self.grid.k_Ainv,
            weights_cm2=self.grid.weights_cm2,
            eigenvalues_eV=self.eigenvalues_eV,
            eigenvectors=self.eigenvectors,
            occupations=self.occupations,
            rho0=self.rho0,
            mu_eV=self.mu_eV,
            temperature_K=self.temperature_K,
            target_density_cm2=self.target_density_cm2,
            actual_density_cm2=self.actual_density_cm2,
            metadata_json=json.dumps(self.summary(), allow_nan=False),
        )
        return filename


def initialize(
    target_density_cm2: float = -5e11,
    temperature_K: float = 0.0,
    parameters: Parameters | None = None,
    grid: MomentumGrid | None = None,
    degeneracy_tolerance_eV: float = 1e-10,
) -> InitialState:
    """Compute all noninteracting eigenpairs, occupations, and rho^(0).

    Defaults: D=50 meV, signed carrier density -5e11 cm^-2, T=0 K.
    All four bands are kept. No Wigner folding or band projection is applied.
    The noninteracting model has no spin-orbit term and is spin independent.
    """
    p = parameters if parameters is not None else Parameters()
    mesh = grid if grid is not None else make_hexagonal_grid(a_angstrom=p.a_angstrom)
    if not np.isfinite(target_density_cm2):
        raise ValueError("target_density_cm2 must be finite.")
    # H shape: (valley=2, spin=2, Nk, sublattice=4, sublattice'=4).
    valley_h = np.stack([hamiltonian(mesh.k_Ainv, valley, p) for valley in VALLEYS])
    h0 = np.repeat(valley_h[:, None, ...], 2, axis=1)
    energies, eigenvectors = np.linalg.eigh(h0)

    # Neutrality: 2 of 4 states per spin/valley at each k in the retained space.
    neutral_density = float(8 * np.sum(mesh.weights_cm2, dtype=np.longdouble))
    target_occupied_density = neutral_density + float(target_density_cm2)
    state_weights = mesh.weights_cm2[None, None, :, None]
    mu, occupations = fill_occupations(
        energies, state_weights, target_occupied_density,
        temperature_K, degeneracy_tolerance_eV,
    )
    rho0 = density_matrix(eigenvectors, occupations)
    return InitialState(p, mesh, float(temperature_K), float(target_density_cm2),
                        mu, energies, eigenvectors, occupations, rho0)


def diagnostics(state: InitialState) -> dict:
    """Numerical consistency checks; fractional occupations need not be projectors."""
    h = np.repeat(np.stack([hamiltonian(state.grid.k_Ainv, v, state.parameters)
                           for v in VALLEYS])[:, None], 2, axis=1)
    u, e, rho = state.eigenvectors, state.eigenvalues_eV, state.rho0
    udagger = u.conj().swapaxes(-1, -2)
    rho_eigenvalues = np.linalg.eigvalsh(rho)
    return {
        "max_eigenpair_residual_eV": float(np.max(np.abs(h @ u - u * e[..., None, :]))),
        "max_orthonormality_error": float(np.max(np.abs(udagger @ u - np.eye(4)))),
        "max_rho_hermiticity_error": float(np.max(np.abs(rho - rho.conj().swapaxes(-1, -2)))),
        "max_commutator_residual_eV": float(np.max(np.abs(h @ rho - rho @ h))),
        "max_trace_occupation_error": float(np.max(np.abs(
            np.trace(rho, axis1=-2, axis2=-1) - state.occupations.sum(axis=-1)))),
        "minimum_rho_eigenvalue": float(rho_eigenvalues.min()),
        "maximum_rho_eigenvalue": float(rho_eigenvalues.max()),
        "max_idempotency_defect": float(np.max(np.abs(rho @ rho - rho))),
        "density_error_cm2": state.actual_density_cm2 - state.target_density_cm2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--D-meV", type=float, default=50.0, help="Layer energy bias, meV (not V/nm).")
    parser.add_argument("--density-cm2", type=float, default=-5e11, help="Signed TOTAL carrier density; negative for holes.")
    parser.add_argument("--temperature-K", type=float, default=0.0)
    parser.add_argument("--shells", type=int, default=24, help="24 produces 1801 points per valley.")
    parser.add_argument("--cutoff-times-a", type=float, default=0.12)
    parser.add_argument("--output", type=Path, default=Path("initial_state.npz"))
    args = parser.parse_args()
    p = Parameters(D_eV=args.D_meV * 1e-3)
    grid = make_hexagonal_grid(args.shells, args.cutoff_times_a, p.a_angstrom)
    state = initialize(args.density_cm2, args.temperature_K, p, grid)
    output = state.save(args.output)
    report = {**state.summary(), "diagnostics": diagnostics(state)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved arrays: {output}")


if __name__ == "__main__":
    main()
