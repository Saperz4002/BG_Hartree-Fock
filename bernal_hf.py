"""Four-band, spin/valley-diagonal Hartree--Fock for Bernal bilayer graphene.

This implements the cycle written in the accompanying notebook. It retains
four orbitals per flavor, with no band projection, SOC, Hund term, IVC, or
momentum-off-diagonal order. D is an imposed layer ENERGY bias.

Notation: P = rho, P[a,b] = <c_b^dagger c_a>; h_ref = Sigma[P_ref]/2.
Arrays P, h0, h_HF: (valley=2, spin=2, Nk, sublattice=4, sublattice'=4).
Units: eV, Angstrom, K; user densities cm^-2; energy densities eV/Angstrom^2.

Sources:
  Koh et al., PRB 110, 245118 (2024), Appendix A, arXiv:2407.09612.
  Koh et al., PRB 109, 035113 (2024), Appendix B, arXiv:2306.12486.
The initializer supplies the already-validated band Hamiltonian and filling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.constants import elementary_charge, epsilon_0
from scipy.special import xlogy
from threadpoolctl import threadpool_limits

from bernal_initializer import (
    KB_EV_PER_K, Parameters, MomentumGrid, VALLEYS,
    density_matrix, fill_occupations, hamiltonian,
)

EV_A2_TO_MEV_NM2 = 1e5  # 1 eV / A^2 = 100000 meV / nm^2
CM2_TO_A2 = 1e-16       # 1 cm^-2 = 1e-16 A^-2
COULOMB_EV_ANGSTROM = elementary_charge / (4 * np.pi * epsilon_0) * 1e10


@dataclass(frozen=True)
class InteractionParameters:
    epsilon_r: float = 20.0
    d_sc_nm: float = 30.0
    strength: float = 1.0  # 0 is the noninteracting verification limit

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Interaction parameters must be finite.")
        if self.epsilon_r <= 0 or self.d_sc_nm <= 0 or self.strength < 0:
            raise ValueError("epsilon_r, d_sc_nm must be > 0; strength >= 0.")


@dataclass(frozen=True)
class SCFSettings:
    target_density_cm2: float = -5e11
    temperature_K: float = 0.0
    alpha: float = 0.25
    epsilon_P: float = 1e-6
    epsilon_E_eV_A2: float = 1e-11
    max_iterations: int = 500
    degeneracy_tolerance_eV: float = 1e-10

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("SCF settings must be finite.")
        if self.temperature_K < 0 or not 0 < self.alpha <= 1:
            raise ValueError("Require T >= 0 and 0 < alpha <= 1.")
        if self.epsilon_P <= 0 or self.epsilon_E_eV_A2 <= 0:
            raise ValueError("Convergence tolerances must be positive.")
        if self.degeneracy_tolerance_eV < 0:
            raise ValueError("Degeneracy tolerance must be nonnegative.")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 2:
            raise ValueError("max_iterations must be an integer >= 2.")


def screened_coulomb(q_Ainv, interaction: InteractionParameters):
    """V(q) in eV A^2, including the analytic q=0 limit.

    V = [2*pi*(e^2/(4*pi*eps0))/epsilon_r] * d * tanh(q*d)/(q*d).
    Never drop the diagonal q=0 kernel element.
    """
    q = np.asarray(q_Ainv, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("q must be finite and nonnegative, in inverse A.")
    d_A = 10.0 * interaction.d_sc_nm
    z = q * d_A
    ratio = np.ones_like(z)
    np.divide(np.tanh(z), z, out=ratio, where=z != 0)
    return (interaction.strength * 2 * np.pi * COULOMB_EV_ANGSTROM
            / interaction.epsilon_r * d_A * ratio)


def full_internal_matrix(blocks):
    """(2,2,Nk,4,4) -> (Nk,16,16), with zero flavor coherences."""
    blocks = np.asarray(blocks)
    if blocks.ndim != 5 or blocks.shape[:2] != (2, 2) or blocks.shape[-2:] != (4, 4):
        raise ValueError("Expected (2,2,Nk,4,4) blocks.")
    full = np.zeros((blocks.shape[2], 16, 16), dtype=blocks.dtype)
    for iv in range(2):
        for isp in range(2):
            start = 4 * (2 * iv + isp)
            full[:, start:start + 4, start:start + 4] = blocks[iv, isp]
    return full


@dataclass
class MapOutput:
    h_HF: np.ndarray
    eigenvalues_eV: np.ndarray
    eigenvectors: np.ndarray
    occupations: np.ndarray
    mu_eV: float
    P_out: np.ndarray


class HFProblem:
    """Fixed mesh, interaction kernel, h0, and neutral reference for one run.

    This is a dense O(Nk^2) exchange quadrature, NOT an FFT or a projected
    band model. A single weighted real kernel is shared by all four flavors.
    Exchange acts within each flavor block; Hartree depends on the total trace.
    """

    def __init__(self, parameters: Parameters, grid: MomentumGrid,
                 interaction: InteractionParameters | None = None):
        self.parameters = parameters
        self.grid = grid
        self.interaction = interaction or InteractionParameters()
        self.weights_A2 = grid.weights_cm2 * CM2_TO_A2
        self.neutral_density_cm2 = float(8 * np.sum(grid.weights_cm2))
        self.V0_eV_A2 = float(screened_coulomb(0.0, self.interaction))
        q = np.linalg.norm(grid.k_Ainv[:, None, :] - grid.k_Ainv[None, :, :], axis=-1)
        # Only one factor of quadrature weight: the summed/source momentum.
        self.kernel_eV = screened_coulomb(q, self.interaction) * self.weights_A2[None, :]
        self.h0 = np.repeat(np.stack([
            hamiltonian(grid.k_Ainv, valley, parameters) for valley in VALLEYS
        ])[:, None], 2, axis=1)
        neutral = SCFSettings(target_density_cm2=0.0, temperature_K=0.0)
        self.P_ref = self.fill_hamiltonian(self.h0, neutral).P_out
        self.h_ref = 0.5 * self.self_energy(self.P_ref)
        for array in (self.h0, self.P_ref, self.h_ref, self.weights_A2, self.kernel_eV):
            array.setflags(write=False)

    def _check_shape(self, P):
        P = np.asarray(P, dtype=complex)
        if P.shape != self.h0.shape or not np.all(np.isfinite(P)):
            raise ValueError(f"P must be finite with shape {self.h0.shape}.")
        return P

    def occupied_density_A2(self, P):
        trace = np.trace(P, axis1=-2, axis2=-1).real
        return float(np.sum(trace * self.weights_A2[None, None, :]))

    def density_by_flavor_cm2(self, P):
        """Signed carrier densities, relative to two bands per flavor."""
        trace = np.trace(P, axis1=-2, axis2=-1).real
        return np.sum((trace - 2) * self.grid.weights_cm2[None, None, :], axis=-1)

    def carrier_density_cm2(self, P):
        return float(self.density_by_flavor_cm2(P).sum())

    def sigma_H(self, P):
        """Uniform Hartree matrix, in eV; uses absolute occupied density."""
        shift = self.V0_eV_A2 * self.occupied_density_A2(P)
        return np.broadcast_to(shift * np.eye(4), self.h0.shape).copy()

    def sigma_F(self, P):
        """Fock matrix -sum_k' w_k' V(k-k') P(k'), in eV."""
        # Treat every flavor and matrix entry as a column for the same kernel.
        data = np.ascontiguousarray(np.moveaxis(P, 2, 0)).reshape(len(self.weights_A2), -1)
        # Real BLAS avoids repeatedly converting the real kernel to complex.
        exchange = -(self.kernel_eV @ data.real + 1j * (self.kernel_eV @ data.imag))
        return np.moveaxis(exchange.reshape((-1, 2, 2, 4, 4)), 0, 2)

    def self_energy(self, P):
        P = self._check_shape(P)
        return self.sigma_H(P) + self.sigma_F(P)

    def build_h_HF(self, P):
        return self.h0 - self.h_ref + self.self_energy(P)

    def build_h_HF_at(self, k_Ainv, P):
        """Evaluate h_HF on a plotting cut using the original integration mesh.

        Recomputes the exchange quadrature at the requested external momenta;
        it does not interpolate bands or fill states on the plotting cut.
        P - P_ref/2 is used ONLY as the linear argument of the self-energy.
        The physical occupation matrix remains P.
        """
        P = self._check_shape(P)
        k = np.asarray(k_Ainv, dtype=float)
        if k.ndim != 2 or k.shape[1] != 2 or not np.all(np.isfinite(k)):
            raise ValueError("k_Ainv must have finite shape (Nplot,2).")
        q = np.linalg.norm(k[:, None, :] - self.grid.k_Ainv[None, :, :], axis=-1)
        kernel = screened_coulomb(q, self.interaction) * self.weights_A2[None, :]
        effective_P = P - 0.5 * self.P_ref
        data = np.ascontiguousarray(np.moveaxis(effective_P, 2, 0)).reshape(len(self.weights_A2), -1)
        exchange = -(kernel @ data.real + 1j * (kernel @ data.imag))
        exchange = np.moveaxis(exchange.reshape((-1, 2, 2, 4, 4)), 0, 2)
        h0 = np.repeat(np.stack([
            hamiltonian(k, valley, self.parameters) for valley in VALLEYS
        ])[:, None], 2, axis=1)
        shift = self.V0_eV_A2 * self.occupied_density_A2(effective_P)
        return h0 + shift * np.eye(4) + exchange

    def fill_hamiltonian(self, h, settings: SCFSettings):
        """Global filling across momenta, valleys, spins, and all four bands."""
        energies, u = np.linalg.eigh(h)
        mu, f = fill_occupations(
            energies, self.grid.weights_cm2[None, None, :, None],
            self.neutral_density_cm2 + settings.target_density_cm2,
            settings.temperature_K, settings.degeneracy_tolerance_eV,
        )
        return MapOutput(h, energies, u, f, mu, density_matrix(u, f))

    def scf_map(self, P_in, settings: SCFSettings):
        return self.fill_hamiltonian(self.build_h_HF(P_in), settings)

    def energy_density(self, P, Sigma=None):
        """E[P]/area in eV/A^2; derivative with respect to P is h_HF[P].

        The fixed one-body reference has NO extra factor 1/2 in the energy.
        Supply Sigma only if it was evaluated at this very same P.
        """
        if Sigma is None:
            Sigma = self.self_energy(P)
        one_body = self.h0 - self.h_ref + 0.5 * Sigma
        trace = np.einsum("...ab,...ba->...", one_body, P).real
        return float(np.sum(trace * self.weights_A2[None, None, :]))

    def entropy_density(self, P):
        """Gaussian fermion entropy per area, in eV/(K A^2)."""
        p = np.linalg.eigvalsh(P)
        if p.min() < -1e-8 or p.max() > 1 + 1e-8:
            raise ValueError("P has unphysical occupation eigenvalues.")
        p = np.clip(p, 0, 1)  # Only removes floating-point endpoint errors.
        s = -(xlogy(p, p) + xlogy(1 - p, 1 - p)).sum(axis=-1)
        return float(KB_EV_PER_K * np.sum(s * self.weights_A2[None, None, :]))

    def diagnostics(self, P, settings: SCFSettings, mapped: MapOutput | None = None):
        mapped = mapped or self.scf_map(P, settings)
        pe = np.linalg.eigvalsh(P)
        commutator = mapped.h_HF @ P - P @ mapped.h_HF
        return {
            "unmixed_residual": float(np.max(np.abs(mapped.P_out - P))),
            "commutator_eV": float(np.max(np.abs(commutator))),
            "hermiticity_error": float(np.max(np.abs(P - P.conj().swapaxes(-1, -2)))),
            "minimum_occupation": float(pe.min()),
            "maximum_occupation": float(pe.max()),
            "density_error_cm2": self.carrier_density_cm2(P) - settings.target_density_cm2,
            "idempotency_defect": float(np.max(np.abs(P @ P - P))),
        }

    def validate_density(self, P, target_density_cm2):
        P = self._check_shape(P)
        if np.max(np.abs(P - P.conj().swapaxes(-1, -2))) > 1e-9:
            raise ValueError("P must be Hermitian.")
        pe = np.linalg.eigvalsh(P)
        if pe.min() < -1e-9 or pe.max() > 1 + 1e-9:
            raise ValueError("P eigenvalues must lie in [0,1].")
        tolerance = max(1.0, 1e-10 * self.neutral_density_cm2)
        if abs(self.carrier_density_cm2(P) - target_density_cm2) > tolerance:
            raise ValueError("Initial P does not have the target carrier density.")

    def initial_state(self, settings: SCFSettings, seed="symmetric", seed_meV=1.0,
                      random_seed=0):
        """Seed h0, then fill. Seeds are absent from all subsequent HF steps.

        'spin', 'valley', 'spin_valley', and 'layer' are diagonal operators;
        'random' is a reproducible small Hermitian perturbation in each block.
        No seed introduces spin- or valley-off-diagonal matrix entries.
        """
        if not np.isfinite(seed_meV) or seed_meV < 0:
            raise ValueError("seed_meV must be finite and nonnegative.")
        M = np.zeros_like(self.h0)
        if seed in ("spin", "valley", "spin_valley", "layer"):
            for iv, tau in enumerate(VALLEYS):
                for isp, spin in enumerate((1, -1)):
                    if seed == "layer":
                        M[iv, isp] = np.diag([1, 1, -1, -1])
                    else:
                        sign = {"spin": spin, "valley": tau,
                                "spin_valley": tau * spin}[seed]
                        M[iv, isp] = sign * np.eye(4)
        elif seed == "random":
            rng = np.random.default_rng(random_seed)
            Z = rng.standard_normal(self.h0.shape) + 1j * rng.standard_normal(self.h0.shape)
            M = 0.5 * (Z + Z.conj().swapaxes(-1, -2))
            M /= np.max(np.abs(np.linalg.eigvalsh(M)))
        elif seed != "symmetric":
            raise ValueError("Unknown seed: " + seed)
        return self.fill_hamiltonian(self.h0 + seed_meV * 1e-3 * M, settings).P_out


@dataclass
class SCFResult:
    P: np.ndarray
    P_initial: np.ndarray
    mapped: MapOutput
    settings: SCFSettings
    seed: str
    converged: bool
    history: list[dict]
    diagnostics: dict
    energy_eV_A2: float
    entropy_eV_K_A2: float
    free_energy_eV_A2: float
    elapsed_s: float

    def summary(self, problem: HFProblem):
        return {
            "seed": self.seed, "converged": self.converged,
            "iterations": len(self.history), "elapsed_s": self.elapsed_s,
            "mu_eV": float(self.mapped.mu_eV) if np.isfinite(self.mapped.mu_eV) else str(self.mapped.mu_eV),
            "carrier_density_cm2": problem.carrier_density_cm2(self.P),
            "density_by_flavor_cm2": problem.density_by_flavor_cm2(self.P).tolist(),
            "energy_eV_A2": self.energy_eV_A2,
            "entropy_eV_K_A2": self.entropy_eV_K_A2,
            "free_energy_eV_A2": self.free_energy_eV_A2,
            "diagnostics": self.diagnostics,
        }

    def save(self, path, problem: HFProblem):
        """Portable arrays + metadata, loadable with allow_pickle=False."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.summary(problem), "settings": asdict(self.settings),
            "parameters": asdict(problem.parameters),
            "interaction": asdict(problem.interaction),
            "reference": "h_ref = 0.5 * Sigma[P_ref]; neutral T=0 noninteracting P_ref",
            "array_order": ["valley", "spin", "k", "sublattice", "sublattice"],
            "flavors": ["K up", "K down", "K' up", "K' down"],
            "sublattices": ["A1", "B1", "A2", "B2"],
        }
        np.savez_compressed(
            path, P=self.P, P_initial=self.P_initial, P_ref=problem.P_ref,
            h_ref=problem.h_ref, h_HF=self.mapped.h_HF,
            eigenvalues_eV=self.mapped.eigenvalues_eV,
            eigenvectors=self.mapped.eigenvectors, occupations=self.mapped.occupations,
            P_out=self.mapped.P_out, k_Ainv=problem.grid.k_Ainv,
            weights_cm2=problem.grid.weights_cm2,
            metadata_json=json.dumps(metadata, allow_nan=False),
            history_json=json.dumps(self.history, allow_nan=False),
        )
        return path


@threadpool_limits.wrap(limits=1, user_api="blas")
def run_scf(problem: HFProblem, settings: SCFSettings | None = None,
            seed="symmetric", seed_meV=1.0, random_seed=0, P_initial=None,
            callback: Callable[[dict], None] | None = None):
    """Damped fixed-point iteration with output-energy and true residual gates.

    At T>0 the energy stopping test is supplemented by the same tolerance on
    free energy. A converged output is checked again with its OWN Hamiltonian.
    An exhausted iteration budget is returned as converged=False, never ranked.
    One BLAS thread avoids overhead from many small matrix operations.
    """
    settings = settings or SCFSettings()
    start = perf_counter()
    P_in = (problem.initial_state(settings, seed, seed_meV, random_seed)
            if P_initial is None else np.array(P_initial, dtype=complex, copy=True))
    problem.validate_density(P_in, settings.target_density_cm2)
    P_initial_copy = P_in.copy()
    previous_E = previous_F = None
    history = []
    converged = False
    mapped_final = None
    for j in range(settings.max_iterations):
        mapped = problem.scf_map(P_in, settings)
        P_out = mapped.P_out
        Sigma_out = problem.self_energy(P_out)
        E = problem.energy_density(P_out, Sigma_out)
        S = problem.entropy_density(P_out) if settings.temperature_K > 0 else 0.0
        F = E - settings.temperature_K * S
        residual = float(np.max(np.abs(P_out - P_in)))
        delta_E = None if previous_E is None else abs(E - previous_E)
        delta_F = None if previous_F is None else abs(F - previous_F)
        row = {
            "j": j, "residual": residual, "energy_eV_A2": E,
            "free_energy_eV_A2": F, "delta_E_eV_A2": delta_E,
            "delta_F_eV_A2": delta_F,
            "mu_eV": float(mapped.mu_eV) if np.isfinite(mapped.mu_eV) else None,
            "density_error_cm2": problem.carrier_density_cm2(P_out) - settings.target_density_cm2,
        }
        history.append(row)
        if callback is not None:
            callback(row.copy())
        if (previous_E is not None and residual < settings.epsilon_P
                and delta_E < settings.epsilon_E_eV_A2
                and delta_F < settings.epsilon_E_eV_A2):
            mapped_final = problem.fill_hamiltonian(problem.h0 - problem.h_ref + Sigma_out, settings)
            check = problem.diagnostics(P_out, settings, mapped_final)
            if check["unmixed_residual"] < settings.epsilon_P:
                converged = True
                break
            mapped_final = None
        previous_E, previous_F = E, F
        P_in = (1 - settings.alpha) * P_in + settings.alpha * P_out
    if mapped_final is None:
        mapped_final = problem.scf_map(P_out, settings)
    check = problem.diagnostics(P_out, settings, mapped_final)
    problem.validate_density(P_out, settings.target_density_cm2)
    # Fractional Fermi-shell occupations can have ensemble entropy even at
    # T=0. Report that entropy honestly; it contributes zero to F at T=0.
    if settings.temperature_K == 0:
        S = problem.entropy_density(P_out)
    return SCFResult(P_out, P_initial_copy, mapped_final, settings, seed, converged,
                     history, check, E, S, F, perf_counter() - start)


def best_converged(results):
    """Lowest F among converged runs at identical settings (F=E at T=0).

    Caller must supply candidates from the SAME HFProblem (including its grid
    and reference). This is selection among candidates, not a global proof.
    """
    valid = [r for r in results if r.converged]
    if not valid:
        return None
    if any((r.settings.target_density_cm2, r.settings.temperature_K) !=
           (valid[0].settings.target_density_cm2, valid[0].settings.temperature_K) for r in valid):
        raise ValueError("Only compare candidates at identical density and temperature.")
    return min(valid, key=lambda r: r.free_energy_eV_A2)
