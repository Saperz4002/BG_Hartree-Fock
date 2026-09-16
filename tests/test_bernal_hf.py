"""Physics and consistency checks. Run: python -m unittest -v test_bernal_hf"""
import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from bernal_initializer import Parameters, make_hexagonal_grid, initialize
from bernal_hf import (
    COULOMB_EV_ANGSTROM, HFProblem, InteractionParameters, SCFSettings,
    screened_coulomb, full_internal_matrix, run_scf, best_converged,
)


class HartreeFockChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = Parameters()
        cls.grid = make_hexagonal_grid(shells=4, a_angstrom=cls.parameters.a_angstrom)
        cls.problem = HFProblem(cls.parameters, cls.grid)
        cls.settings = SCFSettings()

    def test_screened_q0_and_asymptotic_limit(self):
        v = self.problem.interaction
        expected_zero = 2 * np.pi * COULOMB_EV_ANGSTROM / v.epsilon_r * (10 * v.d_sc_nm)
        assert_allclose(screened_coulomb(0, v), expected_zero, rtol=1e-14)
        assert_allclose(screened_coulomb(1e-10, v), expected_zero, rtol=1e-12)
        assert_allclose(screened_coulomb(1.0, v), 2 * np.pi * COULOMB_EV_ANGSTROM / v.epsilon_r)

    def test_exchange_matches_explicit_weighted_sum(self):
        p = self.problem
        P = p.initial_state(self.settings, "random", seed_meV=2, random_seed=23)
        iv, isp, ik = 1, 0, 7
        manual = np.zeros((4, 4), dtype=complex)
        for jk in range(len(self.grid.k_Ainv)):
            q = np.linalg.norm(self.grid.k_Ainv[ik] - self.grid.k_Ainv[jk])
            manual -= screened_coulomb(q, p.interaction) * self.grid.weights_cm2[jk] * 1e-16 * P[iv, isp, jk]
        assert_allclose(p.sigma_F(P)[iv, isp, ik], manual, rtol=1e-13, atol=1e-14)

    def test_hartree_uses_total_absolute_density(self):
        p = self.problem
        P = p.initial_state(self.settings)
        n_abs_A2 = (p.neutral_density_cm2 + self.settings.target_density_cm2) * 1e-16
        expected = p.V0_eV_A2 * n_abs_A2 * np.eye(4)
        assert_allclose(p.sigma_H(P), np.broadcast_to(expected, P.shape), atol=1e-14)

    def test_energy_derivative_matches_hamiltonian(self):
        p = self.problem
        P = p.initial_state(self.settings, "random", random_seed=3)
        rng = np.random.default_rng(42)
        Z = rng.normal(size=P.shape) + 1j * rng.normal(size=P.shape)
        direction = 0.5 * (Z + Z.conj().swapaxes(-1, -2))
        direction /= np.max(np.abs(direction))
        step = 1e-4
        numerical = (p.energy_density(P + step * direction) - p.energy_density(P - step * direction)) / (2 * step)
        analytic = np.sum(np.einsum("...ab,...ba->...", p.build_h_HF(P), direction).real * p.weights_A2)
        assert_allclose(numerical, analytic, rtol=2e-8, atol=1e-13)

    def test_half_reference_energy_cancellation(self):
        p = self.problem
        reference_kinetic = np.sum(np.einsum("...ab,...ba->...", p.h0, p.P_ref).real * p.weights_A2)
        assert_allclose(p.energy_density(p.P_ref), reference_kinetic, rtol=1e-13)
        assert_allclose(p.build_h_HF(p.P_ref), p.h0 + 0.5 * p.self_energy(p.P_ref), atol=1e-14)

    def test_zero_interaction_recovers_original_initializer(self):
        p = HFProblem(self.parameters, self.grid, InteractionParameters(strength=0))
        original = initialize(parameters=self.parameters, grid=self.grid,
                              target_density_cm2=self.settings.target_density_cm2)
        result = run_scf(p, replace(self.settings, alpha=0.5), seed="spin", seed_meV=2)
        self.assertTrue(result.converged)
        assert_allclose(result.P, original.rho0, atol=1e-12)
        assert_allclose(result.mapped.eigenvalues_eV, original.eigenvalues_eV, atol=1e-13)
        self.assertAlmostEqual(result.mapped.mu_eV, original.mu_eV, places=12)

    def test_seeds_preserve_filling_and_physical_occupations(self):
        p = self.problem
        for seed in ("symmetric", "spin", "valley", "spin_valley", "layer", "random"):
            with self.subTest(seed=seed):
                P = p.initial_state(self.settings, seed=seed, seed_meV=2)
                p.validate_density(P, self.settings.target_density_cm2)
        spin = p.initial_state(self.settings, seed="spin", seed_meV=5)
        self.assertGreater(np.max(np.abs(spin[:, 0] - spin[:, 1])), 1e-3)

    def test_full_matrix_and_plotting_cut_consistency(self):
        p = self.problem
        P = p.initial_state(self.settings, "random")
        assert_allclose(p.build_h_HF_at(self.grid.k_Ainv, P), p.build_h_HF(P), atol=2e-14)
        full = full_internal_matrix(P)
        for iv in range(2):
            for isp in range(2):
                start = 4 * (2 * iv + isp)
                assert_allclose(full[:, start:start+4, start:start+4], P[iv, isp])
        block_eigs = np.sort(np.linalg.eigvalsh(P).transpose(2, 0, 1, 3).reshape(len(full), 16), axis=-1)
        assert_allclose(np.linalg.eigvalsh(full), block_eigs, atol=1e-13)

    def test_interacting_fixed_point_and_failure_reporting(self):
        p = self.problem
        result = run_scf(p, self.settings)
        self.assertTrue(result.converged)
        self.assertLess(result.diagnostics["unmixed_residual"], self.settings.epsilon_P)
        self.assertLess(abs(result.diagnostics["density_error_cm2"]), 1)
        assert_allclose(result.energy_eV_A2, p.energy_density(result.P), atol=1e-14)
        failed = run_scf(p, replace(self.settings, max_iterations=2))
        self.assertFalse(failed.converged)
        self.assertIsNone(best_converged([failed]))
        self.assertIs(best_converged([failed, result]), result)

if __name__ == "__main__":
    unittest.main()
