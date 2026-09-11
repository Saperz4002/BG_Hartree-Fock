"""Physics and normalization checks. Run: python -m unittest -v"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bernal_initializer import (
    Parameters, MomentumGrid, density_matrix, diagnostics, fill_occupations,
    hamiltonian, initialize, make_hexagonal_grid,
)


class BernalInitializerTests(unittest.TestCase):
    def test_origin_matches_analytic_spectrum(self):
        for D in (0.0, 0.05, -0.08):
            p = Parameters(D_eV=D)
            splitting = np.hypot(p.t1_eV, D / 2)
            expected = np.sort([D / 2, -D / 2,
                                p.delta_prime_eV - splitting,
                                p.delta_prime_eV + splitting])
            for valley in (+1, -1):
                np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonian([0, 0], valley, p)),
                                           expected, atol=1e-14)

    def test_spinless_time_reversal(self):
        k = np.array([[0.017, -0.011], [-0.004, 0.029], [0, 0]])
        np.testing.assert_allclose(hamiltonian(-k, -1), hamiltonian(k, +1).conj(), atol=1e-14)

    def test_continuum_is_linearization_of_lattice_model(self):
        # Independent lattice expression: Koh et al. PRB 110, 245118, Eq. A2.
        p = Parameters()
        k = np.array([2e-5, -3e-5])
        for valley in (+1, -1):
            qx = valley * 4 * np.pi / (3 * p.a_angstrom) + k[0]
            qy = k[1]
            f = (np.exp(1j * qy * p.a_angstrom / np.sqrt(3))
                 + 2 * np.exp(-1j * qy * p.a_angstrom / (2 * np.sqrt(3)))
                 * np.cos(qx * p.a_angstrom / 2))
            tb = np.array([
                [p.D_eV/2, -p.t0_eV*f, p.t4_eV*f, p.t3_eV*f.conjugate()],
                [-p.t0_eV*f.conjugate(), p.D_eV/2+p.delta_prime_eV, p.t1_eV, p.t4_eV*f],
                [p.t4_eV*f.conjugate(), p.t1_eV, -p.D_eV/2+p.delta_prime_eV, -p.t0_eV*f],
                [p.t3_eV*f, p.t4_eV*f.conjugate(), -p.t0_eV*f.conjugate(), -p.D_eV/2],
            ])
            error = np.max(np.abs(tb - hamiltonian(k, valley, p)))
            self.assertLess(error, p.t0_eV * (p.a_angstrom * np.linalg.norm(k))**2)

    def test_hexagon_count_area_and_inversion(self):
        for shells in (1, 5, 24):
            grid = make_hexagonal_grid(shells=shells)
            self.assertEqual(len(grid.k_Ainv), 1 + 3 * shells * (shells + 1))
            points = {tuple(np.round(k, 13)) for k in grid.k_Ainv}
            self.assertTrue(all(tuple(np.round(-k, 13)) in points for k in grid.k_Ainv))
            radius = 0.12 / 2.46
            hexagon_area = 3 * np.sqrt(3) * radius**2 / 2
            integrated_area = grid.weights_cm2.sum() * (2 * np.pi)**2 / 1e16
            self.assertAlmostEqual(integrated_area, hexagon_area, places=15)

    def test_zero_T_weighted_degenerate_shell_and_permutation(self):
        e = np.array([-1.0, 0.0, 0.0, 2.0])
        w = np.array([1.0, 2.0, 3.0, 4.0])
        mu, f = fill_occupations(e, w, 3.5, temperature_K=0)
        self.assertEqual(mu, 0.0)
        np.testing.assert_allclose(f, [1, 0.5, 0.5, 0])
        self.assertAlmostEqual(np.dot(w, f), 3.5)
        permutation = [2, 3, 0, 1]
        _, f_permuted = fill_occupations(e[permutation], w[permutation], 3.5)
        np.testing.assert_allclose(f_permuted, f[permutation])

    def test_density_matrix_index_order_and_gauge(self):
        u = np.eye(4, dtype=complex)
        u[:2, :2] = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
        rho = density_matrix(u, [1, 0, 0, 0])
        self.assertAlmostEqual(rho[0, 1], -0.5j)
        phases = np.exp(1j * np.array([0.3, 0.7, -1.5, 2.2]))
        np.testing.assert_allclose(density_matrix(u * phases, [1, 0, 0, 0]), rho, atol=1e-14)
        # A unitary change within an equally occupied subspace changes no density.
        angle = 0.37
        rotation = np.eye(4)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]]
        np.testing.assert_allclose(density_matrix(u @ rotation, [0.4, 0.4, 0, 0]),
                                   density_matrix(u, [0.4, 0.4, 0, 0]), atol=1e-14)

    def test_global_density_at_zero_and_finite_temperature(self):
        grid = make_hexagonal_grid(shells=8)
        for temperature in (0.0, 10.0, 100.0):
            for density in (-5e11, 0.0, 3e11):
                with self.subTest(T=temperature, n=density):
                    state = initialize(density, temperature, grid=grid)
                    self.assertAlmostEqual(state.actual_density_cm2, density, delta=2.0)
                    np.testing.assert_allclose(state.density_by_flavor_cm2,
                                               np.full((2, 2), density / 4), atol=1, rtol=1e-10)
                    d = diagnostics(state)
                    self.assertLess(d["max_eigenpair_residual_eV"], 1e-12)
                    self.assertLess(d["max_orthonormality_error"], 1e-12)
                    self.assertLess(d["max_rho_hermiticity_error"], 1e-12)
                    self.assertLess(d["max_commutator_residual_eV"], 1e-12)
                    self.assertGreater(d["minimum_rho_eigenvalue"], -1e-12)
                    self.assertLess(d["maximum_rho_eigenvalue"], 1 + 1e-12)
                    if temperature == 0 and density == 0:
                        self.assertLess(d["max_idempotency_defect"], 1e-12)

    def test_saved_arrays_and_full_internal_matrix(self):
        state = initialize(grid=make_hexagonal_grid(shells=3))
        full = state.full_density_matrix()
        self.assertEqual(full.shape, (37, 16, 16))
        np.testing.assert_allclose(full[:, 8:12, 8:12], state.rho0[1, 0])
        np.testing.assert_allclose(full[:, :4, 4:8], 0)
        with tempfile.TemporaryDirectory() as directory:
            saved = state.save(Path(directory) / "state.npz")
            with np.load(saved, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["rho0"], state.rho0)
                np.testing.assert_array_equal(data["eigenvectors"], state.eigenvectors)

    def test_bounds_and_custom_grid(self):
        custom = MomentumGrid.from_cell_areas([[0, 0], [0.01, 0]], [1e-5, 2e-5])
        state = initialize(0, grid=custom)
        self.assertEqual(state.rho0.shape, (2, 2, 2, 4, 4))
        with self.assertRaises(ValueError):
            initialize(-1e20, grid=custom)
        with self.assertRaises(ValueError):
            initialize(temperature_K=-1, grid=custom)
        with self.assertRaises(ValueError):
            MomentumGrid([[0, 0]], [-1])
        for target, expected in ((0.0, [0, 0]), (2.0, [1, 1])):
            for temperature in (0, 30):
                _, f = fill_occupations([-1, 1], [1, 1], target, temperature)
                np.testing.assert_array_equal(f, expected)


if __name__ == "__main__":
    unittest.main()
