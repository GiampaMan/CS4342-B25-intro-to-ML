import os
import unittest
import numpy as np
import importlib.util
from pathlib import Path

# Allow overriding the module path via env var, default to same dir as this test.
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = THIS_DIR / "homework1_numpy_solution.py"
MODULE_PATH = Path(os.environ.get("HW_MOD_PATH", str(DEFAULT_PATH)))

spec = importlib.util.spec_from_file_location("hw", MODULE_PATH)
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)

class TestPart2Numpy(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_problem1(self):
        A = np.array([[1,2],[3,4]])
        B = np.array([[5,6],[7,8]])
        C = np.array([[1,1],[1,1]])
        expected = A @ B - C
        out = hw.problem1(A,B,C)
        np.testing.assert_allclose(out, expected)

    def test_problem2(self):
        A = np.zeros((3,4))
        out = hw.problem2(A)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_array_equal(out, np.ones(3))

    def test_problem3(self):
        A = np.array([[9,1,2],[3,8,4]])
        out = hw.problem3(A)
        self.assertEqual(out.shape, A.shape)
        self.assertEqual(out[0,0], 0)
        self.assertEqual(out[1,1], 0)
        # Off-diagonals unchanged
        self.assertEqual(out[0,1], 1)
        self.assertEqual(out[0,2], 2)
        self.assertEqual(out[1,0], 3)
        self.assertEqual(out[1,2], 4)
        # Original A not modified
        self.assertEqual(A[0,0], 9)

    def test_problem4(self):
        A = np.array([[1,2,3],[4,5,6]])
        out = hw.problem4(A, 1)
        self.assertEqual(out, 15)

    def test_problem5(self):
        A = np.array([[1,2,3],[4,5,6]], dtype=float)
        out = hw.problem5(A, 2, 5)
        self.assertAlmostEqual(out, (2+3+4+5)/4.0)
        # Empty case returns NaN
        out_empty = hw.problem5(A, 10, 11)
        self.assertTrue(np.isnan(out_empty))

    def test_problem6(self):
        # Diagonal matrix with distinct eigenvalues, easy check
        A = np.array([[2.0, 0.0],[0.0, 1.0]])
        V1 = hw.problem6(A, 1)
        self.assertEqual(V1.shape, (2,1))
        v = V1[:,0]
        # Should satisfy A v ≈ 2 v
        np.testing.assert_allclose(A @ v, 2.0*v, atol=1e-6)

        V2 = hw.problem6(A, 2)
        # First column corresponds to eigenvalue 2, second to 1 (by |eig| sort)
        np.testing.assert_allclose(A @ V2[:,0], 2.0*V2[:,0], atol=1e-6)
        np.testing.assert_allclose(A @ V2[:,1], 1.0*V2[:,1], atol=1e-6)

    def test_problem7(self):
        A = np.array([[3.0, 1.0],[2.0, 4.0]])
        x = np.array([7.0, 10.0])
        expected = np.linalg.solve(A, x)
        out = hw.problem7(A, x)
        np.testing.assert_allclose(out, expected)

    def test_problem8(self):
        x = np.array([1,2,3])
        k = 4
        out = hw.problem8(x,k)
        self.assertEqual(out.shape, (3,4))
        for j in range(k):
            np.testing.assert_array_equal(out[:,j], x)

    def test_problem9(self):
        A = np.arange(10).reshape(5,2)  # rows: [0,1], [2,3], ...
        out = hw.problem9(A)
        # Same rows, permuted order
        np.testing.assert_array_equal(np.sort(out.flatten()), np.sort(A.flatten()))
        # With seed 0 and n=5, permutation is not identity; very likely true
        self.assertFalse(np.array_equal(out, A))

    def test_problem10(self):
        A = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
        out = hw.problem10(A)
        np.testing.assert_allclose(out, np.array([2.0, 5.0]))

    def test_problem11(self):
        n, k = 10, 6
        # Generate expected using same RNG sequence
        np.random.seed(123)
        expected = np.random.randint(0, k+1, size=n)
        expected = expected.copy()
        expected[expected % 2 == 0] = -1

        np.random.seed(123)
        out = hw.problem11(n, k)
        np.testing.assert_array_equal(out, expected)

    def test_problem12(self):
        A = np.array([[1,2],[3,4],[5,6]])
        b = np.array([10,20,30])
        out = hw.problem12(A, b)
        expected = np.array([[11,12],[23,24],[35,36]])
        np.testing.assert_array_equal(out, expected)

    def test_problem13(self):
        # 3 images of 2x2
        A = np.array([
            [[1,2],[3,4]],
            [[5,6],[7,8]],
            [[9,10],[11,12]]
        ])
        out = hw.problem13(A)
        # Should be shape (4,3) with each column flattened image in row-major order
        expected = np.array([
            [1,5,9],
            [2,6,10],
            [3,7,11],
            [4,8,12]
        ])
        np.testing.assert_array_equal(out, expected)

if __name__ == "__main__":
    unittest.main()
