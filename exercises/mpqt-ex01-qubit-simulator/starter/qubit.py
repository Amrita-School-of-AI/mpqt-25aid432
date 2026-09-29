"""
qubit.py: a small state-vector simulator for one and two qubits.

Exercise mpqt-ex01. Fill in every constant marked None and every function
that raises NotImplementedError. Rename the finished file to <ROLLNO>.py
before you submit it.

Conventions used everywhere in this file
----------------------------------------
* A ket is a 1-D NumPy array of complex numbers, shape (n,).
  |0> is array([1, 0]) and |1> is array([0, 1]).
* A basis is a list of kets that are orthonormal.
* A gate or an observable is a 2-D NumPy array, shape (n, n).
* Two-qubit kets are ordered |00>, |01>, |10>, |11>, and the first
  (leftmost) symbol is the first qubit.
* Random numbers come only from the Generator passed in as `rng`, so the
  same seed always gives the same run.

Rules: NumPy and the Python standard maths modules only. No qiskit, no
other quantum library. In Python the imaginary unit is written 1j.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Given: the computational basis and the 2 x 2 identity.
KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)
I2 = np.eye(2, dtype=complex)

# Yours: write each of these from its definition in the notes, as a complex
# NumPy array. Kets are 1-D, gates are 2-D.
PLUS = None      # |+> = (|0> + |1>) / sqrt(2)
MINUS = None     # |-> = (|0> - |1>) / sqrt(2)

X = None         # Pauli X, the bit flip
Y = None         # Pauli Y
Z = None         # Pauli Z, the phase flip
H = None         # Hadamard
S = None         # phase gate, diag(1, i)

CNOT = None      # 4 x 4, first qubit is the control


# ---------------------------------------------------------------------------
# Stage 1: states and normalisation
# ---------------------------------------------------------------------------

def ket(*amplitudes) -> np.ndarray:
    """Build a ket from its amplitudes, as a complex 1-D array.

    Example: ket(1, 1j) -> array([1.+0.j, 0.+1.j])
    """
    raise NotImplementedError


def is_normalised(psi: np.ndarray, tol: float = 1e-9) -> bool:
    """True if the norm of psi is 1, to within tol.

    Example: is_normalised(ket(1, 1)) -> False
    """
    raise NotImplementedError


def normalise(psi: np.ndarray) -> np.ndarray:
    """Return psi divided by its norm. Raise ValueError for the zero vector.

    Example: normalise(ket(1, 1)) -> array([0.7071+0.j, 0.7071+0.j])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 2: inner product and Born probabilities
# ---------------------------------------------------------------------------

def inner(phi: np.ndarray, psi: np.ndarray) -> complex:
    """The inner product <phi|psi>, conjugating the FIRST argument.

    Example: inner(ket(1j, 0), ket(1, 0)) -> -1j
    """
    raise NotImplementedError


def probabilities(psi: np.ndarray, basis: list) -> np.ndarray:
    """Born-rule probabilities p_k = |<b_k|psi>|^2, one per basis ket.

    Return a NumPy array of real floats. Raise ValueError if psi is not
    normalised.
    Example: probabilities(PLUS, [KET0, KET1]) -> array([0.5, 0.5])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 3: gates, unitarity and Hermiticity
# ---------------------------------------------------------------------------

def is_unitary(U: np.ndarray, tol: float = 1e-9) -> bool:
    """True if U is square and U^dagger U = I, entry by entry to within tol.

    A matrix that is not square is not unitary: return False, do not crash.
    Example: is_unitary(H) -> True
    """
    raise NotImplementedError


def is_hermitian(A: np.ndarray, tol: float = 1e-9) -> bool:
    """True if A is square and A^dagger = A, entry by entry to within tol.

    A matrix that is not square is not Hermitian: return False, do not crash.
    Example: is_hermitian(Y) -> True
    """
    raise NotImplementedError


def apply(U: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Return the ket U|psi>. Raise ValueError if the sizes do not match.

    Example: apply(X, KET0) -> array([0.+0.j, 1.+0.j])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 4: measurement and collapse
# ---------------------------------------------------------------------------

def measure(psi: np.ndarray, basis: list, rng: np.random.Generator):
    """Measure psi in the given orthonormal basis.

    Return (k, post) where k is the outcome index, drawn with the Born
    probabilities, and post is basis[k], the state after collapse.
    Use rng for the random draw and nothing else.
    Example: measure(KET1, [KET0, KET1], np.random.default_rng(1)) -> (1, array([0.+0.j, 1.+0.j]))
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 5: tensor products
# ---------------------------------------------------------------------------

def tensor(*items) -> np.ndarray:
    """Kronecker product of kets or of matrices, taken left to right.

    Example: tensor(KET0, KET1) -> array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stage 6: expectation values
# ---------------------------------------------------------------------------

def expectation(A: np.ndarray, psi: np.ndarray) -> float:
    """The expectation value <psi|A|psi> of a Hermitian A, as a real float.

    Raise ValueError if A is not Hermitian.
    Example: expectation(Z, PLUS) -> 0.0
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Try things out here. Code under this line does not run when the grader
    # imports your file, so your Part C experiments can live here too.
    print(KET0, KET1)
