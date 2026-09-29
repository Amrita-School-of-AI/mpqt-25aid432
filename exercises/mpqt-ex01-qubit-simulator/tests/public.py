#!/usr/bin/env python3
"""
Public tests for mpqt-ex01, the qubit simulator. Run one stage at a time:

    python3 tests/public.py qubit.py --stage states
    python3 tests/public.py qubit.py --stage probabilities|gates|measure|tensor|expectation

Exit code 0 means every test in the stage passed. Each failure prints one line
beginning "FAIL: " that says what was expected and what your code gave.

The grader runs these tests plus some hidden ones on the same functions, stage by
stage. A stage earns its marks only when every test in it passes.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print(f"FAIL: NumPy is not installed for {sys.executable}. This is a problem with "
          "the Python running the checker, not with the file being checked: "
          "install NumPy (pip install numpy) and run again.")
    sys.exit(2)

STAGES = ["states", "probabilities", "gates", "measure", "tensor", "expectation"]

# Modules a submission may import. Anything else (qiskit, scipy, sympy, ...) fails.
ALLOWED_IMPORTS = {"numpy", "math", "cmath", "__future__", "typing",
                   "functools", "itertools", "operator"}

# The checker's own copies of the standard kets and gates. Tests of your functions
# use these, so a slip in one of your constants costs only the test of that
# constant, and not every test that happens to use it.
R = 1 / np.sqrt(2)
K0 = np.array([1, 0], dtype=complex)
K1 = np.array([0, 1], dtype=complex)
KP = R * np.array([1, 1], dtype=complex)
KM = R * np.array([1, -1], dtype=complex)
KPI = R * np.array([1, 1j], dtype=complex)
KMI = R * np.array([1, -1j], dtype=complex)
I2 = np.eye(2, dtype=complex)
GX = np.array([[0, 1], [1, 0]], dtype=complex)
GY = np.array([[0, -1j], [1j, 0]], dtype=complex)
GZ = np.array([[1, 0], [0, -1]], dtype=complex)
GH = R * np.array([[1, 1], [1, -1]], dtype=complex)
GS = np.array([[1, 0], [0, 1j]], dtype=complex)
GCNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
COMP = [K0, K1]
PM = [KP, KM]

TESTS: dict[str, list] = {s: [] for s in STAGES}


def test(stage: str):
    """Register a test function under a stage. Its docstring is the 'ok' line."""
    def wrap(fn):
        TESTS[stage].append(fn)
        return fn
    return wrap


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt(a) -> str:
    """Short, readable form of a number or array for a FAIL line."""
    a = np.asarray(a, dtype=complex)
    if np.allclose(a.imag, 0):
        a = a.real
    a = np.round(a, 4) + 0          # the + 0 turns -0.0 into 0.0
    return np.array2string(a, precision=4, suppress_small=True, separator=", ",
                           max_line_width=10**6).replace("\n", "")


def close(got, want, what: str, tol: float = 1e-8) -> None:
    got_a = np.asarray(got)
    want_a = np.asarray(want)
    if got_a.shape != want_a.shape:
        raise AssertionError(f"{what}: expected shape {want_a.shape}, got shape {got_a.shape}")
    if not np.allclose(got_a, want_a, rtol=0, atol=tol):
        if want_a.size > 16:
            bad = np.argwhere(~np.isclose(got_a, want_a, rtol=0, atol=tol))[0]
            idx = tuple(int(i) for i in bad)
            raise AssertionError(f"{what}: entry {idx} should be {fmt(want_a[idx])}, "
                                 f"got {fmt(got_a[idx])}")
        raise AssertionError(f"{what}: expected {fmt(want_a)}, got {fmt(got_a)}")


def need(q, name: str):
    """Fetch a constant from the submission, with a readable message if unset."""
    if not hasattr(q, name):
        raise AssertionError(f"{name} is not defined in your file")
    val = getattr(q, name)
    if val is None:
        raise AssertionError(f"{name} is still None; write it from its definition")
    return np.asarray(val)


def raises(exc, fn, *args, what: str) -> None:
    try:
        fn(*args)
    except exc:
        return
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{what}: expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"{what}: expected {exc.__name__}, but nothing was raised")


def counts(q, psi, basis, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = np.zeros(len(basis), dtype=int)
    for _ in range(n):
        k, _post = q.measure(psi, basis, rng)
        c[int(k)] += 1
    return c


# ---------------------------------------------------------------------------
# Stage: states
# ---------------------------------------------------------------------------

@test("states")
def t_ket(q):
    """ket() builds a complex 1-D array"""
    psi = q.ket(1, 1j)
    if not isinstance(psi, np.ndarray):
        raise AssertionError(f"ket(1, 1j) should return a NumPy array, got {type(psi).__name__}")
    if not np.iscomplexobj(psi):
        raise AssertionError("ket() should return a complex array; use dtype=complex")
    close(psi, [1, 1j], "ket(1, 1j)")
    close(q.ket(2, 0), [2, 0], "ket(2, 0)")


@test("states")
def t_is_normalised(q):
    """is_normalised() accepts unit vectors and rejects others"""
    if not q.is_normalised(KP):
        raise AssertionError("is_normalised(|+>) should be True")
    if q.is_normalised(np.array([1, 1], dtype=complex)):
        raise AssertionError("is_normalised([1, 1]) should be False: its norm is sqrt(2)")
    if not q.is_normalised(np.array([0.6, 0.8j])):
        raise AssertionError("is_normalised([0.6, 0.8i]) should be True: 0.36 + 0.64 = 1")


@test("states")
def t_normalise(q):
    """normalise() divides by the norm"""
    close(q.normalise(np.array([3, 4j], dtype=complex)), [0.6, 0.8j], "normalise([3, 4i])")
    close(q.normalise(np.array([1, 1], dtype=complex)), KP, "normalise([1, 1])")


@test("states")
def t_plus_minus(q):
    """PLUS and MINUS are correct"""
    close(need(q, "PLUS"), KP, "PLUS")
    close(need(q, "MINUS"), KM, "MINUS")


# ---------------------------------------------------------------------------
# Stage: probabilities
# ---------------------------------------------------------------------------

@test("probabilities")
def t_inner_basic(q):
    """inner() gives <0|0> = 1, <0|1> = 0, <+|0> = 1/sqrt(2)"""
    close(q.inner(K0, K0), 1, "inner(|0>, |0>)")
    close(q.inner(K0, K1), 0, "inner(|0>, |1>)")
    close(q.inner(KP, K0), R, "inner(|+>, |0>)")


@test("probabilities")
def t_inner_conjugates_first(q):
    """inner() conjugates the first argument"""
    phi = np.array([1j, 0], dtype=complex)
    got = complex(q.inner(phi, K0))
    if abs(got - (-1j)) > 1e-9:
        hint = ""
        if abs(got - 1j) < 1e-9:
            hint = (" You get +i, which means the first argument was not conjugated"
                    " (or the second one was). In this course <phi|psi> conjugates"
                    " phi, the FIRST argument.")
        raise AssertionError(f"inner([i, 0], [1, 0]) should be -i, got {got:.4g}.{hint}")


@test("probabilities")
def t_probs_computational(q):
    """probabilities() in the computational basis"""
    close(q.probabilities(KP, COMP), [0.5, 0.5], "probabilities(|+>, {|0>,|1>})")
    close(q.probabilities(np.array([0.6, 0.8j]), COMP), [0.36, 0.64],
          "probabilities([0.6, 0.8i], {|0>,|1>})")


@test("probabilities")
def t_probs_pm(q):
    """probabilities() in the +/- basis"""
    close(q.probabilities(KP, PM), [1, 0], "probabilities(|+>, {|+>,|->})")
    close(q.probabilities(K0, PM), [0.5, 0.5], "probabilities(|0>, {|+>,|->})")


@test("probabilities")
def t_probs_real(q):
    """probabilities() returns real numbers"""
    p = np.asarray(q.probabilities(KPI, COMP))
    if np.iscomplexobj(p):
        raise AssertionError("probabilities() returned complex numbers. |z|^2 is real: "
                             "use abs(z)**2, not z * conj(z)")


# ---------------------------------------------------------------------------
# Stage: gates
# ---------------------------------------------------------------------------

@test("gates")
def t_gate_constants(q):
    """X, Y, Z, H and S are correct"""
    for name, want in (("X", GX), ("Y", GY), ("Z", GZ), ("H", GH), ("S", GS)):
        close(need(q, name), want, name)


@test("gates")
def t_is_unitary(q):
    """is_unitary() recognises unitary and non-unitary matrices"""
    for name, U in (("X", GX), ("Y", GY), ("H", GH), ("S", GS)):
        if not q.is_unitary(U):
            raise AssertionError(f"is_unitary({name}) should be True")
    if q.is_unitary(np.array([[1, 1], [0, 1]], dtype=complex)):
        raise AssertionError("is_unitary([[1, 1], [0, 1]]) should be False")


@test("gates")
def t_is_hermitian(q):
    """is_hermitian() recognises Hermitian and non-Hermitian matrices"""
    for name, A in (("X", GX), ("Y", GY), ("Z", GZ), ("H", GH)):
        if not q.is_hermitian(A):
            raise AssertionError(f"is_hermitian({name}) should be True")
    if q.is_hermitian(GS):
        raise AssertionError("is_hermitian(S) should be False: S^dagger has -i where S has i")


@test("gates")
def t_apply(q):
    """apply() gives X|0> = |1>, H|0> = |+>, H|1> = |->"""
    close(q.apply(GX, K0), K1, "apply(X, |0>)")
    close(q.apply(GH, K0), KP, "apply(H, |0>)")
    close(q.apply(GH, K1), KM, "apply(H, |1>)")


@test("gates")
def t_identities(q):
    """your H and Z satisfy HH = I and HZH = X"""
    h, z = need(q, "H"), need(q, "Z")
    close(h @ h, I2, "H @ H")
    close(h @ z @ h, GX, "H @ Z @ H")


# ---------------------------------------------------------------------------
# Stage: measure
# ---------------------------------------------------------------------------

@test("measure")
def t_measure_returns(q):
    """measure() returns (outcome index, basis ket)"""
    rng = np.random.default_rng(1)
    out = q.measure(KP, COMP, rng)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise AssertionError("measure() should return a tuple (k, post)")
    k, post = out
    if int(k) not in (0, 1):
        raise AssertionError(f"outcome index should be 0 or 1, got {k!r}")
    close(post, COMP[int(k)], f"post-measurement state after outcome {int(k)}")


@test("measure")
def t_measure_certain(q):
    """a certain outcome always happens"""
    c = counts(q, K1, COMP, 200, seed=5)
    if c[1] != 200:
        raise AssertionError(f"measuring |1> in {{|0>,|1>}} 200 times gave counts {c.tolist()}; "
                             "expected [0, 200]")
    c = counts(q, KP, PM, 200, seed=6)
    if c[0] != 200:
        raise AssertionError(f"measuring |+> in {{|+>,|->}} 200 times gave counts {c.tolist()}; "
                             "expected [200, 0]")


@test("measure")
def t_measure_statistics(q):
    """measure() follows the Born rule over 4000 shots"""
    c = counts(q, KP, COMP, 4000, seed=11)
    if not 1800 <= c[0] <= 2200:
        raise AssertionError(f"|+> measured 4000 times gave counts {c.tolist()}; "
                             "expected about 2000 each")
    psi = np.array([np.sqrt(0.2), np.sqrt(0.8)], dtype=complex)
    c = counts(q, psi, COMP, 4000, seed=12)
    if not 600 <= c[0] <= 1000:
        raise AssertionError(f"a state with p(0) = 0.2 measured 4000 times gave counts "
                             f"{c.tolist()}; expected about [800, 3200]")


@test("measure")
def t_measure_uses_rng(q):
    """measure() draws its randomness from rng only"""
    a = counts_seq(q, 21)
    b = counts_seq(q, 21)
    if a != b:
        raise AssertionError("two runs with the same seed gave different outcomes. Draw the "
                             "random number from the rng argument, not from np.random or random")


def counts_seq(q, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(q.measure(KP, COMP, rng)[0]) for _ in range(40)]


# ---------------------------------------------------------------------------
# Stage: tensor
# ---------------------------------------------------------------------------

@test("tensor")
def t_tensor_kets(q):
    """tensor() of two kets gives |00>, |01>, |10>, |11> in order"""
    close(q.tensor(K0, K0), [1, 0, 0, 0], "tensor(|0>, |0>)")
    close(q.tensor(K0, K1), [0, 1, 0, 0], "tensor(|0>, |1>)")
    close(q.tensor(K1, K0), [0, 0, 1, 0], "tensor(|1>, |0>)")


@test("tensor")
def t_tensor_matrices(q):
    """tensor() of two matrices"""
    want = np.kron(GX, I2)
    close(q.tensor(GX, I2), want, "tensor(X, I)")


@test("tensor")
def t_cnot(q):
    """CNOT is correct and makes the Bell state from |00>"""
    c = need(q, "CNOT")
    close(c, GCNOT, "CNOT")
    bell = c @ np.kron(GH, I2) @ np.kron(K0, K0)
    close(bell, R * np.array([1, 0, 0, 1]), "CNOT (H x I) |00>")


# ---------------------------------------------------------------------------
# Stage: expectation
# ---------------------------------------------------------------------------

@test("expectation")
def t_expectation_values(q):
    """expectation() gives <Z> and <X> for |0>, |1> and |+>"""
    close(q.expectation(GZ, K0), 1.0, "expectation(Z, |0>)")
    close(q.expectation(GZ, K1), -1.0, "expectation(Z, |1>)")
    close(q.expectation(GZ, KP), 0.0, "expectation(Z, |+>)")
    close(q.expectation(GX, KP), 1.0, "expectation(X, |+>)")


@test("expectation")
def t_expectation_float(q):
    """expectation() returns a real float"""
    v = q.expectation(GY, KPI)
    if not isinstance(v, float):
        raise AssertionError(f"expectation() should return a float, got {type(v).__name__}. "
                             "<psi|A|psi> is real for Hermitian A: return its real part "
                             "as a float")
    close(v, 1.0, "expectation(Y, |+i>)")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def check_imports(path: Path) -> list[str]:
    """Module names imported by the file that are not on the allowed list."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []          # load() reports the syntax error with a line number
    bad = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for n in names:
            if n.split(".")[0] not in ALLOWED_IMPORTS:
                bad.append(n)
    return bad


def load(path: Path):
    spec = importlib.util.spec_from_file_location("qubit_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(path: Path, stage: str) -> int:
    fails = 0

    def fail(msg: str) -> None:
        nonlocal fails
        fails += 1
        print(f"FAIL: {msg}")

    if not path.exists():
        fail(f"no such file: {path}")
        return 1

    bad = check_imports(path)
    if bad:
        fail("only NumPy and the standard maths modules may be imported; found: "
             + ", ".join(sorted(set(bad))))
        print(f"\n{fails} problem(s) in stage '{stage}'")
        return 1

    try:
        q = load(path)
    except Exception as e:  # noqa: BLE001
        fail(f"your file could not be imported: {type(e).__name__}: {e}. "
             "Put experiments under `if __name__ == \"__main__\":`")
        print(f"\n{fails} problem(s) in stage '{stage}'")
        return 1

    for fn in TESTS[stage]:
        label = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        try:
            fn(q)
        except NotImplementedError:
            fail(f"{label}: a function it needs is not implemented yet")
        except AssertionError as e:
            fail(str(e))
        except Exception as e:  # noqa: BLE001
            fail(f"{label}: raised {type(e).__name__}: {e}")
        else:
            print(f"  ok  {label}")

    print()
    if fails:
        print(f"{fails} problem(s) in stage '{stage}'")
        return 1
    print(f"stage '{stage}' passed")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--stage", required=True, choices=STAGES)
    a = ap.parse_args()
    sys.exit(run(a.source, a.stage))


if __name__ == "__main__":
    main()
