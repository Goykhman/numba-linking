import numba

from numba_linking.bind_jit import bind_jit


calculate_sig = numba.float64(numba.float64, numba.float64)


egg = 2.172


@numba.njit(calculate_sig, cache=True)
def calculate(x, y):
    return x + y + egg


@numba.njit(calculate_sig)
def run(x, y):
    return 3.14 * calculate(x, y)


@bind_jit(calculate_sig, cache=True)
def calculate2(x, y):
    return x + y + egg


@numba.njit(calculate_sig)
def run2(x, y):
    return 3.14 * calculate2(x, y)


@bind_jit(calculate_sig, cache=True)
@numba.njit
def calculate3(x, y):
    return x + y + egg


@numba.njit(calculate_sig)
def run3(x, y):
    return 3.14 * calculate3(x, y)


def test_njit():
    """ Default behavior, `calculate` llvm is pasted and/or inlined into `run` """
    x1 = 4.5
    x2 = 1.2
    assert abs(run(x1, x2) - 3.14 * (x1 + x2 + egg)) < 1e-15
    run_llvm = next(iter(run.inspect_llvm().values()))
    assert str(egg) in run_llvm


def test_bind_jit():
    """ Test that `bind_jit` does not paste and/or inline `calculate2` into `run2` """
    x1 = 4.5
    x2 = 1.2
    assert abs(run2(x1, x2) - 3.14 * (x1 + x2 + egg)) < 1e-15
    run2_llvm = next(iter(run2.inspect_llvm().values()))
    assert str(egg) not in run2_llvm


def test_bind_jit_of_njit():
    x1 = 4.5
    x2 = 1.2
    assert abs(run3(x1, x2) - 3.14 * (x1 + x2 + egg)) < 1e-15


if __name__ == '__main__':
    test_njit()
    test_bind_jit()
    test_bind_jit_of_njit()
