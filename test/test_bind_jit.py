import ctypes
import types

import numba
from collections import namedtuple
from numba_linking.bind_jit import bind_jit, get_func_data, make_code_str
from test.aux_structrefs import S1, S1Type


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


def _assert_sub_dict(smaller, larger):
    for k, v in smaller.items():
        assert larger[k] == v


sig = numba.int32(numba.int32, numba.int32)


def aux_1(x, y):
    return x + y


def test_py_func():
    func_data = get_func_data(aux_1, sig)
    _asserts(func_data, aux_1)


@numba.njit
def aux_2(x, y):
    return x + y


def test_jit_func_2():
    func_data = get_func_data(aux_2, sig)
    _asserts(func_data, aux_2, assert_py_func=False)


@numba.njit(sig)
def aux_3(x, y):
    return x + y


def test_jit_func_3():
    func_data = get_func_data(aux_3, sig)
    _asserts(func_data, aux_3, assert_py_func=False)


@numba.njit([numba.float64(numba.float64, numba.int64), sig])
def aux_4(x, y):
    return x + y


def test_jit_func_4():
    func_data = get_func_data(aux_4, sig)
    _asserts(func_data, aux_4, assert_py_func=False)


@numba.njit([numba.float64(numba.float64, numba.int64), numba.int8(numba.int8, numba.int8)])
def aux_5(x, y):
    return x + y


def test_jit_func_5():
    func_data = get_func_data(aux_5, sig)
    _asserts(func_data, aux_5, assert_py_func=False)


@numba.cfunc(numba.float64(numba.float64, numba.int64))
def aux_6(x, y):
    return x + y


def test_jit_func_6():
    func_data = get_func_data(aux_6, sig)
    _asserts(func_data, aux_6, assert_py_func=False)


@numba.cfunc(sig)
def aux_7(x, y):
    return x + y


def test_jit_func_7():
    func_data = get_func_data(aux_7, sig)
    _asserts(func_data, aux_7, assert_py_func=False)


def _asserts(func_data, func, assert_py_func=True):
    _assert_sub_dict(func_data.ns, globals())
    assert func.__name__ in func_data.func_name
    if assert_py_func:
        assert func_data.func_py == func
    assert func_data.func_args_str == 'x, y'
    aux_ = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(func_data.func_p)
    x1, x2 = 137, 141
    assert aux_(x1, x2) == x1 + x2


code_str_ref = """
@intrinsic
def _calculation(typingctx, x, y):
    sig = calculation_sig
    def codegen(context, builder, signature, args):
        func_t = ir.FunctionType(
            context.get_value_type(sig.return_type),
            [context.get_value_type(arg) for arg in sig.args]
        )
        calculation_ = cgutils.get_or_insert_function(builder.module, func_t, "calculation")
        return builder.call(calculation_, args)
    return sig, codegen

@numba.njit(calculation_sig, **calculation_jit_options)
def calculation__(x, y):
    return _calculation(x, y)
"""


def test_make_code_str():
    code_str = make_code_str("calculation", "x, y")
    assert code_str == code_str_ref


aux_8_sig = numba.float64(S1Type)


@bind_jit(aux_8_sig)
def aux_8(s):
    return s.x1 + s.x2 + s.x3 + egg


@numba.njit(aux_8_sig)
def run8(s):
    return 3.14 * aux_8(s)


def test_jit_func_8():
    s1 = S1(1, 1, 1)
    _ = run8(s1)
    run_8_llvm = next(iter(aux_8.inspect_llvm().values()))
    assert str(egg) not in run_8_llvm


a_tuple = namedtuple('a_tuple', ['x', 'y'])
a_tup_1 = a_tuple(3.14, 137)
a_tuple_ty = numba.typeof(a_tup_1)

aux_9_sig = numba.float64(a_tuple_ty)


@bind_jit(aux_9_sig)
def aux_9(t):
    return t.x + 2.1728 * t.y


def test_namedtuple():
    assert "declare double @aux" in next(iter(aux_9.inspect_llvm().values()))


@bind_jit(numba.float64(), inline='never', cache=True)
def aux_10():
    return egg


def test_global_ns():
    func_name = "aux_10"
    from numba_linking.bind_jit import BIND_JIT_SFX, func_sfx, JIT_OPTS_SFX, JIT_SFX, PY_SFX, SIG_SFX
    func_name_sfx = f"{func_name}{BIND_JIT_SFX}"
    sig_ = f"{func_name_sfx}{SIG_SFX}"
    jit_options_ = f"{func_name_sfx}{JIT_OPTS_SFX}"
    py_func_ = f"{func_name_sfx}{PY_SFX}"
    jit_func_ = f"{func_name_sfx}{JIT_SFX}"
    intrinsic_ = f"_{func_name_sfx}"
    declare_func_ = f"{func_name_sfx}{func_sfx}"
    assert aux_10 is globals()[declare_func_]
    assert globals()[sig_] == numba.float64()
    assert globals()[jit_options_] == {'inline': 'never', 'cache': True}
    assert isinstance(globals()[py_func_], types.FunctionType)
    assert isinstance(globals()[jit_func_], numba.core.registry.CPUDispatcher)
    assert isinstance((globals()[intrinsic_]), numba.core.extending._Intrinsic)
    assert isinstance(globals()[declare_func_], numba.core.registry.CPUDispatcher)
    assert globals()[jit_func_].py_func == globals()[py_func_]
    declare_llvm = next(iter(globals()[declare_func_].inspect_llvm().values()))
    assert f"ModuleID = '{declare_func_}'" in declare_llvm
    assert f"declare double @{func_name_sfx}() local_unnamed_addr" in declare_llvm
    assert str(egg) not in declare_llvm
    define_llvm = next(iter(globals()[jit_func_].inspect_llvm().values()))
    assert f"ModuleID = '{func_name}'" in define_llvm
    assert str(egg) in define_llvm


if __name__ == '__main__':
    test_njit()
    test_bind_jit()
    test_bind_jit_of_njit()

    test_py_func()
    test_jit_func_8()
    test_jit_func_8()
    test_jit_func_8()
    test_jit_func_8()
    test_jit_func_8()
    test_jit_func_8()
    test_jit_func_8()

    test_make_code_str()

    test_namedtuple()
    test_global_ns()
