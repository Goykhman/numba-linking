import inspect
import typing

import llvmlite.binding as ll
import numba
import random
import string
import types
from itertools import chain
from llvmlite import ir
from numba.core import cgutils
from numba.extending import intrinsic
from numba.experimental.function_type import _get_wrapper_address


_ = ir, intrinsic


def random_string(n):
    return ''.join(random.choices(string.ascii_letters, k=n))


class FuncData(typing.NamedTuple):
    func_name: str
    func_args_str: str
    func_p: int
    func_py: types.FunctionType


def get_func_data(func, sig, jit_options):
    if isinstance(func, numba.core.registry.CPUDispatcher):
        func_py = func.py_func
    elif isinstance(func, numba.core.ccallback.CFunc):
        func_py = func._pyfunc
    elif isinstance(func, types.FunctionType):
        func_py = func
    else:
        raise ValueError(f"Unsupported {func} of type {type(func)}")
    func_name = f"{func_py.__name__}{random_string(20)}"
    func_args = inspect.getfullargspec(func_py).args
    func_args_str = ', '.join(func_args)
    func_jit_str = f"{func_name}_jit = numba.njit({func_name}_sig, **{func_name}_jit_options)({func_name}_py)"
    func_jit_code = compile(func_jit_str, inspect.getfile(func_py), mode='exec')
    globals()[f'{func_name}_sig'] = sig
    globals()[f'{func_name}_jit_options'] = jit_options
    globals()[f'{func_name}_py'] = func_py
    exec(func_jit_code, globals())
    func_p = _get_wrapper_address(globals()[f'{func_name}_jit'], sig)
    return FuncData(func_name, func_args_str, func_p, func_py)


def bind_jit(sig, **jit_options):
    if not isinstance(sig, numba.core.typing.templates.Signature):
        raise ValueError(f"Expected signature, got {sig}")

    def wrap(func):
        func_data = get_func_data(func, sig, jit_options)
        ll.add_symbol(func_data.func_name, func_data.func_p)
        ret_type = repr(sig.return_type)
        arg_types = [repr(arg) for arg in sig.args]
        for ty in chain([ret_type], arg_types):
            if ty not in globals():
                if hasattr(numba, ty):
                    globals()[ty] = getattr(numba, ty)
                else:
                    raise RuntimeError(f"Undefined type")
        sig_str = f"{ret_type}({', '.join(arg_types)})"
        code_str = f"""
@intrinsic
def _{func_data.func_name}(typingctx, {func_data.func_args_str}):
    sig = {sig_str}
    def codegen(context, builder, signature, args):
        func_t = ir.FunctionType(
            context.get_value_type(sig.return_type),
            [context.get_value_type(arg) for arg in sig.args]
        )
        {func_data.func_name}_ = cgutils.get_or_insert_function(builder.module, func_t, "{func_data.func_name}")
        return builder.call({func_data.func_name}_, args)
    return sig, codegen

@numba.njit
def {func_data.func_name}__({func_data.func_args_str}):
    return _{func_data.func_name}({func_data.func_args_str})
"""
        code_obj = compile(code_str, inspect.getfile(func_data.func_py), mode='exec')
        exec(code_obj, globals())
        func_wrap = globals()[f"{func_data.func_name}__"]
        return func_wrap
    return wrap
