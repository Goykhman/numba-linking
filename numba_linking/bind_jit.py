import inspect
import typing

import llvmlite.binding as ll
import numba
import random
import string
import types
from llvmlite import ir
from numba.core import cgutils
from numba.extending import intrinsic
from numba.experimental.function_type import _get_wrapper_address


_ = ir, intrinsic


random_name_substr_len = 20


def random_string(n):
    return ''.join(random.choices(string.ascii_letters, k=n))


class FuncData(typing.NamedTuple):
    func_name: str
    func_args_str: str
    func_p: int
    func_py: types.FunctionType
    ns: dict


def get_func_data(func, sig, jit_options=None):
    if jit_options is None:
        jit_options = {}
    if isinstance(func, numba.core.registry.CPUDispatcher):
        func_py = func.py_func
    elif isinstance(func, numba.core.ccallback.CFunc):
        func_py = func._pyfunc
    elif isinstance(func, types.FunctionType):
        func_py = func
    else:
        raise ValueError(f"Unsupported {func} of type {type(func)}")
    func_name = f"{func_py.__name__}{random_string(random_name_substr_len)}"
    func_args = inspect.getfullargspec(func_py).args
    func_args_str = ', '.join(func_args)
    func_jit_str = f"{func_name}_jit = numba.njit({func_name}_sig, **{func_name}_jit_options)({func_name}_py)"
    func_jit_code = compile(func_jit_str, inspect.getfile(func_py), mode='exec')
    module = inspect.getmodule(func_py)
    ns = module.__dict__
    ns[f'{func_name}_sig'] = sig
    ns[f'{func_name}_jit_options'] = jit_options
    ns[f'{func_name}_py'] = func_py
    exec(func_jit_code, ns)
    func_p = _get_wrapper_address(ns[f'{func_name}_jit'], sig)
    return FuncData(func_name, func_args_str, func_p, func_py, ns)


def repr_of_type(ty, ns):
    ty_name = repr(ty)
    if hasattr(numba, ty_name):
        ns[ty_name] = ty
        return ty_name
    elif isinstance(ty, numba.core.types.StructRef):
        for k, v in ns.items():
            if ty == v:
                return repr(k).strip("'").strip('"')
    else:
        raise RuntimeError(f"Unknown type {ty}")


def populate_ns(ns: typing.Dict):
    ns['intrinsic'] = intrinsic
    ns['ir'] = ir
    ns['cgutils'] = cgutils


func_sfx = '__'


code_str_template = f"""
@intrinsic
def _{{func_name}}(typingctx, {{func_args_str}}):
    sig = {{sig_str}}
    def codegen(context, builder, signature, args):
        func_t = ir.FunctionType(
            context.get_value_type(sig.return_type),
            [context.get_value_type(arg) for arg in sig.args]
        )
        {{func_name}}_ = cgutils.get_or_insert_function(builder.module, func_t, "{{func_name}}")
        return builder.call({{func_name}}_, args)
    return sig, codegen

@numba.njit
def {{func_name}}{func_sfx}({{func_args_str}}):
    return _{{func_name}}({{func_args_str}})
"""


def make_code_str(func_name, func_args_str, sig_str):
    return code_str_template.format(
        func_name=func_name, func_args_str=func_args_str, sig_str=sig_str
    )


def bind_jit(sig, **jit_options):
    if not isinstance(sig, numba.core.typing.templates.Signature):
        raise ValueError(f"Expected signature, got {sig}")

    def wrap(func):
        func_data = get_func_data(func, sig, jit_options)
        ll.add_symbol(func_data.func_name, func_data.func_p)
        populate_ns(func_data.ns)
        ret_type = repr_of_type(sig.return_type, func_data.ns)
        arg_types = [repr_of_type(arg, func_data.ns) for arg in sig.args]
        sig_str = f"{ret_type}({', '.join(arg_types)})"
        code_str = make_code_str(func_data.func_name, func_data.func_args_str, sig_str)
        code_obj = compile(code_str, inspect.getfile(func_data.func_py), mode='exec')
        exec(code_obj, func_data.ns)
        func_wrap = func_data.ns[f"{func_data.func_name}{func_sfx}"]
        return func_wrap
    return wrap
