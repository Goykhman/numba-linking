import inspect
import llvmlite.binding as ll
import numba
import random
import string
import types
import typing
from llvmlite import ir
from numba.core import cgutils
from numba.extending import intrinsic
from numba.experimental.function_type import _get_wrapper_address


_ = ir, intrinsic

PY_SFX = '_py'
SIG_SFX = '_sig'
JIT_OPTS_SFX = '_jit_options'
JIT_SFX = '_jit'

random_name_substr_len = 20


def random_string(n):
    return ''.join(random.choices(string.ascii_letters, k=n))


class FuncData(typing.NamedTuple):
    func_name: str
    func_args_str: str
    func_p: int
    func_py: types.FunctionType
    ns: dict


def check_and_populate_ns(name, obj, ns):
    look_up = ns.get(name)
    if look_up is not None:
        assert look_up == obj, f"Conflict: adding {obj} as {name}, found existing {look_up}"
    ns[name] = obj


def populate_ns(func_py, func_name, sig, jit_options):
    module = inspect.getmodule(func_py)
    ns = module.__dict__
    check_and_populate_ns(f'{func_name}{SIG_SFX}', sig, ns)
    check_and_populate_ns(f'{func_name}{JIT_OPTS_SFX}', jit_options, ns)
    check_and_populate_ns(f'{func_name}{PY_SFX}', func_py, ns)
    check_and_populate_ns('numba', numba, ns)
    return ns


def extract_py_func(func):
    if isinstance(func, types.FunctionType):
        return func
    elif isinstance(func, numba.core.registry.CPUDispatcher):
        return func.py_func
    elif isinstance(func, numba.core.ccallback.CFunc):
        return func._pyfunc
    else:
        raise ValueError(f"Unsupported {func} of type {type(func)}")


def get_func_data(func, sig, jit_options=None):
    jit_options = {} if jit_options is None else jit_options
    func_py = extract_py_func(func)
    func_name = f"{func_py.__name__}{random_string(random_name_substr_len)}"
    func_args = inspect.getfullargspec(func_py).args
    func_args_str = ', '.join(func_args)
    func_jit_str = f"{func_name}{JIT_SFX} = numba.njit({func_name}{SIG_SFX}, **{func_name}{JIT_OPTS_SFX})({func_name}{PY_SFX})"  # noqa: E501
    func_jit_code = compile(func_jit_str, inspect.getfile(func_py), mode='exec')
    ns = populate_ns(func_py, func_name, sig, jit_options)
    exec(func_jit_code, ns)
    func_p = _get_wrapper_address(ns[f'{func_name}{JIT_SFX}'], sig)
    return FuncData(func_name, func_args_str, func_p, func_py, ns)


def populate_ns_imports(ns: typing.Dict):
    ns['intrinsic'] = intrinsic
    ns['ir'] = ir
    ns['cgutils'] = cgutils


func_sfx = '__'


code_str_template = f"""
@intrinsic
def _{{func_name}}(typingctx, {{func_args_str}}):
    sig = {{func_name}}{SIG_SFX}
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


def make_code_str(func_name, func_args_str):
    return code_str_template.format(
        func_name=func_name, func_args_str=func_args_str
    )


def bind_jit(sig, **jit_options):
    if not isinstance(sig, numba.core.typing.templates.Signature):
        raise ValueError(f"Expected signature, got {sig}")

    def wrap(func):
        func_data = get_func_data(func, sig, jit_options)
        ll.add_symbol(func_data.func_name, func_data.func_p)
        populate_ns_imports(func_data.ns)
        code_str = make_code_str(func_data.func_name, func_data.func_args_str)
        code_obj = compile(code_str, inspect.getfile(func_data.func_py), mode='exec')
        exec(code_obj, func_data.ns)
        func_wrap = func_data.ns[f"{func_data.func_name}{func_sfx}"]
        return func_wrap
    return wrap
