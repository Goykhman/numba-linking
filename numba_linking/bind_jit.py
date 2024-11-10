import inspect
import llvmlite.binding as ll
import numba
import types
from itertools import chain
from llvmlite import ir
from numba.core import cgutils
from numba.extending import intrinsic
from numba.experimental.function_type import _get_wrapper_address


_ = ir, intrinsic


def bind_jit(sig, **jit_options):
    if not isinstance(sig, numba.core.typing.templates.Signature):
        raise ValueError(f"Expected signature, got {sig}")

    def wrap(func):
        if isinstance(func, numba.core.registry.CPUDispatcher):
            func_sigs = func.nopython_signatures
            if sig not in func_sigs:
                raise ValueError(f"Incompatible signatures {func_sigs} and {sig}")
            func_p = _get_wrapper_address(func, sig)
        elif isinstance(func, numba.core.ccallback.CFunc):
            if not func._sig == sig:
                raise ValueError(f"Incompatible signatures {func._sig} and {sig}")
            func_p = func.address
        elif isinstance(func, types.FunctionType):
            func_jit = numba.njit(**jit_options)(func)
            func_p = _get_wrapper_address(func_jit, sig)
        else:
            raise ValueError(f"Unsupported {func} of type {type(func)}")
        func_name = f"{func.__name__}"
        ll.add_symbol(func_name, func_p)
        func_args = inspect.getfullargspec(func).args
        func_args_str = ', '.join(func_args)

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
def _{func_name}(typingctx, {func_args_str}):
    sig = {sig_str}
    def codegen(context, builder, signature, args):
        func_t = ir.FunctionType(
            context.get_value_type(sig.return_type),
            [context.get_value_type(arg) for arg in sig.args]
        )
        {func_name}_ = cgutils.get_or_insert_function(builder.module, func_t, "{func_name}")
        return builder.call({func_name}_, args)
    return sig, codegen

@numba.njit
def {func_name}__({func_args_str}):
    return _{func_name}({func_args_str})
"""
        code_obj = compile(code_str, __file__, mode='exec')
        exec(code_obj, globals())
        func_wrap = globals()[f"{func_name}__"]
        return func_wrap
    return wrap
