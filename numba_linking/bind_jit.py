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
    import numba
    ns = {}
    ns['numba'] = numba
    if not isinstance(sig, numba.core.typing.templates.Signature):
        raise ValueError(f"Expected signature, got {sig}")
    def wrap(func):
        if isinstance(func, numba.core.registry.CPUDispatcher):
            func_sigs = func.nopython_signatures
            if not len(func_sigs) == 1 or not func_sigs[0] == sig:
                raise ValueError(f"Incompatible signatures {func_sigs} and {sig}")
            func_jit = func
        elif isinstance(func, types.FunctionType):
            func_jit = numba.njit(**jit_options)(func)
        else:
            raise ValueError(f"Unsupported {func} of type {type(func)}")
        func_p = _get_wrapper_address(func_jit, sig)
        func_name = f"{func.__name__}"
        ll.add_symbol(func_name, func_p)
        func_args = inspect.getfullargspec(func).args
        func_args_str = ', '.join(func_args)

        ret_type = repr(sig.return_type)
        arg_types = [repr(arg) for arg in sig.args]
        for ty in chain([ret_type], arg_types):
            if ty in globals():
                ns[ty] = globals()[ty]
            elif hasattr(numba, ty):
                ns[ty] = getattr(numba, ty)
            else:
                raise RuntimeError(f"Undefined type {ty}")
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
        from llvmlite import ir
        from numba.core import cgutils
        from numba.extending import intrinsic
        ns['ir'] = ir
        ns['intrinsic'] = intrinsic
        ns['cgutils'] = cgutils
        exec(code_obj, ns)
        func_wrap = ns[f"{func_name}__"]
        globals()[f"_{func_name}"] = ns[f"_{func_name}"]
        return func_wrap
    return wrap


calculate_sig = numba.float64(numba.float64, numba.float64)


@bind_jit(calculate_sig)
def calculate(x, y):
    return x + y


@numba.njit(calculate_sig)
def run(x, y):
    return 3.14 * calculate(x, y)


if __name__ == '__main__':
    x1 = 4.5
    x2 = 1.2
    assert abs(run(x1, x2) - 3.14 * (x1 + x2)) < 1e-15
