import inspect
import numba
from numba_linking.bind_jit import bind_jit, BIND_JIT_SFX


sig = numba.float64(numba.float64)


def aux_0_(x):
    return 2 * x


@numba.extending.overload(aux_0_, jit_options=dict(cache=True))
def aux_0(x):
    return 2 * x


code_txt_template = """
def aux_{i}(x):
    return 7.{i} * aux_{j}(x)
"""


ns_glob = {'numba': numba, 'bind_jit': bind_jit, 'aux_0': aux_0}
ns = ns_glob


def run(jit_wrap):
    for i_ in range(1, 100):
        j_ = i_ - 1
        code_txt = code_txt_template.format(i=i_, j=j_)
        code_obj = compile(code_txt, __file__, mode='exec')
        exec(code_obj, ns)
        this_module_name = inspect.getmodule(aux_0).__name__
        setattr(ns[f'aux_{i_}'], '__module__', this_module_name)
        aux_ = jit_wrap(ns[f'aux_{i_}'])
        ns[f'aux_{i_}'] = aux_


def _test_nested_bind_jit():
    """ Notice stable .nbc file size for all aux_nn """
    run(bind_jit(sig, cache=True))
    aux_99_llvm = next(iter(ns['aux_99'].inspect_llvm().values()))
    assert f'declare double @aux_99{BIND_JIT_SFX}(double)' in aux_99_llvm
    assert f'7.1' not in aux_99_llvm
    assert f'7.99' not in aux_99_llvm


def test_nested_njit():
    """ Notice growing .nbc file size for higher aux_nn """
    run(numba.extending.register_jitable(sig, cache=True))
    # aux_99_llvm = next(iter(ns['aux_99'].inspect_llvm().values()))
    # globals()['ns'] = ns_glob
    # assert '7.1' in aux_99_llvm
    # assert '7.99' in aux_99_llvm
