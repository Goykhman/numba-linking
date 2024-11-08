import ctypes
import llvmlite.binding as ll
import llvmlite.ir as ir

from ctypes import CFUNCTYPE, c_double


double_t = ir.DoubleType()


def make_calc_module():
    """
    Return LLVM Module object that contains a function `add`
    that returns the result of adding two doubles.

    This module defines `add` function that will be referenced
    to in `run_module`. Notice the defining linkage here and
    declaration linkage in the referencing module.
    """
    calc_module = ir.Module(name="calc_module")
    f_add_type = ir.FunctionType(double_t, [double_t, double_t])
    f_add = ir.Function(calc_module, f_add_type, name="add")
    f_add.linkage = 'linkonce_odr'
    f_add_block = f_add.append_basic_block(name="f_add_block")
    builder = ir.IRBuilder(f_add_block)
    x1 = f_add.args[0]
    x2 = f_add.args[1]
    sum_ = builder.fadd(x1, x2, name="sum_")
    builder.ret(sum_)
    return calc_module


calc_module = make_calc_module()
calc_module_str = str(calc_module)
calc_module_str_ref = """; ModuleID = "calc_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define linkonce_odr double @"add"(double %".1", double %".2")
{
f_add_block:
  %"sum_" = fadd double %".1", %".2"
  ret double %"sum_"
}
"""
assert calc_module_str == calc_module_str_ref


def make_runner_module():
    """
    Return LLVM Module object that contains a function `fun`
    that returns 3 * (x1 + x2) for two arguments x1, x2.
    """
    runner_module = ir.Module(name="runner")
    f_add_type = ir.FunctionType(double_t, [double_t, double_t])
    f_add_declare = ir.Function(runner_module, f_add_type, name="add")
    f_add_declare.linkage = 'external'
    f_run_type = ir.FunctionType(double_t, [double_t, double_t])
    f_run = ir.Function(runner_module, f_run_type, name="run")
    f_run_block = f_run.append_basic_block(name="f_run_block")
    builder = ir.IRBuilder(f_run_block)
    x1 = f_run.args[0]
    x2 = f_run.args[1]
    add_x1_x2 = builder.call(f_add_declare, [x1, x2], name="add_x1_x2")
    res_ = builder.fmul(add_x1_x2, ir.Constant(double_t, 3.0), name="res_")
    builder.ret(res_)
    return runner_module


runner_module = make_runner_module()
runner_module_str = str(runner_module)
runner_module_str_ref = """; ModuleID = "runner"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare external double @"add"(double %".1", double %".2")

define double @"run"(double %".1", double %".2")
{
f_run_block:
  %"add_x1_x2" = call double @"add"(double %".1", double %".2")
  %"res_" = fmul double %"add_x1_x2", 0x4008000000000000
  ret double %"res_"
}
"""
assert runner_module_str == runner_module_str_ref


llvm_runner_static_ref = """; ModuleID = '<string>'
source_filename = "<string>"
target triple = "unknown-unknown-unknown"

define double @run(double %.1, double %.2) {
f_run_block:
  %add_x1_x2 = call double @add(double %.1, double %.2)
  %res_ = fmul double %add_x1_x2, 3.000000e+00
  ret double %res_
}

define linkonce_odr double @add(double %.1, double %.2) {
f_add_block:
  %sum_ = fadd double %.1, %.2
  ret double %sum_
}
"""


def ll_initialize():
    ll.initialize()
    ll.initialize_native_target()
    ll.initialize_native_asmprinter()


def compile_run_func_static():
    """
    Statically link calc_module and runner_module
    """
    ll_initialize()

    llvm_calc_module = ll.parse_assembly(calc_module_str)
    llvm_runner_module = ll.parse_assembly(runner_module_str)

    llvm_runner_module.link_in(llvm_calc_module)
    assert str(llvm_runner_module) == llvm_runner_static_ref

    llvm_runner_module.verify()

    target_machine = ll.Target.from_default_triple().create_target_machine()
    engine = ll.create_mcjit_compiler(llvm_runner_module, target_machine)
    engine.finalize_object()
    return engine


engine_run_static = compile_run_func_static()


def get_run_from_ptr(engine):
    run_ptr = engine.get_function_address("run")
    run_fn = CFUNCTYPE(c_double, c_double, c_double)(run_ptr)
    return run_fn


run_static_fn = get_run_from_ptr(engine_run_static)


llvm_runner_dynamic_ref = """; ModuleID = '<string>'
source_filename = "<string>"
target triple = "unknown-unknown-unknown"

declare double @add(double, double)

define double @run(double %.1, double %.2) {
f_run_block:
  %add_x1_x2 = call double @add(double %.1, double %.2)
  %res_ = fmul double %add_x1_x2, 3.000000e+00
  ret double %res_
}
"""


def get_dy_calc_p():
    libcalc = ctypes.CDLL('./libcalc.dylib')
    add_f = libcalc.add
    add_p = ctypes.cast(add_f, ctypes.c_void_p).value
    return add_p


def compile_run_func_dynamic():
    """
    Dynamically link calc_module and runner_module
    """
    add_p = get_dy_calc_p()
    ll.add_symbol("add", add_p)

    ll_initialize()

    llvm_runner_module = ll.parse_assembly(runner_module_str)
    assert str(llvm_runner_module) == llvm_runner_dynamic_ref

    llvm_runner_module.verify()

    target_machine = ll.Target.from_default_triple().create_target_machine()
    engine = ll.create_mcjit_compiler(llvm_runner_module, target_machine)
    engine.finalize_object()
    return engine


engine_run_dynamic = compile_run_func_dynamic()

run_dynamic_fn = get_run_from_ptr(engine_run_dynamic)


if __name__ == '__main__':
    x1 = 3.14
    x2 = 1.41
    assert abs(run_static_fn(x1, x2) - 3 * (x1 + x2)) < 1e-15
    assert abs(run_dynamic_fn(x1, x2) - 3 * (x1 + x2)) < 1e-15
