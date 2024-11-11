import llvmlite.binding as ll
import numba

from llvmlite import ir
from numba.core import cgutils
from numba.experimental.function_type import _get_wrapper_address
from numba.extending import intrinsic

from numba_linking.make_llvm import double_t


add_sig = numba.float64(numba.float64, numba.float64)


@numba.njit(add_sig, cache=True)
def add_prototype(x, y):
    return x + y


add_p = _get_wrapper_address(add_prototype, add_sig)
# add_p = get_dy_calc_p()

ll.add_symbol("add", add_p)


@intrinsic
def _add(typingctx, x_t, y_t):
    sig = numba.types.float64(x_t, y_t)

    def codegen(context, builder, signature, args):
        add_t = ir.FunctionType(double_t, (double_t, double_t))
        add_ = cgutils.get_or_insert_function(builder.module, add_t, "add")
        return builder.call(add_, args)
    return sig, codegen


@numba.njit(add_sig)
def add(x, y):
    return _add(x, y)


@numba.njit(add_sig)
def run(x, y):
    return 3.14 * add(x, y)


add_prototype_llvm_ref = r"""; ModuleID = 'add_prototype'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.2.0"

@.const.add_prototype = internal constant [14 x i8] c"add_prototype\00"
@_ZN08NumbaEnv8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd" = internal constant [107 x i8] c"missing Environment: _ZN08NumbaEnv8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd\00"
@PyExc_RuntimeError = external global i8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define i32 @_ZN8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, double %arg.x, double %arg.y) local_unnamed_addr #0 {
entry:
  %.6 = fadd double %arg.x, %arg.y
  store double %.6, double* %retptr, align 8
  ret i32 0
}

define i8* @_ZN7cpython8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = alloca i8*, align 8
  %.7 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.const.add_prototype, i64 0, i64 0), i64 2, i64 2, i8** nonnull %.5, i8** nonnull %.6)
  %.8 = icmp eq i32 %.7, 0
  %.39 = alloca double, align 8
  br i1 %.8, label %common.ret, label %entry.endif, !prof !0

common.ret:                                       ; preds = %entry.endif.endif.endif, %entry.endif.endif, %entry, %entry.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ %.58, %entry.endif.endif.endif.endif ], [ null, %entry ], [ null, %entry.endif.endif ], [ null, %entry.endif.endif.endif ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.12 = load i8*, i8** @_ZN08NumbaEnv8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd, align 8
  %.17 = icmp eq i8* %.12, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !0

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([107 x i8], [107 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.21 = load i8*, i8** %.5, align 8
  %.22 = call i8* @PyNumber_Float(i8* %.21)
  %.23 = call double @PyFloat_AsDouble(i8* %.22)
  call void @Py_DecRef(i8* %.22)
  %.25 = call i8* @PyErr_Occurred()
  %.26.not = icmp eq i8* %.25, null
  br i1 %.26.not, label %entry.endif.endif.endif, label %common.ret, !prof !1

entry.endif.endif.endif:                          ; preds = %entry.endif.endif
  %.30 = load i8*, i8** %.6, align 8
  %.31 = call i8* @PyNumber_Float(i8* %.30)
  %.32 = call double @PyFloat_AsDouble(i8* %.31)
  call void @Py_DecRef(i8* %.31)
  %.34 = call i8* @PyErr_Occurred()
  %.35.not = icmp eq i8* %.34, null
  br i1 %.35.not, label %entry.endif.endif.endif.endif, label %common.ret, !prof !1

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif.endif
  store double 0.000000e+00, double* %.39, align 8
  %.43 = call i32 @_ZN8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.39, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.23, double %.32) #1
  %.53 = load double, double* %.39, align 8
  %.58 = call i8* @PyFloat_FromDouble(double %.53)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i8* @PyNumber_Float(i8*) local_unnamed_addr

declare double @PyFloat_AsDouble(i8*) local_unnamed_addr

declare void @Py_DecRef(i8*) local_unnamed_addr

declare i8* @PyErr_Occurred() local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define double @cfunc._ZN8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double %.1, double %.2) local_unnamed_addr #0 {
entry:
  %.4 = alloca double, align 8
  store double 0.000000e+00, double* %.4, align 8
  %.8 = call i32 @_ZN8__main__13add_prototypeB2v1B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.4, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.1, double %.2) #1
  %.18 = load double, double* %.4, align 8
  ret double %.18
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn writeonly }
attributes #1 = { noinline }

!0 = !{!"branch_weights", i32 1, i32 99}
!1 = !{!"branch_weights", i32 99, i32 1}
"""  # noqa: E501


add_llvm_ref = r"""; ModuleID = 'add'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.2.0"

@.const.add = internal constant [4 x i8] c"add\00"
@_ZN08NumbaEnv8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd" = internal constant [96 x i8] c"missing Environment: _ZN08NumbaEnv8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd\00"
@PyExc_RuntimeError = external global i8

define i32 @_ZN8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, double %arg.x, double %arg.y) local_unnamed_addr {
entry:
  %.6 = tail call double @add(double %arg.x, double %arg.y)
  store double %.6, double* %retptr, align 8
  ret i32 0
}

declare double @add(double, double) local_unnamed_addr

define i8* @_ZN7cpython8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = alloca i8*, align 8
  %.7 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.const.add, i64 0, i64 0), i64 2, i64 2, i8** nonnull %.5, i8** nonnull %.6)
  %.8 = icmp eq i32 %.7, 0
  %.39 = alloca double, align 8
  br i1 %.8, label %common.ret, label %entry.endif, !prof !0

common.ret:                                       ; preds = %entry.endif.endif.endif, %entry.endif.endif, %entry, %entry.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ %.58, %entry.endif.endif.endif.endif ], [ null, %entry ], [ null, %entry.endif.endif ], [ null, %entry.endif.endif.endif ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.12 = load i8*, i8** @_ZN08NumbaEnv8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd, align 8
  %.17 = icmp eq i8* %.12, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !0

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([96 x i8], [96 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.21 = load i8*, i8** %.5, align 8
  %.22 = call i8* @PyNumber_Float(i8* %.21)
  %.23 = call double @PyFloat_AsDouble(i8* %.22)
  call void @Py_DecRef(i8* %.22)
  %.25 = call i8* @PyErr_Occurred()
  %.26.not = icmp eq i8* %.25, null
  br i1 %.26.not, label %entry.endif.endif.endif, label %common.ret, !prof !1

entry.endif.endif.endif:                          ; preds = %entry.endif.endif
  %.30 = load i8*, i8** %.6, align 8
  %.31 = call i8* @PyNumber_Float(i8* %.30)
  %.32 = call double @PyFloat_AsDouble(i8* %.31)
  call void @Py_DecRef(i8* %.31)
  %.34 = call i8* @PyErr_Occurred()
  %.35.not = icmp eq i8* %.34, null
  br i1 %.35.not, label %entry.endif.endif.endif.endif, label %common.ret, !prof !1

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif.endif
  store double 0.000000e+00, double* %.39, align 8
  %.43 = call i32 @_ZN8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.39, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.23, double %.32) #0
  %.53 = load double, double* %.39, align 8
  %.58 = call i8* @PyFloat_FromDouble(double %.53)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i8* @PyNumber_Float(i8*) local_unnamed_addr

declare double @PyFloat_AsDouble(i8*) local_unnamed_addr

declare void @Py_DecRef(i8*) local_unnamed_addr

declare i8* @PyErr_Occurred() local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

define double @cfunc._ZN8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double %.1, double %.2) local_unnamed_addr {
entry:
  %.4 = alloca double, align 8
  store double 0.000000e+00, double* %.4, align 8
  %.8 = call i32 @_ZN8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.4, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.1, double %.2) #0
  %.18 = load double, double* %.4, align 8
  ret double %.18
}

attributes #0 = { noinline }

!0 = !{!"branch_weights", i32 1, i32 99}
!1 = !{!"branch_weights", i32 99, i32 1}
"""  # noqa: E501


run_llvm_ref = r"""; ModuleID = 'run'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.2.0"

@.const.run = internal constant [4 x i8] c"run\00"
@_ZN08NumbaEnv8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd" = internal constant [96 x i8] c"missing Environment: _ZN08NumbaEnv8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd\00"
@_ZN08NumbaEnv8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd = common local_unnamed_addr global i8* null
@PyExc_RuntimeError = external global i8

define i32 @_ZN8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, double %arg.x, double %arg.y) local_unnamed_addr {
common.ret:
  %.6.i = tail call double @add(double %arg.x, double %arg.y), !noalias !0
  %.33 = fmul double %.6.i, 3.140000e+00
  store double %.33, double* %retptr, align 8
  ret i32 0
}

define i8* @_ZN7cpython8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = alloca i8*, align 8
  %.7 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.const.run, i64 0, i64 0), i64 2, i64 2, i8** nonnull %.5, i8** nonnull %.6)
  %.8 = icmp eq i32 %.7, 0
  %.39 = alloca double, align 8
  br i1 %.8, label %common.ret, label %entry.endif, !prof !3

common.ret:                                       ; preds = %entry.endif.endif.endif, %entry.endif.endif, %entry, %entry.endif.endif.endif.endif, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ %.58, %entry.endif.endif.endif.endif ], [ null, %entry ], [ null, %entry.endif.endif ], [ null, %entry.endif.endif.endif ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.12 = load i8*, i8** @_ZN08NumbaEnv8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd, align 8
  %.17 = icmp eq i8* %.12, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !3

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([96 x i8], [96 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.21 = load i8*, i8** %.5, align 8
  %.22 = call i8* @PyNumber_Float(i8* %.21)
  %.23 = call double @PyFloat_AsDouble(i8* %.22)
  call void @Py_DecRef(i8* %.22)
  %.25 = call i8* @PyErr_Occurred()
  %.26.not = icmp eq i8* %.25, null
  br i1 %.26.not, label %entry.endif.endif.endif, label %common.ret, !prof !4

entry.endif.endif.endif:                          ; preds = %entry.endif.endif
  %.30 = load i8*, i8** %.6, align 8
  %.31 = call i8* @PyNumber_Float(i8* %.30)
  %.32 = call double @PyFloat_AsDouble(i8* %.31)
  call void @Py_DecRef(i8* %.31)
  %.34 = call i8* @PyErr_Occurred()
  %.35.not = icmp eq i8* %.34, null
  br i1 %.35.not, label %entry.endif.endif.endif.endif, label %common.ret, !prof !4

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif.endif
  store double 0.000000e+00, double* %.39, align 8
  %.43 = call i32 @_ZN8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.39, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.23, double %.32) #0
  %.53 = load double, double* %.39, align 8
  %.58 = call i8* @PyFloat_FromDouble(double %.53)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i8* @PyNumber_Float(i8*) local_unnamed_addr

declare double @PyFloat_AsDouble(i8*) local_unnamed_addr

declare void @Py_DecRef(i8*) local_unnamed_addr

declare i8* @PyErr_Occurred() local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

define double @cfunc._ZN8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double %.1, double %.2) local_unnamed_addr {
entry:
  %.4 = alloca double, align 8
  store double 0.000000e+00, double* %.4, align 8
  %.8 = call i32 @_ZN8__main__3runB2v3B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.4, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.1, double %.2) #0
  %.18 = load double, double* %.4, align 8
  ret double %.18
}

declare double @add(double, double) local_unnamed_addr

attributes #0 = { noinline }

!0 = !{!1}
!1 = distinct !{!1, !2, !"_ZN8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd: %retptr"}
!2 = distinct !{!2, !"_ZN8__main__3addB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd"}
!3 = !{!"branch_weights", i32 1, i32 99}
!4 = !{!"branch_weights", i32 99, i32 1}
"""  # noqa: E501


if __name__ == '__main__':
    add_prototype_llvm = next(iter(add_prototype.inspect_llvm().values()))
    assert add_prototype_llvm == add_prototype_llvm_ref
    add_llvm = next(iter(add.inspect_llvm().values()))
    assert add_llvm == add_llvm_ref
    run_llvm = next(iter(run.inspect_llvm().values()))
    assert run_llvm == run_llvm_ref
