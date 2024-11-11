import numba


add_sig = numba.float64(numba.float64, numba.float64)


@numba.cfunc(add_sig, cache=True)
def add(x, y):
    return x + y + 2.17


@numba.njit(add_sig, cache=True)
def run(x, y):
    """ This will not cache, see comment in main """
    return 3.14 * add(x, y)


add_llvm_ref = r"""; ModuleID = 'add'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.2.0"

@_ZN08NumbaEnv8__main__3addB2v1B52c8tJTIeFIjxB2IKSgI4CrvQClUYkACQB1EiFSRRB9GgCAA_3d_3dEdd = common local_unnamed_addr global i8* null

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define i32 @_ZN8__main__3addB2v1B52c8tJTIeFIjxB2IKSgI4CrvQClUYkACQB1EiFSRRB9GgCAA_3d_3dEdd(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, double %arg.x, double %arg.y) local_unnamed_addr #0 {
entry:
  %.6 = fadd double %arg.x, %arg.y
  %.7 = fadd double %.6, 2.170000e+00
  store double %.7, double* %retptr, align 8
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define double @cfunc._ZN8__main__3addB2v1B52c8tJTIeFIjxB2IKSgI4CrvQClUYkACQB1EiFSRRB9GgCAA_3d_3dEdd(double %.1, double %.2) local_unnamed_addr #0 {
entry:
  %.4 = alloca double, align 8
  store double 0.000000e+00, double* %.4, align 8
  %.8 = call i32 @_ZN8__main__3addB2v1B52c8tJTIeFIjxB2IKSgI4CrvQClUYkACQB1EiFSRRB9GgCAA_3d_3dEdd(double* nonnull %.4, { i8*, i32, i8*, i8*, i32 }** nonnull undef, double %.1, double %.2) #1
  %.18 = load double, double* %.4, align 8
  ret double %.18
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn writeonly }
attributes #1 = { noinline }
"""  # noqa: E501


run_llvm_ref = r"""; ModuleID = 'run'
source_filename = "<string>"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.2.0"

@numba.dynamic.globals.103684028 = linkonce local_unnamed_addr global i8* inttoptr (i64 4352131112 to i8*)
@numba.dynamic.globals.105428dd0 = linkonce local_unnamed_addr global i8* inttoptr (i64 4383215056 to i8*)
@.const.pickledata.4395387328 = internal constant [143 x i8] c"\80\04\95\84\00\00\00\00\00\00\00\8C\08builtins\94\8C\0CRuntimeError\94\93\94\8C_missing Environment: _ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd\94\85\94N\87\94."
@.const.pickledata.4395387328.sha1 = internal constant [20 x i8] c"\\\02R\DE\C5\92D\80\93\D8H\A9mb\DB\11DA\C1R"
@.const.picklebuf.4395387328 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([143 x i8], [143 x i8]* @.const.pickledata.4395387328, i32 0, i32 0), i32 143, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4395387328.sha1, i32 0, i32 0), i8* null, i32 0 }
@.const.numba.experimental.function_type = internal constant [33 x i8] c"numba.experimental.function_type\00"
@.const._get_wrapper_address = internal constant [21 x i8] c"_get_wrapper_address\00"
@.const.pickledata.4395390016 = internal constant [262 x i8] c"\80\04\95\FB\00\00\00\00\00\00\00\8C\1Bnumba.core.typing.templates\94\8C\09Signature\94\93\94)\81\94(\8C\19numba.core.types.abstract\94\8C\13_type_reconstructor\94\93\94\8C\07copyreg\94\8C\0E_reconstructor\94\93\94\8C\18numba.core.types.scalars\94\8C\05Float\94\93\94\8C\08builtins\94\8C\06object\94\93\94N\87\94}\94(\8C\04name\94\8C\07float64\94\8C\08bitwidth\94K@\8C\05_code\94K\1Au\87\94R\94h\17h\17\86\94NNt\94b."
@.const.pickledata.4395390016.sha1 = internal constant [20 x i8] c"\8F\F2\DE\84\D9G\F6\81YS\AD\A3\A4`\F3f\D0\CD\A5\CF"
@.const.pickledata.4395206848 = internal constant [199 x i8] c"\80\04\95\BC\00\00\00\00\00\00\00\8C\08builtins\94\8C\0CRuntimeError\94\93\94\8C2float64(float64, float64) function address is null\94\85\94\8C\03run\94\8CY/Users/mikhailgoykhman/Dropbox/root/Notes/numba/numba-linking/numba_linking/make_cfunc.py\94K\0E\87\94\87\94."
@.const.pickledata.4395206848.sha1 = internal constant [20 x i8] c"\CB_\14\0Ft\C9o\08\1B\BF\05\A4\EB\84\FD\B7\1Cyg("
@.const.picklebuf.4395206848 = internal constant { i8*, i32, i8*, i8*, i32 } { i8* getelementptr inbounds ([199 x i8], [199 x i8]* @.const.pickledata.4395206848, i32 0, i32 0), i32 199, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4395206848.sha1, i32 0, i32 0), i8* null, i32 0 }
@.const.run = internal constant [4 x i8] c"run\00"
@_ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd = common local_unnamed_addr global i8* null
@".const.missing Environment: _ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd" = internal constant [96 x i8] c"missing Environment: _ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd\00"
@".const.Error creating Python tuple from runtime exception arguments" = internal constant [61 x i8] c"Error creating Python tuple from runtime exception arguments\00"
@".const.unknown error when calling native function" = internal constant [43 x i8] c"unknown error when calling native function\00"
@PyExc_RuntimeError = external global i8
@".const.Error creating Python tuple from runtime exception arguments.1" = internal constant [61 x i8] c"Error creating Python tuple from runtime exception arguments\00"
@PyExc_StopIteration = external global i8
@PyExc_SystemError = external global i8
@".const.unknown error when calling native function.2" = internal constant [43 x i8] c"unknown error when calling native function\00"
@".const.<numba.core.cpu.CPUContext object at 0x105fd4e00>" = internal constant [50 x i8] c"<numba.core.cpu.CPUContext object at 0x105fd4e00>\00"

define i32 @_ZN8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture writeonly %excinfo, double %arg.x, double %arg.y) local_unnamed_addr {
entry:
  %.33 = alloca i32, align 4
  store i32 0, i32* %.33, align 4
  %.9 = load i8*, i8** @numba.dynamic.globals.103684028, align 8
  %.12 = load i8*, i8** @numba.dynamic.globals.105428dd0, align 8
  %.17 = icmp eq i8* %.9, null
  br i1 %.17, label %B0.if, label %B0.endif, !prof !0

B0.if:                                            ; preds = %entry
  %.19 = load i8*, i8** @_ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd, align 8
  %.20 = icmp eq i8* %.19, null
  br i1 %.20, label %B0.if.if, label %B0.if.endif, !prof !0

common.ret:                                       ; preds = %B0.if.endif.if, %B0.if.if, %B0.endif
  %common.ret.op = phi i32 [ 0, %B0.endif ], [ 1, %B0.if.if ], [ 1, %B0.if.endif.if ]
  ret i32 %common.ret.op

B0.endif:                                         ; preds = %entry, %B0.if.endif.endif
  %"fptr_of_$6load_global.1.0.in" = phi i8* [ %.54, %B0.if.endif.endif ], [ %.9, %entry ]
  %"fptr_of_$6load_global.1.0" = bitcast i8* %"fptr_of_$6load_global.1.0.in" to double (double, double)*
  %.64 = call double %"fptr_of_$6load_global.1.0"(double %arg.x, double %arg.y)
  %.65 = fmul double %.64, 3.140000e+00
  store double %.65, double* %retptr, align 8
  br label %common.ret

B0.if.if:                                         ; preds = %B0.if
  store { i8*, i32, i8*, i8*, i32 }* @.const.picklebuf.4395387328, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8, !numba_exception_output !1
  br label %common.ret

B0.if.endif:                                      ; preds = %B0.if
  call void @numba_gil_ensure(i32* nonnull %.33)
  %.36 = call i8* @PyImport_ImportModuleNoBlock(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.const.numba.experimental.function_type, i64 0, i64 0))
  %.37 = call i8* @PyObject_GetAttrString(i8* %.36, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.const._get_wrapper_address, i64 0, i64 0))
  call void @Py_DecRef(i8* %.36)
  %.45 = call i8* @numba_unpickle(i8* getelementptr inbounds ([262 x i8], [262 x i8]* @.const.pickledata.4395390016, i64 0, i64 0), i32 262, i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.const.pickledata.4395390016.sha1, i64 0, i64 0))
  %.46 = call i8* (i8*, ...) @PyObject_CallFunctionObjArgs(i8* %.37, i8* %.12, i8* %.45, i8* null)
  %.47 = icmp eq i8* %.46, null
  br i1 %.47, label %B0.if.endif.if, label %B0.if.endif.endif, !prof !0

B0.if.endif.if:                                   ; preds = %B0.if.endif
  store { i8*, i32, i8*, i8*, i32 }* @.const.picklebuf.4395206848, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8, !numba_exception_output !1
  br label %common.ret

B0.if.endif.endif:                                ; preds = %B0.if.endif
  %.54 = call i8* @PyLong_AsVoidPtr(i8* nonnull %.46)
  call void @Py_DecRef(i8* nonnull %.46)
  call void @numba_gil_release(i32* nonnull %.33)
  br label %B0.endif
}

declare void @numba_gil_ensure(i32*) local_unnamed_addr

declare i8* @PyImport_ImportModuleNoBlock(i8*) local_unnamed_addr

declare i8* @PyObject_GetAttrString(i8*, i8*) local_unnamed_addr

declare void @Py_DecRef(i8*) local_unnamed_addr

declare i8* @numba_unpickle(i8*, i32, i8*) local_unnamed_addr

declare i8* @PyObject_CallFunctionObjArgs(i8*, ...) local_unnamed_addr

declare i8* @PyLong_AsVoidPtr(i8*) local_unnamed_addr

declare void @numba_gil_release(i32*) local_unnamed_addr

define i8* @_ZN7cpython8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(i8* nocapture readnone %py_closure, i8* %py_args, i8* nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca i8*, align 8
  %.6 = alloca i8*, align 8
  %.7 = call i32 (i8*, i8*, i64, i64, ...) @PyArg_UnpackTuple(i8* %py_args, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.const.run, i64 0, i64 0), i64 2, i64 2, i8** nonnull %.5, i8** nonnull %.6)
  %.8 = icmp eq i32 %.7, 0
  %.39 = alloca double, align 8
  %excinfo = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  br i1 %.8, label %common.ret, label %entry.endif, !prof !0

common.ret:                                       ; preds = %entry.endif.endif.endif.endif.endif.if.endif, %entry.endif.endif.endif.endif.endif.if.endif.if, %entry.endif.endif.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif, %entry.endif.endif, %entry, %entry.endif.endif.endif.endif.endif.if.if.if, %entry.endif.endif.endif.endif.if.endif, %entry.endif.if
  %common.ret.op = phi i8* [ null, %entry.endif.if ], [ %.58, %entry.endif.endif.endif.endif.if.endif ], [ null, %entry.endif.endif.endif.endif.endif.if.if.if ], [ null, %entry ], [ null, %entry.endif.endif ], [ null, %entry.endif.endif.endif ], [ null, %entry.endif.endif.endif.endif.endif.endif.endif.endif ], [ null, %entry.endif.endif.endif.endif.endif.if.endif.if ], [ null, %entry.endif.endif.endif.endif.endif.if.endif ]
  ret i8* %common.ret.op

entry.endif:                                      ; preds = %entry
  %.12 = load i8*, i8** @_ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd, align 8
  %.17 = icmp eq i8* %.12, null
  br i1 %.17, label %entry.endif.if, label %entry.endif.endif, !prof !0

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([96 x i8], [96 x i8]* @".const.missing Environment: _ZN08NumbaEnv8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd", i64 0, i64 0))
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.21 = load i8*, i8** %.5, align 8
  %.22 = call i8* @PyNumber_Float(i8* %.21)
  %.23 = call double @PyFloat_AsDouble(i8* %.22)
  call void @Py_DecRef(i8* %.22)
  %.25 = call i8* @PyErr_Occurred()
  %.26.not = icmp eq i8* %.25, null
  br i1 %.26.not, label %entry.endif.endif.endif, label %common.ret, !prof !2

entry.endif.endif.endif:                          ; preds = %entry.endif.endif
  %.30 = load i8*, i8** %.6, align 8
  %.31 = call i8* @PyNumber_Float(i8* %.30)
  %.32 = call double @PyFloat_AsDouble(i8* %.31)
  call void @Py_DecRef(i8* %.31)
  %.34 = call i8* @PyErr_Occurred()
  %.35.not = icmp eq i8* %.34, null
  br i1 %.35.not, label %entry.endif.endif.endif.endif, label %common.ret, !prof !2

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif.endif
  store double 0.000000e+00, double* %.39, align 8
  %.43 = call i32 @_ZN8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.39, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo, double %.23, double %.32) #0
  %.44 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.51 = icmp sgt i32 %.43, 0
  %.52 = select i1 %.51, { i8*, i32, i8*, i8*, i32 }* %.44, { i8*, i32, i8*, i8*, i32 }* undef
  switch i32 %.43, label %entry.endif.endif.endif.endif.endif [
    i32 -2, label %entry.endif.endif.endif.endif.if.endif
    i32 0, label %entry.endif.endif.endif.endif.if.endif
  ]

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif
  %0 = icmp sgt i32 %.43, 0
  br i1 %0, label %entry.endif.endif.endif.endif.endif.if, label %entry.endif.endif.endif.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.if.endif:           ; preds = %entry.endif.endif.endif.endif, %entry.endif.endif.endif.endif
  %.53 = load double, double* %.39, align 8
  %.58 = call i8* @PyFloat_FromDouble(double %.53)
  br label %common.ret

entry.endif.endif.endif.endif.endif.if:           ; preds = %entry.endif.endif.endif.endif.endif
  call void @PyErr_Clear()
  %.63 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.52, align 8
  %.64 = extractvalue { i8*, i32, i8*, i8*, i32 } %.63, 4
  %.65 = icmp sgt i32 %.64, 0
  %.68 = extractvalue { i8*, i32, i8*, i8*, i32 } %.63, 0
  %.70 = extractvalue { i8*, i32, i8*, i8*, i32 } %.63, 1
  br i1 %.65, label %entry.endif.endif.endif.endif.endif.if.if, label %entry.endif.endif.endif.endif.endif.if.else

entry.endif.endif.endif.endif.endif.if.if:        ; preds = %entry.endif.endif.endif.endif.endif.if
  %.71 = sext i32 %.70 to i64
  %.72 = call i8* @PyBytes_FromStringAndSize(i8* %.68, i64 %.71)
  %.73 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.52, align 8
  %.74 = extractvalue { i8*, i32, i8*, i8*, i32 } %.73, 2
  %.76 = extractvalue { i8*, i32, i8*, i8*, i32 } %.73, 3
  %.77 = bitcast i8* %.76 to i8* (i8*)*
  %.78 = call i8* %.77(i8* %.74)
  %.79 = icmp eq i8* %.78, null
  br i1 %.79, label %entry.endif.endif.endif.endif.endif.if.if.if, label %entry.endif.endif.endif.endif.endif.if.if.endif, !prof !0

entry.endif.endif.endif.endif.endif.if.else:      ; preds = %entry.endif.endif.endif.endif.endif.if
  %.92 = extractvalue { i8*, i32, i8*, i8*, i32 } %.63, 2
  %.93 = call i8* @numba_unpickle(i8* %.68, i32 %.70, i8* %.92)
  br label %entry.endif.endif.endif.endif.endif.if.endif

entry.endif.endif.endif.endif.endif.if.endif:     ; preds = %entry.endif.endif.endif.endif.endif.if.if.endif, %entry.endif.endif.endif.endif.endif.if.else
  %.95 = phi i8* [ %.83, %entry.endif.endif.endif.endif.endif.if.if.endif ], [ %.93, %entry.endif.endif.endif.endif.endif.if.else ]
  %.96.not = icmp eq i8* %.95, null
  br i1 %.96.not, label %common.ret, label %entry.endif.endif.endif.endif.endif.if.endif.if, !prof !0

entry.endif.endif.endif.endif.endif.if.if.if:     ; preds = %entry.endif.endif.endif.endif.endif.if.if
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @".const.Error creating Python tuple from runtime exception arguments", i64 0, i64 0))
  br label %common.ret

entry.endif.endif.endif.endif.endif.if.if.endif:  ; preds = %entry.endif.endif.endif.endif.endif.if.if
  %.83 = call i8* @numba_runtime_build_excinfo_struct(i8* %.72, i8* nonnull %.78)
  %.84 = bitcast { i8*, i32, i8*, i8*, i32 }* %.52 to i8*
  call void @NRT_Free(i8* nonnull %.84)
  br label %entry.endif.endif.endif.endif.endif.if.endif

entry.endif.endif.endif.endif.endif.if.endif.if:  ; preds = %entry.endif.endif.endif.endif.endif.if.endif
  call void @numba_do_raise(i8* nonnull %.95)
  br label %common.ret

entry.endif.endif.endif.endif.endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.endif.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_SystemError, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @".const.unknown error when calling native function", i64 0, i64 0))
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(i8*, i8*, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(i8*, i8*) local_unnamed_addr

declare i8* @PyNumber_Float(i8*) local_unnamed_addr

declare double @PyFloat_AsDouble(i8*) local_unnamed_addr

declare i8* @PyErr_Occurred() local_unnamed_addr

declare i8* @PyFloat_FromDouble(double) local_unnamed_addr

declare void @PyErr_Clear() local_unnamed_addr

declare i8* @PyBytes_FromStringAndSize(i8*, i64) local_unnamed_addr

declare i8* @numba_runtime_build_excinfo_struct(i8*, i8*) local_unnamed_addr

declare void @NRT_Free(i8*) local_unnamed_addr

declare void @numba_do_raise(i8*) local_unnamed_addr

declare void @PyErr_SetNone(i8*) local_unnamed_addr

define double @cfunc._ZN8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double %.1, double %.2) local_unnamed_addr {
entry:
  %.4 = alloca double, align 8
  store double 0.000000e+00, double* %.4, align 8
  %excinfo = alloca { i8*, i32, i8*, i8*, i32 }*, align 8
  store { i8*, i32, i8*, i8*, i32 }* null, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.8 = call i32 @_ZN8__main__3runB2v2B38c8tJTIeFIjxB2IKSgI4CrvQClQZ6FczSBAA_3dEdd(double* nonnull %.4, { i8*, i32, i8*, i8*, i32 }** nonnull %excinfo, double %.1, double %.2) #0
  %.9 = load { i8*, i32, i8*, i8*, i32 }*, { i8*, i32, i8*, i8*, i32 }** %excinfo, align 8
  %.10.not = icmp eq i32 %.8, 0
  %.16 = icmp sgt i32 %.8, 0
  %.17 = select i1 %.16, { i8*, i32, i8*, i8*, i32 }* %.9, { i8*, i32, i8*, i8*, i32 }* undef
  %.18 = load double, double* %.4, align 8
  %.20 = alloca i32, align 4
  store i32 0, i32* %.20, align 4
  br i1 %.10.not, label %common.ret, label %entry.if, !prof !2

entry.if:                                         ; preds = %entry
  %0 = icmp sgt i32 %.8, 0
  call void @numba_gil_ensure(i32* nonnull %.20)
  br i1 %0, label %entry.if.if, label %entry.if.endif

common.ret:                                       ; preds = %entry, %.23, %entry.if.if.if.if
  %common.ret.op = phi double [ 0.000000e+00, %entry.if.if.if.if ], [ %.18, %.23 ], [ %.18, %entry ]
  ret double %common.ret.op

.23:                                              ; preds = %entry.if.endif, %entry.if.if.endif, %entry.if.if.endif.if, %entry.if.endif.endif.endif, %entry.if.endif.if
  %.71 = call i8* @PyUnicode_FromString(i8* getelementptr inbounds ([50 x i8], [50 x i8]* @".const.<numba.core.cpu.CPUContext object at 0x105fd4e00>", i64 0, i64 0))
  call void @PyErr_WriteUnraisable(i8* %.71)
  call void @Py_DecRef(i8* %.71)
  call void @numba_gil_release(i32* nonnull %.20)
  br label %common.ret

entry.if.if:                                      ; preds = %entry.if
  call void @PyErr_Clear()
  %.26 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.17, align 8
  %.27 = extractvalue { i8*, i32, i8*, i8*, i32 } %.26, 4
  %.28 = icmp sgt i32 %.27, 0
  %.31 = extractvalue { i8*, i32, i8*, i8*, i32 } %.26, 0
  %.33 = extractvalue { i8*, i32, i8*, i8*, i32 } %.26, 1
  br i1 %.28, label %entry.if.if.if, label %entry.if.if.else

entry.if.endif:                                   ; preds = %entry.if
  switch i32 %.8, label %entry.if.endif.endif.endif [
    i32 -3, label %entry.if.endif.if
    i32 -1, label %.23
  ]

entry.if.if.if:                                   ; preds = %entry.if.if
  %.34 = sext i32 %.33 to i64
  %.35 = call i8* @PyBytes_FromStringAndSize(i8* %.31, i64 %.34)
  %.36 = load { i8*, i32, i8*, i8*, i32 }, { i8*, i32, i8*, i8*, i32 }* %.17, align 8
  %.37 = extractvalue { i8*, i32, i8*, i8*, i32 } %.36, 2
  %.39 = extractvalue { i8*, i32, i8*, i8*, i32 } %.36, 3
  %.40 = bitcast i8* %.39 to i8* (i8*)*
  %.41 = call i8* %.40(i8* %.37)
  %.42 = icmp eq i8* %.41, null
  br i1 %.42, label %entry.if.if.if.if, label %entry.if.if.if.endif, !prof !0

entry.if.if.else:                                 ; preds = %entry.if.if
  %.55 = extractvalue { i8*, i32, i8*, i8*, i32 } %.26, 2
  %.56 = call i8* @numba_unpickle(i8* %.31, i32 %.33, i8* %.55)
  br label %entry.if.if.endif

entry.if.if.endif:                                ; preds = %entry.if.if.if.endif, %entry.if.if.else
  %.58 = phi i8* [ %.46, %entry.if.if.if.endif ], [ %.56, %entry.if.if.else ]
  %.59.not = icmp eq i8* %.58, null
  br i1 %.59.not, label %.23, label %entry.if.if.endif.if, !prof !0

entry.if.if.if.if:                                ; preds = %entry.if.if.if
  call void @PyErr_SetString(i8* nonnull @PyExc_RuntimeError, i8* getelementptr inbounds ([61 x i8], [61 x i8]* @".const.Error creating Python tuple from runtime exception arguments.1", i64 0, i64 0))
  br label %common.ret

entry.if.if.if.endif:                             ; preds = %entry.if.if.if
  %.46 = call i8* @numba_runtime_build_excinfo_struct(i8* %.35, i8* nonnull %.41)
  %.47 = bitcast { i8*, i32, i8*, i8*, i32 }* %.17 to i8*
  call void @NRT_Free(i8* nonnull %.47)
  br label %entry.if.if.endif

entry.if.if.endif.if:                             ; preds = %entry.if.if.endif
  call void @numba_do_raise(i8* nonnull %.58)
  br label %.23

entry.if.endif.if:                                ; preds = %entry.if.endif
  call void @PyErr_SetNone(i8* nonnull @PyExc_StopIteration)
  br label %.23

entry.if.endif.endif.endif:                       ; preds = %entry.if.endif
  call void @PyErr_SetString(i8* nonnull @PyExc_SystemError, i8* getelementptr inbounds ([43 x i8], [43 x i8]* @".const.unknown error when calling native function.2", i64 0, i64 0))
  br label %.23
}

declare i8* @PyUnicode_FromString(i8*) local_unnamed_addr

declare void @PyErr_WriteUnraisable(i8*) local_unnamed_addr

attributes #0 = { noinline }

!0 = !{!"branch_weights", i32 1, i32 99}
!1 = !{i1 true}
!2 = !{!"branch_weights", i32 99, i32 1}
"""  # noqa: E501


if __name__ == '__main__':
    # assert add.inspect_llvm() == add_llvm_ref
    """ This assert doesn't quite work due to dynamic entries in LLVM's instructions
        such as @numba.dynamic.globals.######
        Notice that `2.17` from `add` is absent from `run_llvm`
        unlike what would have happened if `add` was `njit` rather than `cfunc`
        `run` cannot be cached for the same reason... """
    run_llvm = next(iter(run.inspect_llvm().values()))
    # assert run_llvm == run_llvm_ref
    # print(run_llvm)
