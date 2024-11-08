; ModuleID = "calc_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define linkonce_odr double @"add"(double %".1", double %".2")
{
f_add_block:
  %"sum_" = fadd double %".1", %".2"
  ret double %"sum_"
}
