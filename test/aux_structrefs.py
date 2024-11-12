from numba import njit, float64, int16, int64
from numba.core import types
from numba.experimental import structref


@structref.register
class S1TypeClass(types.StructRef):
    pass


class S1(structref.StructRefProxy):
    def __new__(cls, x1, x2, x3):
        return s1_constructor(x1, x2, x3)


fields_s1 = [("x1", int16), ("x2", int64), ("x3", float64)]
structref.define_proxy(S1, S1TypeClass, [field[0] for field in fields_s1])

S1Type = S1TypeClass(fields_s1)


@njit(S1Type(int16, int64, float64))
def s1_constructor(x1, x2, x3):
    return S1(x1, x2, x3)
