from _typeshed import Incomplete

class Stub:
    """
    A stub object to represent special objects that are meaningless
    outside the context of a CUDA kernel
    """
    _description_: str
    __slots__: Incomplete
    def __new__(cls) -> None: ...
    def __repr__(self) -> str: ...

def stub_function(fn):
    """
    A stub function to represent special functions that are meaningless
    outside the context of a CUDA kernel
    """

class Dim3(Stub):
    """A triple, (x, y, z)"""
    _description_: str
    @property
    def x(self) -> None: ...
    @property
    def y(self) -> None: ...
    @property
    def z(self) -> None: ...

class threadIdx(Dim3):
    """
    The thread indices in the current thread block. Each index is an integer
    spanning the range from 0 inclusive to the corresponding value of the
    attribute in :attr:`numba.cuda.blockDim` exclusive.
    """
    _description_: str

class blockIdx(Dim3):
    """
    The block indices in the grid of thread blocks. Each index is an integer
    spanning the range from 0 inclusive to the corresponding value of the
    attribute in :attr:`numba.cuda.gridDim` exclusive.
    """
    _description_: str

class blockDim(Dim3):
    '''
    The shape of a block of threads, as declared when instantiating the kernel.
    This value is the same for all threads in a given kernel launch, even if
    they belong to different blocks (i.e. each block is "full").
    '''
    _description_: str

class gridDim(Dim3):
    """
    The shape of the grid of blocks. This value is the same for all threads in
    a given kernel launch.
    """
    _description_: str

class warpsize(Stub):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    """
    _description_: str

class laneid(Stub):
    """
    This thread's lane within a warp. Ranges from 0 to
    :attr:`numba.cuda.warpsize` - 1.
    """
    _description_: str

class shared(Stub):
    """
    Shared memory namespace
    """
    _description_: str
    def array(shape, dtype) -> None:
        """
        Allocate a shared array of the given *shape* and *type*. *shape* is
        either an integer or a tuple of integers representing the array's
        dimensions.  *type* is a :ref:`Numba type <numba-types>` of the
        elements needing to be stored in the array.

        The returned array-like object can be read and written to like any
        normal device array (e.g. through indexing).
        """

class local(Stub):
    """
    Local memory namespace
    """
    _description_: str
    def array(shape, dtype) -> None:
        """
        Allocate a local array of the given *shape* and *type*. The array is
        private to the current thread, and resides in global memory. An
        array-like object is returned which can be read and written to like any
        standard array (e.g.  through indexing).
        """

class const(Stub):
    """
    Constant memory namespace
    """
    def array_like(ndarray) -> None:
        """
        Create a const array from *ndarry*. The resulting const array will have
        the same shape, type, and values as *ndarray*.
        """

class syncwarp(Stub):
    """
    syncwarp(mask=0xFFFFFFFF)

    Synchronizes a masked subset of threads in a warp.
    """
    _description_: str

class shfl_sync_intrinsic(Stub):
    """
    shfl_sync_intrinsic(mask, mode, value, mode_offset, clamp)

    Nvvm intrinsic for shuffling data across a warp
    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-datamove
    """
    _description_: str

class vote_sync_intrinsic(Stub):
    """
    vote_sync_intrinsic(mask, mode, predictate)

    Nvvm intrinsic for performing a reduce and broadcast across a warp
    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-vote
    """
    _description_: str

class match_any_sync(Stub):
    """
    match_any_sync(mask, value)

    Nvvm intrinsic for performing a compare and broadcast across a warp.
    Returns a mask of threads that have same value as the given value from
    within the masked warp.
    """
    _description_: str

class match_all_sync(Stub):
    """
    match_all_sync(mask, value)

    Nvvm intrinsic for performing a compare and broadcast across a warp.
    Returns a tuple of (mask, pred), where mask is a mask of threads that have
    same value as the given value from within the masked warp, if they
    all have the same value, otherwise it is 0. Pred is a boolean of whether
    or not all threads in the mask warp have the same warp.
    """
    _description_: str

class activemask(Stub):
    """
    activemask()

    Returns a 32-bit integer mask of all currently active threads in the
    calling warp. The Nth bit is set if the Nth lane in the warp is active when
    activemask() is called. Inactive threads are represented by 0 bits in the
    returned mask. Threads which have exited the kernel are always marked as
    inactive.
    """
    _description_: str

class lanemask_lt(Stub):
    """
    lanemask_lt()

    Returns a 32-bit integer mask of all lanes (including inactive ones) with
    ID less than the current lane.
    """
    _description_: str

class threadfence_block(Stub):
    """
    A memory fence at thread block level
    """
    _description_: str

class threadfence_system(Stub):
    """
    A memory fence at system level: across devices
    """
    _description_: str

class threadfence(Stub):
    """
    A memory fence at device level
    """
    _description_: str

class popc(Stub):
    """
    popc(x)

    Returns the number of set bits in x.
    """
class brev(Stub):
    """
    brev(x)

    Returns the reverse of the bit pattern of x. For example, 0b10110110
    becomes 0b01101101.
    """
class clz(Stub):
    """
    clz(x)

    Returns the number of leading zeros in z.
    """
class ffs(Stub):
    """
    ffs(x)

    Returns the position of the first (least significant) bit set to 1 in x,
    where the least significant bit position is 1. ffs(0) returns 0.
    """
class selp(Stub):
    """
    selp(a, b, c)

    Select between source operands, based on the value of the predicate source
    operand.
    """
class fma(Stub):
    """
    fma(a, b, c)

    Perform the fused multiply-add operation.
    """
class cbrt(Stub):
    '''"
    cbrt(a)

    Perform the cube root operation.
    '''

class atomic(Stub):
    """Namespace for atomic operations
    """
    _description_: str
    class add(Stub):
        """add(ary, idx, val)

        Perform atomic ``ary[idx] += val``. Supported on int32, float32, and
        float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class sub(Stub):
        """sub(ary, idx, val)

        Perform atomic ``ary[idx] -= val``. Supported on int32, float32, and
        float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class and_(Stub):
        """and_(ary, idx, val)

        Perform atomic ``ary[idx] &= val``. Supported on int32, int64, uint32
        and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class or_(Stub):
        """or_(ary, idx, val)

        Perform atomic ``ary[idx] |= val``. Supported on int32, int64, uint32
        and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class xor(Stub):
        """xor(ary, idx, val)

        Perform atomic ``ary[idx] ^= val``. Supported on int32, int64, uint32
        and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class inc(Stub):
        """inc(ary, idx, val)

        Perform atomic ``ary[idx] += 1`` up to val, then reset to 0. Supported
        on uint32, and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class dec(Stub):
        """dec(ary, idx, val)

        Performs::

           ary[idx] = (value if (array[idx] == 0) or
                       (array[idx] > value) else array[idx] - 1)

        Supported on uint32, and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class exch(Stub):
        """exch(ary, idx, val)

        Perform atomic ``ary[idx] = val``. Supported on int32, int64, uint32 and
        uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class max(Stub):
        """max(ary, idx, val)

        Perform atomic ``ary[idx] = max(ary[idx], val)``.

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class min(Stub):
        """min(ary, idx, val)

        Perform atomic ``ary[idx] = min(ary[idx], val)``.

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class nanmax(Stub):
        """nanmax(ary, idx, val)

        Perform atomic ``ary[idx] = max(ary[idx], val)``.

        NOTE: NaN is treated as a missing value such that:
        nanmax(NaN, n) == n, nanmax(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class nanmin(Stub):
        """nanmin(ary, idx, val)

        Perform atomic ``ary[idx] = min(ary[idx], val)``.

        NOTE: NaN is treated as a missing value, such that:
        nanmin(NaN, n) == n, nanmin(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
    class compare_and_swap(Stub):
        """compare_and_swap(ary, old, val)

        Conditionally assign ``val`` to the first element of an 1D array ``ary``
        if the current value matches ``old``.

        Supported on int32, int64, uint32, uint64 operands only.

        Returns the old value as if it is loaded atomically.
        """
    class cas(Stub):
        """cas(ary, idx, old, val)

        Conditionally assign ``val`` to the element ``idx`` of an array
        ``ary`` if the current value of ``ary[idx]`` matches ``old``.

        Supported on int32, int64, uint32, uint64 operands only.

        Returns the old value as if it is loaded atomically.
        """

class nanosleep(Stub):
    """
    nanosleep(ns)

    Suspends the thread for a sleep duration approximately close to the delay
    `ns`, specified in nanoseconds.
    """
    _description_: str

class fp16(Stub):
    """Namespace for fp16 operations
    """
    _description_: str
    class hadd(Stub):
        """hadd(a, b)

        Perform fp16 addition, (a + b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the addition.

        """
    class hsub(Stub):
        """hsub(a, b)

        Perform fp16 subtraction, (a - b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the subtraction.

        """
    class hmul(Stub):
        """hmul(a, b)

        Perform fp16 multiplication, (a * b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the multiplication.

        """
    class hdiv(Stub):
        """hdiv(a, b)

        Perform fp16 division, (a / b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the division

        """
    class hfma(Stub):
        """hfma(a, b, c)

        Perform fp16 multiply and accumulate, (a * b) + c in round to nearest
        mode. Supported on fp16 operands only.

        Returns the fp16 result of the multiplication.

        """
    class hneg(Stub):
        """hneg(a)

        Perform fp16 negation, -(a). Supported on fp16 operands only.

        Returns the fp16 result of the negation.

        """
    class habs(Stub):
        """habs(a)

        Perform fp16 absolute value, |a|. Supported on fp16 operands only.

        Returns the fp16 result of the absolute value.

        """
    class hsin(Stub):
        """hsin(a)

        Calculate sine in round to nearest even mode. Supported on fp16
        operands only.

        Returns the sine result.

        """
    class hcos(Stub):
        """hsin(a)

        Calculate cosine in round to nearest even mode. Supported on fp16
        operands only.

        Returns the cosine result.

        """
    class hlog(Stub):
        """hlog(a)

        Calculate natural logarithm in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the natural logarithm result.

        """
    class hlog10(Stub):
        """hlog10(a)

        Calculate logarithm base 10 in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the logarithm base 10 result.

        """
    class hlog2(Stub):
        """hlog2(a)

        Calculate logarithm base 2 in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the logarithm base 2 result.

        """
    class hexp(Stub):
        """hexp(a)

        Calculate natural exponential, exp(a), in round to nearest mode.
        Supported on fp16 operands only.

        Returns the natural exponential result.

        """
    class hexp10(Stub):
        """hexp10(a)

        Calculate exponential base 10 (10 ** a) in round to nearest mode.
        Supported on fp16 operands only.

        Returns the exponential base 10 result.

        """
    class hexp2(Stub):
        """hexp2(a)

        Calculate exponential base 2 (2 ** a) in round to nearest mode.
        Supported on fp16 operands only.

        Returns the exponential base 2 result.

        """
    class hfloor(Stub):
        """hfloor(a)

        Calculate the floor, the largest integer less than or equal to 'a'.
        Supported on fp16 operands only.

        Returns the floor result.

        """
    class hceil(Stub):
        """hceil(a)

        Calculate the ceil, the smallest integer greater than or equal to 'a'.
        Supported on fp16 operands only.

        Returns the ceil result.

        """
    class hsqrt(Stub):
        """hsqrt(a)

        Calculate the square root of the input argument in round to nearest
        mode. Supported on fp16 operands only.

        Returns the square root result.

        """
    class hrsqrt(Stub):
        """hrsqrt(a)

        Calculate the reciprocal square root of the input argument in round
        to nearest even mode. Supported on fp16 operands only.

        Returns the reciprocal square root result.

        """
    class hrcp(Stub):
        """hrcp(a)

        Calculate the reciprocal of the input argument in round to nearest
        even mode. Supported on fp16 operands only.

        Returns the reciprocal result.

        """
    class hrint(Stub):
        """hrint(a)

        Round the input argument to nearest integer value. Supported on fp16
        operands only.

        Returns the rounded result.

        """
    class htrunc(Stub):
        """htrunc(a)

        Truncate the input argument to its integer portion. Supported
        on fp16 operands only.

        Returns the truncated result.

        """
    class heq(Stub):
        """heq(a, b)

        Perform fp16 comparison, (a == b). Supported
        on fp16 operands only.

        Returns True if a and b are equal and False otherwise.

        """
    class hne(Stub):
        """hne(a, b)

        Perform fp16 comparison, (a != b). Supported
        on fp16 operands only.

        Returns True if a and b are not equal and False otherwise.

        """
    class hge(Stub):
        """hge(a, b)

        Perform fp16 comparison, (a >= b). Supported
        on fp16 operands only.

        Returns True if a is >= b and False otherwise.

        """
    class hgt(Stub):
        """hgt(a, b)

        Perform fp16 comparison, (a > b). Supported
        on fp16 operands only.

        Returns True if a is > b and False otherwise.

        """
    class hle(Stub):
        """hle(a, b)

        Perform fp16 comparison, (a <= b). Supported
        on fp16 operands only.

        Returns True if a is <= b and False otherwise.

        """
    class hlt(Stub):
        """hlt(a, b)

        Perform fp16 comparison, (a < b). Supported
        on fp16 operands only.

        Returns True if a is < b and False otherwise.

        """
    class hmax(Stub):
        """hmax(a, b)

        Perform fp16 maximum operation, max(a,b) Supported
        on fp16 operands only.

        Returns a if a is greater than b, returns b otherwise.

        """
    class hmin(Stub):
        """hmin(a, b)

        Perform fp16 minimum operation, min(a,b). Supported
        on fp16 operands only.

        Returns a if a is less than b, returns b otherwise.

        """

def make_vector_type_stubs():
    """Make user facing objects for vector types"""
def map_vector_type_stubs_to_alias(vector_type_stubs) -> None:
    """For each of the stubs, create its aliases.

    For example: float64x3 -> double3
    """

_vector_type_stubs: Incomplete
