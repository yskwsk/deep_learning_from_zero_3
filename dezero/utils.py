from __future__ import annotations

import os
import subprocess
import urllib.request
from collections.abc import Sequence
from typing import Callable, Literal

import numpy as np

from dezero import Function
from dezero import Variable
from dezero import SimpleFunction, SimpleVariable
from dezero import cuda
from dezero import as_variable
from dezero.array_type import ArrayType


# =============================================================================
# Visualize for computational graph
# =============================================================================

def _dot_var(v: Variable | SimpleVariable, verbose: bool = False) -> str:
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        name += ': ' + str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f: Function | SimpleFunction) -> str:
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        ret += dot_edge.format(id(f), id(y()))

    return ret


def get_dot_graph(output: Variable | SimpleVariable, verbose: bool = True) -> str:
    txt = ''
    funcs: list[Function | SimpleFunction] = []
    seen_set: set[Function | SimpleFunction] = set()

    def add_func(f: Function | SimpleFunction | None) -> None:
        if (f is not None) and (f not in seen_set):
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)
            add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


def plot_dot_graph(
    output: Variable | SimpleVariable,
    verbose: bool = True,
    to_file: str = "graph.png"
) -> None:
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, mode="w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================

def sum_to(x: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Sum elements along axes to output an array of a given shape

    Args:
        x (ndarray): Input array.
        shape (Sequence[int]): the shape of output array

    Retuens:
        ndarray: Output array of the shape
    """
    ndim = len(shape)

    # 入力配列xの次元とshapeの次元の差
    lead = x.ndim - ndim

    # 次元の差の軸
    lead_axis = tuple(range(lead))

    # 入力配列の和をとる次元の軸は、次元の差の軸 + 出力の形状で要素数が1となっている箇所に対応する軸
    # 例えば、入力配列x.shape=(2, 1, 2, 3)、出力の配列の形状shape=(1, 3)のとする。
    # このとき、x.ndim = 4、ndim = len(shape) = 2となる。
    # よって、入力と出力の次元の差leadは2となり、lead_axes=(0, 1)となる。
    # shape[0]=1で、shape[0]は入力配列の次元の軸の2次元目の軸となる(軸の次元は0から数える)。
    # したがって、入力配列において和をとる軸は、(0, 1, 2)となる。
    # 他の例として、入力配列の形状は同じx.shape=(2, 1, 2, 3)で、出力配列の形状がshape=(2, 1)の場合、
    # shape[1]=1で、これに対応する入力配列の軸の次元は3次元目である。
    # したがって、入力配列において和をとる軸は、(0, 1, 3)となる。
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(
    gy: Variable,
    x_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | list[int] | None,
    keepdims: bool
) -> Variable:
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)

    tupled_axis: tuple[int, ...] | None = None
    if isinstance(axis, int):
        tupled_axis = (axis,)
    elif isinstance(axis, tuple | list):
        tupled_axis = tuple(axis)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape_list = list(gy.shape)
        for a in sorted(actual_axis):
            shape_list.insert(a, 1)
        shape = tuple(shape_list)
    else:
        shape = gy.shape

    # reshape
    gy = gy.reshape(shape)
    return gy


def logsumexp(
    x: ArrayType,
    axis: int | tuple[int, ...] | None = 1
) -> ArrayType:
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    # y = xp.exp(y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    # s = xp.log(s)
    m += s
    return m


def max_backward_shape(
    x: ArrayType | Variable,
    axis: int | Sequence[int] | None
):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(
    f: Callable[[Variable], Variable],
    x: np.ndarray | Variable,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    *args,
    **kwargs
):
    """Test backward procedure of a given function.

    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing the gradient.
        *args: If `f` needs variables except `x`, you can specify with this argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this argument.

    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    x = as_variable(x)
    assert x.data is not None
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    assert x.grad is not None
    assert x.grad.data is not None
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print('')
        print('========== FAILED (Gradient Check) ==========')
        print('Numerical Grad')
        print(' shape: {}'.format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
        print('Backprop Grad')
        print(' shape: {}'.format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(' values: {} ...'.format(val[1:-1]))
    return res


def numerical_grad(f, x, *args, **kwargs) -> ArrayType:
    """Computes numerical gradient by finite differences.

    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A target `Variable` for computing the gradient.
        *args: If `f` needs variables except `x`, you can specify with this argument.
        **kwargs: If `f` needs keyword variables, you can specify with this argument.

    Returns:
        `ndarray`: Gradient.
    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    assert x is not None
    xp = cuda.get_array_module(x)
    if xp is not np:
        np_x = cuda.as_numpy(x)
    else:
        np_x = x
    grad = xp.zeros_like(x)

    flags: list[Literal["multi_index"]] = ["multi_index"]
    op_flags: list[Literal["readwrite"]] = ["readwrite"]

    it = np.nditer(np_x, flags=flags, op_flags=op_flags)

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1 = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        assert y1 is not None
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2 = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        assert y2 is not None
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad


def array_equal(a: ArrayType | Variable, b: ArrayType | Variable) -> bool:
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays to compare

    Returns:
        bool: True if the two arrays are equal.
    """
    if isinstance(a, Variable):
        if a.data is None:
            raise ValueError("a.data is None")
        _a = cuda.as_numpy(a.data)
    else:
        _a = cuda.as_numpy(a)

    if isinstance(b, Variable):
        if b.data is None:
            raise ValueError("b.data is None")
        _b = cuda.as_numpy(b.data)
    else:
        _b = cuda.as_numpy(b)

    return np.array_equal(_a, _b)


def array_allclose(
    a: ArrayType | Variable,
    b: ArrayType | Variable,
    rtol: float = 1e-4,
    atol: float = 1e-5
) -> bool:
    """Returns True if two arrays(or variables) are element-wise equal within a
    tolerance.

    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays to compare
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.

    Returns:
        bool: True if the two arrays are equal within the given tolerance, False otherwise.
    """
    if isinstance(a, Variable):
        if a.data is None:
            raise ValueError("a.data is None")
        _a = cuda.as_numpy(a.data)
    else:
        _a = cuda.as_numpy(a)

    if isinstance(b, Variable):
        if b.data is None:
            raise ValueError("b.data is None")
        _b = cuda.as_numpy(b.data)
    else:
        _b = cuda.as_numpy(b)

    return np.allclose(_a, _b, atol=atol, rtol=rtol)


# =============================================================================
# download function
# =============================================================================

def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def get_file(url: str, file_name: str | None = None) -> str:
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        print(f"File '{file_name}' already exists: {file_path}")
        return file_path

    print(f"Downloading: {file_name}, Save path: {file_path}")
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(
    size: int,
    k: int,
    s: int,
    p: int
) -> int:
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(
    input_size: int,
    kernel_size: int,
    stride: int,
    pad: int
) -> int:
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise TypeError(f"{type(x)} is not supported")
