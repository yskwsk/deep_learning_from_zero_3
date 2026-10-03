from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

import numpy as np

from dezero import Function
from dezero import Variable
from dezero import SimpleFunction, SimpleVariable


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
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
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
