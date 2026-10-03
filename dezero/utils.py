from __future__ import annotations

import os
import subprocess

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
