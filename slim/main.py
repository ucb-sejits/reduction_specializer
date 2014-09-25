"""
specializer slim
"""

import numpy as np
from ctree.jit import LazySpecializedFunction
from ctree.frontend import get_ast
from ctree import browser_show_ast
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode
from ctree.c.nodes import For, SymbolRef, Assign, Lt, PostInc, \
    Constant
import ctypes as ct

import ast


class PointsLoop(CtreeNode):
    _fields = ['target', 'iter_target', 'body']

    def __init__(self, target, iter_target, body):
        self.target = target
        self.iter_target = iter_target
        self.body = body

    def label(self):
        return "{}".format(self.iter_target)


class SlimFrontEnd(PyBasicConversions):
    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and \
           isinstance(node.iter.func, ast.Attribute) and \
           node.iter.func.attr is 'points' and \
           node.iter.func.value.id is 'self':
            target = node.target.id
            iter_target = node.iter.args[0].id
            body = [self.visit(statement) for statement in node.body]
            return PointsLoop(target, iter_target, body)
        else:
            raise Exception("Unsupport for loop")
        # node.iter = something
        # ..
        return node


class SlimCBackend(ast.NodeTransformer):
    def __init__(self, arg_cfg):
        self.arg_cfg = arg_cfg

    def visit_PointsLoop(self, node):
        target = node.target
        return For(
            Assign(SymbolRef(target, ct.c_int()), Constant(0)),
            Lt(SymbolRef(target), Constant(self.arg_cfg[3])),
            PostInc(SymbolRef(target)),
            list(map(self.visit, node.body))
        )


class Slim(LazySpecializedFunction):
    def __init__(self):
        super(Slim, self).__init__(get_ast(self.kernel))

    def args_to_subconfig(self, args):
        A = args[0]
        return (A.dtype, A.shape, A.ndim, A.size)

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        tree = SlimFrontEnd().visit(tree)
        tree = SlimCBackend(arg_cfg).visit(tree)
        browser_show_ast(tree, 'tmp.png')
        print(tree)

    def points(self, input):
        iter = np.nditer(input, flags=['c_index'])
        while not iter.finished:
            yield iter.index
            iter.iternext()

    # def __call__(self, input):
    # 
    #     return self.kernel(input)


class SumReduce(Slim):
    def kernel(self, input):
        result = 0
        for x in self.points(input):
            result += input[x]
        return result

    def kernel(self, input, result):
        # int kernel(int* input):
        # return result




if __name__ == '__main__':
    special_sum = SumReduce()
    a = np.arange(1, 1000100, 1)
    b = special_sum(a)
    assert b == sum(a), "FAILED"
    print("PASSED")


