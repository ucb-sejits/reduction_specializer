"""
specializer slim
"""

import numpy as np
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.frontend import get_ast
from ctree import browser_show_ast
from ctree.transformations import PyBasicConversions
from ctree.nodes import CtreeNode
from ctree.c.nodes import For, SymbolRef, Assign, Lt, PostInc, \
    Constant, Deref
import ctree.np
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
        self.retval = None

    def visit_FunctionDecl(self, node):
        arg_type = np.ctypeslib.ndpointer(self.arg_cfg[0],
                                          self.arg_cfg[2],
                                          self.arg_cfg[1])

        param = node.params[1]
        param.type = arg_type()
        node.params = [param]
        retval = SymbolRef("output", arg_type())
        self.retval = "output"
        node.params.append(retval)
        node.defn = list(map(self.visit, node.defn))
        node.defn[0].left.type = arg_type._dtype_.type()
        return node

    def visit_PointsLoop(self, node):
        target = node.target
        return For(
            Assign(SymbolRef(target, ct.c_int()), Constant(0)),
            Lt(SymbolRef(target), Constant(self.arg_cfg[3])),
            PostInc(SymbolRef(target)),
            list(map(self.visit, node.body))
        )

    def visit_Return(self, node):
        return Assign(Deref(SymbolRef(self.retval)), node.value)


class ConcreteSlim(ConcreteSpecializedFunction):
    def finalize(self, entry_name, tree, entry_type):
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, input):
        output = None
        if input.dtype == np.float32:
            output = ct.c_float()
        elif input.dtype == np.int32:
            output = ct.c_int()

        self._c_function(input, ct.byref(output))
        return output.value



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
        # browser_show_ast(tree, 'tmp.png')
        fn = ConcreteSlim()
        arg_type = np.ctypeslib.ndpointer(arg_cfg[0],
                                          arg_cfg[2],
                                          arg_cfg[1])
        print(tree.files[0])
        return fn.finalize('kernel',
                           tree,
                           ct.CFUNCTYPE(None, arg_type, ct.POINTER(ct.c_float)))

    def points(self, input):
        iter = np.nditer(input, flags=['c_index'])
        while not iter.finished:
            yield iter.index
            iter.iternext()


class SumReduce(Slim):
    def kernel(self, input):
        result = input[0]
        for x in self.points(input):
            result = max(x, result)
        return result


if __name__ == '__main__':
    special_sum = SumReduce()
    a = (np.random.rand(1024 * 1024) * 100).astype(np.float32)
    b = special_sum(a)
    print(b)
    print(np.sum(a))
    assert b == sum(a), "FAILED"
    print("PASSED")


