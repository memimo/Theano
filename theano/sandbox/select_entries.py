import theano
import numpy


class SelectEntriesTwo(theano.gof.Op):
    def __init__(self, max_allowed=None):
        self.max_allowed = max_allowed

    def __eq__(self, other):
        return (type(self) == type(other)) and \
               (self.max_allowed == other.max_allowed)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.max_allowed)

    def __str__(self):
        if self.max_allowed is None:
            return '%s' % self.__class__.__name__
        else:
            return '%s{%d}' %(self.__class__.__name__,
                              self.max_allowed)

    def make_node(self, x,y):
        return theano.gof.Apply(self, [x,y], [x.type()])

    def perform(self, node, inp, out_):
        x,y = inp
        out, = out_
        #indices = numpy.arange(y.shape[0]).astype('float32')

        if self.max_allowed is None:
            out[0] = x[y != -1]
        else:
            out[0] = x[ (y != -1) * (y < self.max_allowed)]

    def grad(self, inp, grads):
        x,y = inp
        gz, = grads
        return [self(gz,y), TT.zeros_like(y)]

    def R_op(self, inp, epoints):
        x,y = inp
        ep, = epoints
        return [self(ep,y), TT.zeros_like(y)]

select_entries_two = lambda x : SelectEntriesTwo(x)

class SelectEntries(theano.gof.Op):
    def __init__(self, max_allowed=None):
        self.max_allowed = max_allowed

    def __eq__(self, other):
        return (type(self) == type(other)) and \
               (self.max_allowed == other.max_allowed)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.max_allowed)

    def __str__(self):
        if self.max_allowed is None:
            return '%s' % self.__class__.__name__
        else:
            return '%s{%d}' %(self.__class__.__name__,
                              self.max_allowed)

    def make_node(self, x):
        return theano.gof.Apply(self, [x], [x.type(), x.type()])

    def perform(self, node, inp, out_):
        x, = inp
        out, indxs = out_
        indx = numpy.arange(x.shape[0]).astype('int32')
        if self.max_allowed is None:
            out[0] = x[x != -1]
            indxs[0] = indx[x!= -1]
        else:
            out[0] = x[ (x != -1) * (x < self.max_allowed)]
            indxs[0] = indx[ ( x!=-1) * (x < self.max_allowed)]

    def grad(self, inp, grads):
        x,_ = inp
        gz,gi = grads
        # restore the broadcasting pattern of the input
        return [select_entries_two(self.max_allowed)(gz,x),
                select_entries_two(self.max_allowed)(gi,x) ]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return [select_entries_two(self.max_allowed)(eval_points[0],
                                                     inputs[0]),
               select_entries_two(self.max_allowed)(eval_points[1],
                                                    inputs[0]) ]

select_entries=lambda x: SelectEntries(x)

class SelectHigherEntries(theano.gof.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return '%s' % self.__class__.__name__

    def make_node(self, x, y):
        return theano.gof.Apply(self, [x, y], [x.type(), theano.tensor.ivector()])

    def perform(self, node, inp, out_):
        x, y = inp
        out, indxs = out_
        indx = numpy.arange(x.shape[0]).astype('int32')
        out[0] = x[ (x != -1) * (x > y)]
        indxs[0] = indx[ ( x!=-1) * (x > y)]

select_higher_entries= SelectHigherEntries()

