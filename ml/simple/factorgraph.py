import numpy as np
import logging

log = logging.getLogger("factorgraph")


class FactorGraph(object):

    def __init__(self):
        self.factors = []
        self.variables = []
        self.nodes = []
        """type: list of [_Node]"""

    def add_node(self, node):
        assert node in self.nodes, "node already part of factor graph"
        self.nodes.append(node)

        if isinstance(node, Factor):
            self.factors.append(node)
        elif isinstance(node, Variable):
            self.variables.append(node)
        else:
            assert False, "unknown node type"

    def prepare(self):
        for n in self.nodes:
            n.prepare()

    def initiate_message_passing(self):
        for n in self.nodes:
            if n.is_leaf():
                n.transmit_all_possible_msgs()

    def backtrack_best_state(self):
        return self.variables[0].initiate_backtrack()


class _Node(object):

    def __init__(self, name):
        self.name = name
        self.received_msgs = {}
        """:type: dict of [_Node, np.ndarray]"""
        self.transmitted = []

    @property
    def neighbours(self):
        """:rtype: list"""
        raise NotImplementedError()

    @property
    def is_leaf(self):
        return len(self.neighbours) == 1

    def calculate_msg(self, target):
        raise NotImplementedError()

    def receive_msg(self, sender, msg):
        assert sender in self.neighbours, "sender must be connected to this node"
        self.received_msgs[sender] = msg
        self.transmit_all_possible_msgs()

    def transmit_msg(self, target):
        assert target in self.neighbours, "target must be connected to this node"
        assert target not in self.transmitted, "target has already received a message from this node"

        log.debug("%s->%s: calculating message", self.name, target.name)
        msg = self.calculate_msg(target)
        self.transmitted.append(target)

        log.debug("%s->%s: transmitting", self.name, target.name)
        target.receive_msg(self, msg)

    def prepare(self):
        self.transmitted = []
        self.received_msgs = {}

    def transmit_all_possible_msgs(self):
        if len(self.received_msgs) == len(self.neighbours) - 1:
            # message from one input is missing, can only send into that direction
            log.debug("%s: have all but one input msgs, passing into missing direction", self.name)
            for nb in self.neighbours:
                if nb not in self.received_msgs:
                    self.transmit_msg(nb)
                    break
        elif len(self.received_msgs) == len(self.neighbours):
            # messages from all inputs are available, can send into all directions
            log.debug("%s: have all input msgs, passing into all directions", self.name)
            for nb in self.neighbours:
                if nb not in self.transmitted:
                    self.transmit_msg(nb)


class Factor(_Node):

    def __init__(self, name):
        super(Factor, self).__init__(name)
        self.variables = []
        """:type: list of [Variable]"""
        self.factor = None
        """:type: np.ndarray"""
        self.max_track = {}

    @property
    def neighbours(self):
        return self.variables

    @property
    def n_variables(self):
        return len(self.variables)

    def set_factor(self, factor, variables):
        assert isinstance(factor, np.ndarray)
        assert factor.ndim == len(variables)

        # deregister with old variables
        for var in self.variables:
            var._deregister_factor(self)

        self.variables = variables
        self.factor = factor

        # register with new variables
        for var in self.variables:
            var._register_factor(self)

    def _nanmax(self, x, dim, keep_dim_in_argmax=False):
        """Calculates the maximum of x[:^(dim-1), i, ...] for each value of i.
        Returns (maximum, (argmax-dim-1, argmax-dim-2, ...)).
        """

        # make stationary axis the first one
        x = np.rollaxis(x, dim)

        # flatten other axes
        xflat = np.reshape(x, (x.shape[0], -1))

        # calculate maxima
        x_max = np.nanmax(xflat, axis=1)
        xflat_argmax = np.nanargmax(xflat, axis=1)

        # transform indices
        x_argmax = np.unravel_index(xflat_argmax, x.shape[1:])

        # reintroduce dimension over which iteration takes place, if requested
        if keep_dim_in_argmax:
            x_argmax_new = list(x_argmax[0:dim])
            x_argmax_new.append(np.arange(x.shape[0]))
            x_argmax_new.extend(x_argmax[dim:])
            x_argmax = tuple(x_argmax_new)

        return x_max, x_argmax

    def calculate_msg(self, target):
        # factor
        m = self.factor[:]

        # sum all received messages except from target
        for dim, var in enumerate(self.variables):
            if var is target:
                continue

            # broadcast received messages
            nshape = [1 for _ in range(dim)]
            nshape.append(-1)
            nshape.extend([1 for _ in range(self.n_variables - dim - 1)])
            var_msg = self.received_msgs[var].reshape(tuple(nshape))

            m += var_msg

        # maximize over all non-target dimensions
        target_dim = self.variables.index(target)
        msg, mt = self._nanmax(m, target_dim, keep_dim_in_argmax=True)
        self.max_track[target] = mt

        return msg

    def backtrack_maximum(self, source, value):
        assert source in self.variables, "unknown source"
        src_dim = self.variables.index(source)
        assert isinstance(value, int), "value must be integer"
        assert 0 <= value < source.n_states, "value out of source range"

        # propagate to neighbouring variables
        for dim, var in enumerate(self.variables):
            if var is source:
                continue

            var.backtrack_maximum(self, self.max_track[source][dim][value])


class Variable(_Node):

    def __init__(self, name, n_states):
        super(Variable, self).__init__(name)
        self.n_states = n_states
        self.factors = []
        """:type: list of [Factor]"""
        self.best_state = None

    @property
    def neighbours(self):
        return self.factors

    def _register_factor(self, factor):
        assert factor not in self.factors, "specified factor already connected to this variable"
        self.factors.append(factor)

    def _deregister_factor(self, factor):
        assert factor in self.factors, "specified factor not connected to this variable"
        self.factors.remove(factor)

    # def init_empty_msgs(self):
    #     for fac in self.factors:
    #         self.received_msgs[fac] = np.zeros(self.n_states)

    def calculate_msg(self, target):
        # sum over all received msgs except from target
        msg = np.zeros(self.n_states)
        for fac in self.factors:
            if fac is target:
                continue
            msg += self.received_msgs[fac]
        return msg

    def initiate_backtrack(self):
        m = np.zeros(self.n_states)
        for fac in self.factors:
            m += self.received_msgs[fac]

        p_max = np.nanmax(m)
        self.best_state = np.nanargmax(p_max)
        log.debug("%s: best state from p_max is %d with log p=%g", self.name, self.best_state, p_max)

        return p_max

    def backtrack_maximum(self, source, value):
        assert source in self.factors, "unknown source"
        assert isinstance(value, int), "value must be integer"
        assert 0 <= value < self.n_states, "value out of variable range"

        # store value
        self.best_state = value
        log.debug("%s: best state from backtrack is %d", self.name, self.best_state)

        # propage to neighbouring factors
        for factor in self.factors:
            if factor is source:
                continue

            factor.backtrack_maximum(self, value)

