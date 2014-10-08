import numpy as np
import logging
from scipy.misc.common import logsumexp

log = logging.getLogger(__name__)


class FactorGraph(object):

    def __init__(self, loopy_propagation=False):
        self.factors = []
        self.variables = []
        self.nodes = []
        """type: list of [_Node]"""
        self.loopy_propagation = loopy_propagation
        self.prepared = False

    def add_node(self, node):
        assert node not in self.nodes, "node already part of factor graph"
        self.nodes.append(node)

        node.factorgraph = self
        if isinstance(node, Factor):
            self.factors.append(node)
        elif isinstance(node, Variable):
            self.variables.append(node)
        else:
            assert False, "unknown node type"

    def prepare(self):
        log.debug("preparing all nodes for message passing")
        for n in self.nodes:
            n.prepare()
        self.prepared = True

    def do_message_passing(self, n_iterations=None, only_leafs=True):
        assert self.prepared
        if self.loopy_propagation:
            if n_iterations is None:
                n_iterations = len(self.variables) + 1

            log.debug("doing loopy message passing for %d iterations", n_iterations)
            for i in range(n_iterations):
                log.debug("message passing iteration %d / %d -- variables", i+1, n_iterations)
                for v in self.variables:
                    v.transmit_all_possible_msgs()
                log.debug("message passing iteration %d / %d -- factors", i+1, n_iterations)
                for f in self.factors:
                    f.transmit_all_possible_msgs()
            log.debug("loopy message passing is done")
        else:
            log.debug("starting initial message passing")
            for n in self.nodes:
                if not only_leafs or n.is_leaf:
                    log.debug("asking node %s to send its message", n.name)
                    n.transmit_all_possible_msgs()
            log.debug("initial message passing is done")

    def backtrack_best_state(self):
        log.debug("backtracking best state")
        for v in self.variables:
            if v.is_leaf:
                return v.initiate_backtrack()
        assert False, "no leaf variable available"

    def calculate_marginals(self):
        log.debug("calculating marginals")
        for v in self.variables:
            v.calculate_marginal()


class _Node(object):

    def __init__(self, name):
        self.name = name
        self.received_msgs = {}
        """:type: dict of [_Node, np.ndarray]"""
        self.transmitted = []
        self.factorgraph = None
        """:type: _FactorGraph"""
        self.backtracked = False

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

        if not self.factorgraph.loopy_propagation:
            self.transmit_all_possible_msgs()

    def transmit_msg(self, target):
        assert target in self.neighbours, "target must be connected to this node"
        if not self.factorgraph.loopy_propagation:
            assert target not in self.transmitted, "target has already received a message from this node"

        log.debug("%s->%s: calculating message", self.name, target.name)
        msg = self.calculate_msg(target)

        if not self.factorgraph.loopy_propagation:
            self.transmitted.append(target)

        log.debug("%s->%s: transmitting", self.name, target.name)
        target.receive_msg(self, msg)

    def prepare(self):
        self.transmitted = []
        self.received_msgs = {}
        self.backtracked = False

    def transmit_all_possible_msgs(self):
        if self.is_leaf:
            # message from leaf can always be sent
            nb = self.neighbours[0]
            if self.factorgraph.loopy_propagation or nb not in self.transmitted:
                log.debug("%s: passing message from this leaf node to %s", self.name, nb.name)
                self.transmit_msg(nb)
        elif len(self.received_msgs) == len(self.neighbours) - 1:
            # message from one input is missing, can only send into that direction
            for nb in self.neighbours:
                if nb not in self.received_msgs:
                    if self.factorgraph.loopy_propagation or nb not in self.transmitted:
                        log.debug("%s: have all but one input msgs, passing into missing direction %s",
                                  self.name, nb.name)
                        self.transmit_msg(nb)
                    break
        elif len(self.received_msgs) == len(self.neighbours):
            # messages from all inputs are available, can send into all directions
            log.debug("%s: have all input msgs, passing into all directions", self.name)
            for nb in self.neighbours:
                if self.factorgraph.loopy_propagation or nb not in self.transmitted:
                    self.transmit_msg(nb)

    def backtrack_maximum(self, source, value, nodes=None):
        assert source is None or source in self.neighbours, "unknown source"
        assert isinstance(value, int), "value must be integer"

        if nodes is not None and self not in nodes:
            return False

        if self.factorgraph.loopy_propagation:
            bt = self.backtracked
            self.backtracked = True
            return not bt
        else:
            assert not self.backtracked, "node already backtracked"
            return True


class Factor(_Node):

    def __init__(self, name, factor, variables):
        super(Factor, self).__init__(name)
        self.variables = []
        """:type: list of [Variable]"""
        self.factor = None
        """:type: np.ndarray"""
        self.max_track = {}

        log.debug("Created factor %s", name)

        self._set_factor(factor, variables)

    @property
    def neighbours(self):
        return self.variables

    @property
    def n_variables(self):
        return len(self.variables)

    def _set_factor(self, factor, variables):
        assert isinstance(factor, np.ndarray)
        assert factor.ndim == len(variables)
        for v in variables:
            assert isinstance(v, Variable)

        # deregister with old variables
        for var in self.variables:
            var._deregister_factor(self)

        self.variables = variables
        self.factor = factor

        # register with new variables
        for i, var in enumerate(self.variables):
            assert var.n_states == self.factor.shape[i], \
                "factor shape (%d for dimension %d) does not match number of states (%d) of variable %s" % \
                (self.factor.shape[i], i, var.n_states, var.name)
            var._register_factor(self)

        var_str = ", ".join([v.name for v in self.variables])
        log.debug("Connected factor %s with variables %s", self.name, var_str)

    def _nanmax(self, x, dim, keep_dim_in_argmax=False):
        """Calculates the maximum of x[:^(dim-1), i, ...] for each value of i.
        Returns (maximum, (argmax-dim-1, argmax-dim-2, ...)).
        """

        # handle case when there is no maximization to be done
        if x.ndim == 1 and dim == 0:
            return x, ()

        # make stationary axis the first one
        x = np.rollaxis(x, dim)

        # flatten other axes
        xflat = np.reshape(x, (x.shape[0], -1))

        # calculate maxima
        x_max = np.nanmax(xflat, axis=1)
        xflat_argmax = np.nanargmax(xflat, axis=1)

        # transform indices
        x_argmax = np.unravel_index(xflat_argmax, x.shape[1:])

        # reintroduce dimension in argmax over which iteration takes place, if requested
        if keep_dim_in_argmax:
            x_argmax_new = list(x_argmax[0:dim])
            x_argmax_new.append(np.arange(x.shape[0]))
            x_argmax_new.extend(x_argmax[dim:])
            x_argmax = tuple(x_argmax_new)

        # convert to int dtype
        x_argmax = [np.asarray(p, dtype='int') for p in x_argmax]

        return x_max, x_argmax

    def _logsum(self, x, dim):
        """Calculates the sum of x[:^(dim-1), i, ...] for each value of i.
        Returns a vector of size x.shape[dim]."""

        # handle case when there is no summation to be done
        if x.ndim == 1 and dim == 0:
            return x

        # make stationary axis the first one
        x = np.rollaxis(x, dim)

        # flatten other axes
        xflat = np.reshape(x, (x.shape[0], -1))

        # calculate sum
        x_sum = logsumexp(xflat, axis=1)

        return x_sum

    def calculate_msg(self, target):
        # remove dimensions corresponding to clamped variables from factor
        sl = []
        for var in self.variables:
            if var.clamped_value is None:
                sl.append(slice(None))
            else:
                sl.append(slice(var.clamped_value, var.clamped_value+1))
        sl = tuple(sl)
        # log.debug("%s: factor slice selector: %s", self.name, str(sl))

        # get factor
        m_sp = np.copy(self.factor[sl])
        m_ms = np.copy(self.factor[sl])

        # sum all received messages except from target
        for dim, var in enumerate(self.variables):
            if var is target:
                continue

            var_msg_sp, var_msg_ms = self.received_msgs[var]

            # broadcast received messages
            nshape = [1 for _ in range(dim)]
            nshape.append(-1)
            nshape.extend([1 for _ in range(self.n_variables - dim - 1)])
            var_msg_sp = var_msg_sp.reshape(tuple(nshape))
            var_msg_ms = var_msg_ms.reshape(tuple(nshape))

            m_sp += var_msg_sp
            m_ms += var_msg_ms

        target_dim = self.variables.index(target)

        # sum over all non-target dimensions
        msg_sp = self._logsum(m_sp, target_dim)

        # maximize over all non-target dimensions
        msg_ms, self.max_track[target] = self._nanmax(m_ms, target_dim, keep_dim_in_argmax=True)

        return msg_ms, msg_sp

    def backtrack_maximum(self, source, value, nodes=None):
        if not super(Factor, self).backtrack_maximum(source, value, nodes):
            return
        assert 0 <= value < source.n_states, "value out of source range"

        # propagate to neighbouring variables
        for dim, var in enumerate(self.variables):
            if var is source:
                continue

            var.backtrack_maximum(self, self.max_track[source][dim][value], nodes)


class Variable(_Node):

    def __init__(self, name, n_states):
        super(Variable, self).__init__(name)
        self.n_states = n_states
        self.factors = []
        """:type: list of [Factor]"""
        self.best_state = None
        self.clamped_value = None
        self.marginal = None
        self.marginal_mean = None
        self.marginal_variance = None

        log.debug("Created variable %s", name)

    @property
    def neighbours(self):
        return self.factors

    def _register_factor(self, factor):
        assert factor not in self.factors, "specified factor already connected to this variable"
        self.factors.append(factor)

    def _deregister_factor(self, factor):
        assert factor in self.factors, "specified factor not connected to this variable"
        self.factors.remove(factor)

    def clamp(self, value):
        assert isinstance(value, int), "value must be integer"
        assert 0 <= value < self.n_states, "value out of variable range"
        self.clamped_value = value
        log.debug("%s: clamped to value %d", self.name, self.clamped_value)

    def unclamp(self):
        self.clamped_value = None
        log.debug("%s: unclamped")

    def prepare(self):
        super(Variable, self).prepare()
        if self.factorgraph.loopy_propagation:
            for fac in self.factors:
                self.assume_uniform_msg(fac)

    def assume_uniform_msg(self, fac):
        """Assume unity message from specified factor before real message arrives."""
        assert fac in self.factors, "specified factor not connected to this variable"
        log.debug("%s: assuming unity message from factor %s", self.name, fac.name)
        self.received_msgs[fac] = (np.zeros(self.n_states), np.zeros(self.n_states))

    def calculate_msg(self, target):
        # sum over all received msgs except from target
        if self.clamped_value is not None:
            msg_sp = np.zeros(1)
            msg_ms = np.zeros(1)
        else:
            msg_sp = np.zeros(self.n_states)
            msg_ms = np.zeros(self.n_states)
        for fac in self.factors:
            if fac is target:
                continue
            fac_msg_sp, fac_msg_ms = self.received_msgs[fac]
            msg_sp += fac_msg_sp
            msg_ms += fac_msg_ms
        return msg_sp, msg_ms

    def find_best_state_for_node(self):
        if self.clamped_value is not None:
            m = np.zeros(1)
        else:
            m = np.zeros(self.n_states)
        for fac in self.factors:
            assert fac in self.received_msgs, "no message received from %s" % fac.name
            _, fac_msg_ms = self.received_msgs[fac]
            m += fac_msg_ms

        p_max = np.nanmax(m)
        best_state = int(np.nanargmax(m))
        log.debug("%s: best state from p_max is %d with log p=%g", self.name, best_state, p_max)

        return p_max, best_state

    def initiate_backtrack(self):
        p_max, best_state = self.find_best_state_for_node()
        self.backtrack_maximum(None, best_state)
        return p_max

    def backtrack_maximum(self, source, value, nodes=None):
        if not super(Variable, self).backtrack_maximum(source, value, nodes):
            return
        assert 0 <= value < self.n_states, "value out of variable range"

        if self.clamped_value is not None:
            assert value == 0, "clamped variable must be in 0 state"
            self.best_state = self.clamped_value
            log.debug("%s: using clamped value %d during backtrack", self.name, self.best_state)
        else:
            # store value
            self.best_state = value
            log.debug("%s: best state from backtrack is %d", self.name, self.best_state)

        # propage to neighbouring factors
        for factor in self.factors:
            if factor is source:
                continue

            factor.backtrack_maximum(self, value, nodes)

    def calculate_marginal(self):
        assert len(self.received_msgs) == len(self.neighbours), "message from at least one factor is missing"
        if self.clamped_value is not None:
            marginal = np.zeros(1)
        else:
            marginal = np.zeros(self.n_states)
        for fac in self.factors:
            fac_msg_sp, _ = self.received_msgs[fac]
            marginal += fac_msg_sp

        # renormalize
        fac = logsumexp(marginal)
        self.marginal = marginal - fac

        # calculate mean and variance
        vals = np.arange(self.n_states)
        self.marginal_mean = np.sum(np.exp(self.marginal) * vals)
        self.marginal_variance = np.sum(np.exp(self.marginal) * vals**2) - self.marginal_mean**2

