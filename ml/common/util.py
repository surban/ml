# -*- coding: utf-8 -*-

import ml.common.gpu
import os
import sys
import glob
import math
import fractions
import numpy as np
import gnumpy as gp
import gc
import theano
import scipy.io
import time
import ctypes
import imp
import itertools
import ml.common.progress as progress
import matplotlib.pyplot as plt
import pickle
import signal

import __main__ as main

from operator import mul

if sys.platform == 'nt':
    import msvcrt

try:    
    import IPython.core.display
    have_notebook = True
except ImportError:
    have_notebook = False


np.set_printoptions(precision=3, suppress=True)


def print_total_garray_size():
    gp.free_reuse_cache()
    tot = 0
    for obj in gc.get_objects():
        if isinstance(obj, gp.garray):
            tot += obj.size
    print "Total GPU memory used by garrays:        %.1f MB" % (tot / 1e6)
    print "Total GPU memory use reported by gnumpy: %.1f MB" % (gp.memory_in_use() / 1e6)

def find_garrays(obj, path="", max_depth=5):
    if max_depth < 0:
        return

    #print "Trying: ", path
    if isinstance(obj, gp.garray):
        if path == "":
            print "obj itself"
        else:
            print path
    else:
        members = dir(obj)
        for m in members:
            if not m.startswith('__'):
                find_garrays(getattr(obj, m), path + "." + m, max_depth-1)

        try:
            keys = obj.keys()
            for k in keys:
                find_garrays(obj[k], path + "['" + str(k) + "']", max_depth-1)
        except Exception:
            pass


def plot_box(x, lower, upper, middle):
    width = 0.5
    plt.gca().add_patch(plt.Rectangle((x-width/2,lower), width, upper-lower, fill=False))
    plt.hlines(middle, x-width/2, x+width/2, 'r')
    
def add_noise(samples, p):
    noise = np.random.binomial(1, p, size=samples.shape)
    noise = 1 - noise * 2
    samples = samples - 0.5
    samples = samples * noise
    samples = samples + 0.5
    return samples

def flatten_samples(samples):
    return samples.reshape((samples.shape[0], -1))

def unflatten_samples_like(samples, ref):
    return samples.reshape(ref.shape)

def output_table(titles, values):
    if have_notebook:
        header_cells = ["<th>%s</th>" % t for t in titles]
        value_cells = ["<td>%s</td>" % v for v in values]
        header_row = "<tr>" + "".join(header_cells) + "</tr>"
        value_row = "<tr>" + "".join(value_cells) + "</tr>"
        table = "<table>" + header_row + value_row + "</table>"
        IPython.core.display.display_html(table, raw=True)
    else:
        header = "\t".join(titles)
        values = "\t".join(values)
        print header
        print values

def masked_set(var, mask, val):
    """Sets var[i,j] = val[i,j] where mask[i,j] == 1"""
    var[:, :] = var * (1. - mask) + val * mask
    return var

def ipy_plot_samples(samples, samples_force=None, twod=False, width=28, height=28):
    return plt.imshow(plot_samples(samples, samples_force=samples_force, 
                                   twod=twod, width=width, height=height),
                      interpolation='none')

def plot_samples(samples, samples_force=None, twod=False, width=28, height=28):
    samples = gp.as_numpy_array(samples)
    samples = np.asarray(samples)
    if samples_force is not None:
        samples_force = gp.as_numpy_array(samples_force)
        samples_force = np.asarray(samples_force)
   
    if (not twod and samples.ndim == 1) or (twod and samples.ndim == 2):
        if twod:
            height = samples.shape[0]
            width = samples.shape[1]
        return _plot_one_sample(samples, samples_force, twod=twod,
                                width=width, height=height)
    else:
        n_samples = samples.shape[0]
        if twod:
            height = samples.shape[1]
            width = samples.shape[2]
        out = np.zeros((height, width*n_samples, 3))
        for s in range(n_samples):
            if samples_force is not None:
                o = _plot_one_sample(samples[s], samples_force[s], twod=twod,
                                     width=width, height=height)
            else:
                o =  _plot_one_sample(samples[s], None, twod=twod,
                                      width=width, height=height)
            out[:, s*width : (s+1)*width, :] = o
        return out

def _plot_one_sample(sample, sample_force, 
                     twod, width=None, height=None):  
    if twod:
        height = sample.shape[0]
        width = sample.shape[1]
        s = sample
    else:
        s = np.reshape(sample, (height, width))
    
    red = np.zeros((height, width))
    green = np.zeros((height, width))
    blue = np.zeros((height, width))

    red[:] = s[:]
    green[:] = s[:]
    blue[:] = s[:]

    if sample_force is not None:
        if twod:
            sf = sample_force
        else:
            sf = np.reshape(sample_force, (height, width))    

        red[(sf == 1) & (s == 1)] = 0.9
        green[(sf == 1) & (s == 1)] = 0
        blue[(sf == 1) & (s == 1)] = 0

        red[(sf == 1) & (s == 0)] = 0.3
        green[(sf == 1) & (s == 0)] = 0
        blue[(sf == 1) & (s == 0)] = 0


    out = np.zeros((height, width, 3))    
    out[:, :, 0] = red
    out[:, :, 1] = green
    out[:, :, 2] = blue
    return out    
    

def map_reduce(X, batch_size, map_func, reduce_func,
               samples_are='rows'):
    rs = []
    for x in draw_slices(X, batch_size, kind='sequential', 
                         samples_are=samples_are, stop=True):
        m = map_func(x)
        r = reduce_func(m)
        rs.append(r)
    return reduce_func(rs)


def map(X, batch_size, map_func, 
        caption="", force_output_type=None):

    if force_output_type is not None:
        if force_output_type == 'gnumpy':
            xp = gp
        elif force_output_type == 'numpy':
            xp = np
        else:
            assert False, "force_output_type must be either numpy or gnumpy"
    else:
        if isinstance(X, gp.garray):
            xp = gp
        else:
            xp = np

    ms = None
    for b, x in enumerate(draw_slices(X, batch_size, kind='sequential', 
                                      samples_are='rows', stop=True, 
                                      last_batch_may_be_smaller=True)):
        progress.status(b, X.shape[0] / batch_size, caption)

        m = map_func(x)
        if ms is None:
            if m.ndim == 1:
                ms = xp.zeros((X.shape[0],))
            elif m.ndim == 2:
                ms = xp.zeros((X.shape[0], m.shape[1]))
            elif m.ndim == 3:
                ms = xp.zeros((X.shape[0], m.shape[1], m.shape[2]))
            elif m.ndim == 4:
                ms = xp.zeros((X.shape[0], m.shape[1], m.shape[2], m.shape[3]))
            else:
                assert False, "%d dimensions are not supported" % m.ndim
        
        if ms.ndim == 1:
            ms[b*batch_size : (b+1)*batch_size] = m
        elif ms.ndim == 2:
            ms[b*batch_size : (b+1)*batch_size, :] = m
        elif ms.ndim == 3:
            ms[b*batch_size : (b+1)*batch_size, :, :] = m
        elif ms.ndim == 4:
            ms[b*batch_size : (b+1)*batch_size, :, :, :] = m

    progress.done()
    return ms


def interval_contains(interval, x):
    "Returns true if interval contains x"
    assert interval[1] >= interval[0]
    return interval[0] <= x <= interval[1]

class LCGRand(object):
    a = 1664525
    c = 1013904223
    m = 2**32

    def __init__(self):
        self.x = None

    def seed(self, j):
        self.x = j

    def get_uint32(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x

    def get_float(self):
        return self.get_uint32() / float(self.m)

    def rand(self, shape):
        x = np.zeros(shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i,j] = self.get_float()
        return gp.as_garray(x)


myrand = LCGRand()
      


class Tee:
    def __init__(self, _fd1, _fd2) :
        self.fd1 = _fd1
        self.fd2 = _fd2

    def __del__(self) :
        if self.fd1 != sys.stdout and self.fd1 != sys.stderr :
            self.fd1.close()
        if self.fd2 != sys.stdout and self.fd2 != sys.stderr :
            self.fd2.close()

    def write(self, text) :
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self) :
        self.fd1.flush()
        self.fd2.flush()

stdoutsav = None
outputlog = None

def tee_output_to_log(filename):
    global stdoutsav, outputlog

    if stdoutsav is not None:
        untee_output()

    stdoutsav = sys.stdout
    #stderrsav = sys.stderr
    try:
        os.rename(filename, filename + ".bak")
    except Exception, e:
        pass
    outputlog = open(filename, "w")
    sys.stdout = Tee(stdoutsav, outputlog)
    #sys.stderr = Tee(stderrsav, outputlog)

def untee_output():
    global stdoutsav, outputlog
    
    if stdoutsav is not None:
        sys.stdout.flush()
        sys.stdout = stdoutsav
        outputlog.close()

        stdoutsav = None
        outputlog = None


def pack_in_batches(gen, batch_size):
    """Packs batch_size samples from gen into a batch"""
    pos = 0
    for sample in gen:
        if pos == 0:
            batch = np.zeros((batch_size, len(sample)))
        batch[pos,:] = sample
        pos += 1        
        if pos == batch_size:
            yield gp.as_garray(batch)
            pos = 0

    if pos > 0:
        yield gp.as_garray(batch[0:pos,:])

def draw_slices(X, batch_size, kind='sequential', samples_are='rows', 
                stop=False, last_batch_may_be_smaller=False):
    assert kind == 'sequential' or kind == 'random'
    assert samples_are == 'rows' or samples_are == 'columns'

    if samples_are == 'rows':
        n_samples = X.shape[0]
    elif samples_are == 'columns':
        n_samples = X.shape[1]

    if kind == 'sequential':
        if not last_batch_may_be_smaller:
            assert n_samples % batch_size == 0
        pos = 0
        while True:
            if samples_are == 'rows':
                slice = X[pos:pos+batch_size,:]
                yield slice
                pos += slice.shape[0]
            elif samples_are == 'columns':
                slice = X[:,pos:pos+batch_size]
                yield slice
                pos += slice.shape[1]
            if pos == n_samples:
                if stop:
                    break
                else:
                    pos = 0
    elif kind == 'random':
        drawn_samples = 0
        while True:
            r = np.random.randint(n_samples, size=batch_size)
            drawn_samples += batch_size
            yield r
            if stop and drawn_samples >= n_samples:
                break

def in_plot_directory(value=None):
    if value != None:
        if value:
            os.environ['IN_PLOT_DIRECTORY'] = '1'
        else:
            os.environ['IN_PLOT_DIRECTORY'] = '0'
    else:
        if 'IN_PLOT_DIRECTORY' in os.environ:
            return os.environ['IN_PLOT_DIRECTORY'] == '1'
        else:
            return False


def multiglob(*patterns):
    return itertools.chain.from_iterable(glob.glob(pattern) for pattern in patterns)

def standard_cfg(clean_plots=False, prepend_scriptname=True, with_checkpoint=False):
    """Reads the configuration file cfg.py from the configuration directory
    specified as the first parameter on the command line.
    Returns a tuple consisting of the configuration module and the plot 
    directory."""
    if len(sys.argv) < 2:
        if with_checkpoint:
            print "Usage: %s <config> [continue]" % sys.argv[0]
        else:
            print "Usage: %s <config>" % sys.argv[0]
        sys.exit(1)

    scriptname, _ = os.path.splitext(os.path.basename(main.__file__))
    if prepend_scriptname:
        cfgdir = os.path.join(scriptname, sys.argv[1])
    else:
        cfgdir = sys.argv[1]
    cfgname = os.path.join(cfgdir, 'cfg.py')
    if not os.path.exists(cfgname):
        print "Configuration %s not found" % cfgname
        sys.exit(2)
    print "Using configuration %s" % cfgname
    sys.dont_write_bytecode = True
    cfg = imp.load_source('cfg', cfgname)

    # load checkpoint if requested
    checkpoint = None
    if with_checkpoint:
        cp_handler = CheckpointHandler(cfgdir)
        if (('JOB_REQUEUED' in os.environ and os.environ['JOB_REQUEUED'] == 'yes') or
                (len(sys.argv) >= 3 and sys.argv[2].startswith("cont"))):
            checkpoint = cp_handler.load()
        else:
            print "Using no checkpoint"
            cp_handler.remove()

    # clean plot directory
    if clean_plots and checkpoint is None:
        curdir = os.path.abspath(os.curdir)
        os.chdir(cfgdir)
        for file in multiglob('*.png', '*.pdf'):
            os.remove(file)
        os.chdir(curdir)

    if with_checkpoint:
        return cfg, cfgdir, cp_handler, checkpoint
    else:
        return cfg, cfgdir

def enter_plot_directory(dirname, clean=False):
    """Creates and chdirs into given dirname. 
    
    If clean is true deletes all *.png files in the directory.
    """
    if in_plot_directory():
        leave_plot_directory()

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)
    in_plot_directory(True)
    print "Entered plot directory %s" % os.getcwd()

    if clean:
        for file in glob.glob("*.png"):
            os.remove(file)

def enter_plot_directory_from_cfg(task, **cfg):
    #assert len(cfg) > 0, "At least one configuration parameter must be specified"

    dir = task
    for name in sorted(cfg.keys()):
        if name == 'clean':
            continue
        s = "%s=%s" % (name, str(cfg[name]))
        dir = dir + "-" + s

    if 'clean' in cfg:
        enter_plot_directory(dir, clean=cfg['clean'])
    else:
        enter_plot_directory(dir)




def leave_plot_directory():
    if in_plot_directory():
        print "Leaving plot directory %s" % os.getcwd()
        os.chdir("..")
        in_plot_directory(False)
        #print "Now in directory %s" % os.getcwd()


def mean_over_dataset(f, dataset, batch_size, sample_axis):
    """Calculates the mean of function f over dataset.
    The dataset is split into batches of size batch_size and f is evaluated
    on each batch. Finally the average value of the results of the evaluations
    on the batches is returned. 
    If sample_axis is 0, every row is a sample.
    If sample_axis is 1, every column is a sample."""
    # we cut off as many samples as necesarry so that the number of samples
    # in the dataset becomes a multiple of the batch_size
    n_batches = int(dataset.shape[sample_axis] / batch_size)

    for s in range(n_batches):
        if sample_axis == 0:
            val = f(dataset[s*batch_size : (s+1)*batch_size,:])
        elif sample_axis == 1:
            val = f(dataset[:,s*batch_size : (s+1)*batch_size])

        if s == 0:
            acc = val
        else:
            acc += val

    return acc / n_batches

def logsum(x, shift_method='mean'):
    """Calculates log(sum(exp(x))) in such a way that precision is hopefully
    preserved."""
    if shift_method == 'mean':
        s = np.mean(x)
    elif shift_method == 'roweis':
        M = np.finfo(x.dtype).max
        s = np.max(x) - np.log(M) / 2.0
    else:
        assert False

    return np.log(np.sum(np.exp(x - s))) + s

def logplus(x, y, shift_method='mean'):
    """Calculates log(exp(x) + exp(y)) in such a way that precision is hopefully
    preserved."""
    return logsum(np.array([x, y]), shift_method)

def logminus(x, y, shift_method='mean', raise_when_negative=True):
    """Calculates log(exp(x) - exp(y)) in such a way that precision is hopefully
    preserved."""
    if shift_method == 'mean':
        s = (x + y) / 2.0
    elif shift_method == 'roweis':
        M = sys.float_info.max
        s = max(x, y) - np.log(M) / 2.0
    else:
        assert False

    if raise_when_negative:
        return np.log(np.exp(x - s) - np.exp(y - s)) + s
    else:
        eres = np.exp(x - s) - np.exp(y - s)
        if eres > 0:
            return np.log(eres) + s
        else:
            return float('-inf')


def get_base_dir():
    """Base directory of the ML project"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def classification_error(predicted, truth):
    """Calculates the classification error probability given the predicted 
    and true classes"""
    diff = predicted - truth
    errors = np.count_nonzero(diff)
    return errors / float(len(truth))

class ParameterMerger(object):
    """Merges and unmerges parameter vectors for optimizers.
    Do not use. Use breze.util.ParameterSet instead."""

    def __init__(self, *args):
        self.shapes = [arg.shape for arg in args]

        self.total_size = 0
        for s in self.shapes:
            l = reduce(mul, s)
            self.total_size += l

    def make_func(self, f):
        def wrapper(merged):
            args = self.unflatten(merged)
            ret = f(*args)
            if isinstance(ret, (list, tuple)):
                return self.flatten(*ret)
            elif ret.size > 1:
                return self.flatten(ret)
            else:
                return ret
            return self.flatten(*ret)
        return wrapper

    def flatten(self, *args):
        assert len(args) == len(self.shapes), "wrong number of args"

        merged = np.zeros((self.total_size,))
        pos = 0
        for arg, s in zip(args, self.shapes):
            assert arg.shape == s, "argument shape %s does not equal %s" % \
                (arg.shape, s)
            l = reduce(mul, s)
            merged[pos : pos+l] = np.reshape(arg, (l,))
            pos += l
        return merged

    def unflatten(self, merged):
        pos = 0
        args = []
        for s in self.shapes:
            l = reduce(mul, s)
            arg = np.reshape(merged[pos : pos+l], s)
            args.append(arg)
            pos += l
        return tuple(args)


class ParameterHistory(object):
    """Keeps track of parameter history, corresponding loses and optimization
    termination criteria."""

    def __init__(self, max_missed_val_improvements=200, show_progress=True,
                 desired_loss=None, min_improvement=0.00001, max_iters=None,
                 min_iters=None):
        self.max_missed_val_improvements = max_missed_val_improvements
        self.show_progress = show_progress
        self.desired_loss = desired_loss
        self.min_improvement = min_improvement
        self.max_iters = max_iters
        self.min_iters = min_iters

        self.best_val_loss = float('inf')
        self.history = np.zeros((4, 0))
        self.last_val_improvement = 0
        self.should_terminate = False
        self.start_time = time.time()
        self.best_iter = None

        self.reset_best()

    def reset_best(self):
        self.best_val_loss = float('inf')
        self.last_val_improvement = 0
        self.should_terminate = False
        self.best_iter = None

    def add(self, iter, pars, trn_loss, val_loss, tst_loss):
        # keep track of best results so far
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_iter = iter
            self.best_val_loss = val_loss
            self.best_tst_loss = tst_loss
            if isinstance(pars, gp.garray):
                self.best_pars = gp.garray(pars, copy=True)
            else:
                self.best_pars = np.copy(pars)
            self.last_val_improvement = iter

        # termination criteria
        if (self.max_missed_val_improvements is not None and 
            iter - self.last_val_improvement > self.max_missed_val_improvements):
            self.should_terminate = True
        if self.min_iters is not None and iter < self.min_iters:
            self.should_terminate = False
        if self.desired_loss is not None and val_loss <= self.desired_loss:
            self.should_terminate = True
        if self.max_iters is not None and iter >= self.max_iters:
            self.should_terminate = True

        # store current losses
        self.history = np.hstack((self.history, [[iter],
                                                 [trn_loss], 
                                                 [val_loss], 
                                                 [tst_loss]]))

        # display progress
        if self.show_progress:
            progress.status(iter, caption=
                            "training: %9.5f  validation: %9.5f (best: %9.5f)  test: %9.5f" %
                            (trn_loss, val_loss, self.best_val_loss, tst_loss))

        # termination by user
        if get_key() == "q":
            print
            print "Termination by user."
            self.should_terminate = True

    def plot(self, final=True, logscale=True):
        self.end_time = time.time()

        if 'figsize' in dir(plt):
            plt.figsize(10,5)
        # plt.clf()
        plt.hold(True)
        if logscale:
            plt.yscale('log')
            #plt.xscale('log')
        plt.plot(self.history[0], self.history[1], 'b')
        plt.plot(self.history[0], self.history[2], 'c')
        plt.plot(self.history[0], self.history[3], 'r')
        yl = plt.ylim()
        if self.best_iter is not None:
            plt.vlines(self.best_iter, yl[0], yl[1])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(['training', 'validation', 'test'])

        if final and self.best_iter is not None:
            print "best iteration: %5d  best validation test loss: %9.5f  best test loss: %9.5f" % \
                (self.best_iter, self.best_val_loss, self.best_tst_loss)
            print "training took %.2f s" % (self.end_time - self.start_time)

    @property
    def performed_iterations(self):
        return np.max(self.history[0])

    @property
    def converged(self):
        return self.best_val_loss <= self.desired_loss + self.min_improvement


def load_theano_data(filename):
    """Loads a .mat file into Theano shared variables"""
    dat = scipy.io.loadmat(filename)
    RX = theano.shared(floatx(dat['RX']))
    RZ = theano.shared(floatx(dat['RZ']))
    VX = theano.shared(floatx(dat['VX']))
    VZ = theano.shared(floatx(dat['VZ']))
    TX = theano.shared(floatx(dat['TX']))
    TZ = theano.shared(floatx(dat['TZ']))
    return RX, RZ, VX, VZ, TX, TZ


class ValueIter(object):
    """Maps values (learning rates, etc.) to iterations."""

    def __init__(self, iters, values, 
                 transition='hard', transition_length=1000):
        self.iters = iters
        self.values = values
        self.transition = transition
        self.transition_length = transition_length

    def find_iter_index(self,iter):
        for i in range(len(self.iters)):
            if self.iters[i] > iter:
                return i-1
        return len(self.iters)-1

    def value_for_iter(self, current_iter):
        """Returns value[i] where i is the smallest i so that iters[0], ...,
        iters[i] are all smaller than or equal to current_iter.
        If iters is None than values is returned, i.e. it is assumed that the same
        value should be used for all iterations."""
        if self.iters is None:
            return self.values

        if self.transition == 'hard':
            return self.values[self.find_iter_index(current_iter)]
        elif self.transition == 'linear':
            cn = self.find_iter_index(current_iter)
            ifp = current_iter - self.iters[cn]
            if ifp < self.transition_length and cn > 0:
                f = float(ifp) / float(self.transition_length)
                return f*self.values[cn] + (1-f)*self.values[cn-1]
            else:
                return self.values[cn]

    def __getitem__(self, key):
        return self.value_for_iter(key)


def get_key():
    if sys.platform == 'nt':
        if msvcrt.kbhit():
            return msvcrt.getch()
        else:
            return None
    else:
        return None


class CheckpointHandler(object):
    def __init__(self, directory, filename="checkpoint.dat"):
        self._path = os.path.join(directory, filename)
        self._directory = directory
        self._requested = False

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._requested = True

    @staticmethod
    def _replace_file(src, dest):
        if sys.platform == 'win32':
            if os.path.exists(dest):
                os.remove(dest)
            assert not os.path.exists(dest), "%s still exists after deleting it" % dest
        os.rename(src, dest)

    @property
    def requested(self):
        return self._requested

    def save(self, **kwargs):
        explicit = False
        if 'explicit' in kwargs:
            explicit = kwargs['explicit']

        if self._requested:
            print "Saving checkpoint %s" % self._path
        if self._requested or explicit:
            with open(self._path + ".tmp", 'wb') as f:
                pickle.dump(kwargs, f, -1)
            self._replace_file(self._path + ".tmp", self._path)
        if self._requested:
            print "Checkpoint saved. Exiting."
            sys.exit(9)

    def load(self):
        print "Loading checkpoint %s" % self._path
        with open(self._path, 'rb') as f:
            return pickle.load(f)

    def remove(self):
        if os.path.exists(self._path):
            os.remove(self._path)



