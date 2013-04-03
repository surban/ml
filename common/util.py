# -*- coding: utf-8 -*-

import os
import sys
import glob
import math
import fractions
import numpy as np
import gnumpy as gp
import matplotlib.pyplot as plt

import common.progress as progress

try:    
    import IPython.core.display
    have_notebook = True
except ImportError:
    have_notebook = False


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
        samples_are='rows', caption=""):
    if isinstance(X, gp.garray):
        xp = gp
    else:
        xp = np
    ms = None
    for b, x in enumerate(draw_slices(X, batch_size, kind='sequential', 
                                      samples_are=samples_are, stop=True, 
                                      last_batch_may_be_smaller=True)):
        progress.status(b, X.shape[0] / batch_size, caption)

        m = map_func(x)
        if ms is None:
            if m.ndim == 1:
                ms = xp.zeros((X.shape[0],))
            elif m.ndim == 2:
                if samples_are == 'rows':
                    ms = xp.zeros((X.shape[0], m.shape[1]))
                elif samples_are == 'columns':
                    ms = xp.zeros((m.shape[0], X.shape[1]))
                else:
                    assert False
            else:
                assert False
        
        if ms.ndim == 1:
            ms[b*batch_size : (b+1)*batch_size] = m
        elif ms.ndim == 2:
            if samples_are == 'rows':
                ms[b*batch_size : (b+1)*batch_size, :] = m
            elif samples_are == 'columns':
                ms[:, b*batch_size : (b+1)*batch_size] = m 

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

