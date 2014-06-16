
from itertools import repeat
import numpy as np
import os
import h5py
import glob
import math
import sys
import warnings

if os.name == 'nt':
    default_directory = r"\\brml.tum.de\dfs\nthome\surban\dev\indentor\indentor\apps\out"
else:
    default_directory = r"/nthome/surban/dev/indentor/indentor/apps/out"


class SkinDataset(object):

    num_taxels = 8

    testset_ratio = 0.1
    valiset_ratio = 0.1
    extenstion = ".skn"

    def __init__(self, filename=None, directory=None):
        assert filename is None or directory is None
        if filename:
            if "." not in filename:
                filename += self.extenstion
            if not os.path.exists(filename):
                filename = os.path.join(default_directory, filename)
            self._storage = h5py.File(filename, 'r')
        elif directory:
            directory = os.path.abspath(directory)
            parent, name = os.path.split(directory)
            filename = os.path.join(parent, name + self.extenstion)
            self._storage = h5py.File(filename, 'w')
            self._build(directory)

    def _group_path(self, purpose, taxel):
        if purpose == 'trn': purpose = 'train'
        if purpose == 'val': purpose = 'validation'
        if purpose == 'tst': purpose = 'test'
        assert purpose in ['train', 'validation', 'test']
        path = '%s/%d,%d' % (purpose, taxel[0], taxel[1])
        return path

    def group(self, purpose, taxel):
        return self._storage[self._group_path(purpose, taxel)]

    def _record_path(self, purpose, taxel, record):
        return self._group_path(purpose, taxel) + '/' + str(record)

    def record(self, purpose, taxel, record=None):
        if record is None:
            record = range(self.record_count(purpose, taxel))

        if isinstance(record, list):
            return [self._storage[self._record_path(purpose, taxel, r)] for r in record]
        else:
            return self._storage[self._record_path(purpose, taxel, record)]

    def available_taxels(self):
        grp = self._storage['train']
        taxels = []
        for taxel_str in grp.iterkeys():
            x, y = taxel_str.split(',')
            taxels.append((int(x), int(y)))
        return taxels

    def record_count(self, purpose, taxel):
        return len(self.group(purpose, taxel))

    @property
    def timestep(self):
        return self._storage['metadata'].attrs['interval']

    def _build(self, directory):
        metadata = self._storage.create_group('metadata')

        interval = None
        freqs = 0

        for x in range(self.num_taxels):
            for y in range(self.num_taxels):
                taxel = (x, y)

                data_dir = os.path.join(directory, "%d%d" % (x, y))
                if not os.path.isdir(data_dir):
                    continue

                datas, my_interval = self._load_directory(data_dir)
                n_curves = len(datas)
                if 'skin_freqs' in datas[0]:
                    my_freqs = datas[0]['skin_freqs']
                else:
                    my_freqs = None

                if freqs == 0:
                    freqs = my_freqs
                    metadata.attrs['freqs'] = freqs
                else:
                    if freqs != my_freqs:
                        print "Taxel %s has different frequencies (%s) than other taxels (%s)" % \
                              (str(taxel), str(my_freqs), str(freqs))

                if not interval:
                    interval = my_interval
                    metadata.attrs['interval'] = interval
                else:
                    if not abs(my_interval - interval) < 0.0001:
                        print "Taxel %s has time step (%g s) that is different from other taxels (%g s)" % \
                            (str(taxel), my_interval, interval)

                testset_size = int(n_curves * self.testset_ratio)
                valiset_size = int(n_curves * self.valiset_ratio)
                trngset_size = n_curves - testset_size - valiset_size

                self._storage.create_group(self._group_path('train', taxel))
                self._storage.create_group(self._group_path('validation', taxel))
                self._storage.create_group(self._group_path('test', taxel))

                datas = self._assign_to_dataset(datas, 'train', taxel, trngset_size, freqs)
                datas = self._assign_to_dataset(datas, 'validation', taxel, valiset_size, freqs)
                datas = self._assign_to_dataset(datas, 'test', taxel, testset_size, freqs)
                assert datas == []

    def _assign_to_dataset(self, datas, purpose, taxel, count, freqs):
        for i in range(count):
            n_samples = datas[i]['force'].shape[0]
            if freqs is None:
                ds = self._storage.create_dataset(self._record_path(purpose, taxel, i),
                                                  shape=(2, n_samples), dtype='f')
                ds[0, :] = datas[i]['force']
                ds[1, :] = datas[i]['skin']
            else:
                ds = self._storage.create_dataset(self._record_path(purpose, taxel, i),
                                                  shape=(1 + 2*len(freqs), n_samples), dtype='f')
                ds[0, :] = datas[i]['force'][i]
                for f in range(len(freqs)):
                    ds[1 + 2*f] = datas[i]['skin_amp'][f, :]
                    ds[1 + 2*f + 1] = datas[i]['skin_phase'][f, :]

        return datas[count:]

    def _load_directory(self, directory):
        data = []
        deltat = None

        files = glob.glob(os.path.join(directory, "*.fsd.npz"))
        files = sorted(files)
        for filename in files:
            try:
                d = np.load(filename)
                data.append(d)
                dt = np.diff(d['time'])
            except Exception as e:
                print "Error loading %s: %s" % (filename, str(e))
                continue

            if not np.all(abs(dt - dt[0]) < 0.0001):
                print "Curve %s has non-constant time step, using deltat=%g s" % \
                    (filename, dt[0])
            if not deltat:
                deltat = dt[0]
            else:
                if not abs(deltat - dt[0]) < 0.0001:
                    print "Curve %s has time step (%g s) that is different from other curves (%g s)" % \
                        (filename, dt[0], deltat)

        return data, deltat

    def print_statistics(self):
        print "Dataset %s:" % self._storage.filename
        print "     taxel          train   validation   test"
        for taxel in self.available_taxels():
            trng_smpls = self.record_count('train', taxel)
            vali_smpls = self.record_count('validation', taxel)
            test_smpls = self.record_count('test', taxel)
            print "     %d,%d           %4d    %4d         %4d" % (taxel[0], taxel[1],
                                                                   trng_smpls, vali_smpls, test_smpls)

        n_points = 0
        n_records = 0
        for taxel in self.available_taxels():
            for i in range(self.record_count('train', taxel)):
                n_records += 1
                n_points += self.record('train', taxel, i).shape[1]
        points_per_record = n_points / n_records
        print "Avg. datapoints per record: %d" % points_per_record
        print "Sampling interval:          %g s" % self.timestep
        print




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: %s <dataset directory>" % sys.argv[0]
        sys.exit(1)
    directory = sys.argv[1]
    print "Building dataset from %s" % directory
    ds = SkinDataset(directory=directory)
    print
    ds.print_statistics()

