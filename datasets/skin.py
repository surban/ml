from itertools import repeat
import numpy as np
import os
import h5py
import glob
import math
import sys


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
            self._storage = h5py.File(filename, 'r')
        elif directory:
            directory = os.path.abspath(directory)
            _, name = os.path.split(directory)
            filename = os.path.join(directory, name + self.extenstion)
            self._storage = h5py.File(filename, 'w')
            self._build(directory)

    def _group_path(self, purpose, taxel):
        assert purpose in ['train', 'validation', 'test']
        path = '%s/%d,%d' % (purpose, taxel[0], taxel[1])
        self._storage.require_group(path)
        return path

    def group(self, purpose, taxel):
        return self._storage[self._group_path(purpose, taxel)]

    def _record_path(self, purpose, taxel, record):
        return self._group_path(purpose, taxel) + '/' + str(record)

    def available_taxels(self):
        grp = self._storage['train']
        taxels = []
        for taxel_str in grp.iterkeys():
            x, y = taxel_str.split(',')
            taxels.append((int(x), int(y)))
        return taxels

    def _build(self, directory):
        n_curves = None

        for x in range(self.num_taxels):
            for y in range(self.num_taxels):
                taxel = (x,y)

                data_dir = os.path.join(directory, "%d%d" % (x, y))
                if not os.path.isdir(data_dir):
                    continue

                forces, skins, interval = self._load_directory(data_dir)
                if not n_curves:
                    n_curves = len(forces)
                else:
                    assert n_curves == len(forces)

                testset_size = math.floor(n_curves * self.testset_ratio)
                valiset_size = math.floor(n_curves * self.valiset_ratio)
                trngset_size = n_curves - testset_size - valiset_size

                dest = []
                j = 0
                for i in range(trngset_size):
                    ds = self._storage.create_dataset(self._record_path('train', taxel, i),
                                                      shape=(2, forces[j].shape[0]), dtype='f')
                    ds[0, :] = forces[j]
                    ds[1, :] = skins[j]
                    j += 1
                for i in range(valiset_size):
                    ds = self._storage.create_dataset(self._record_path('validation', taxel, i),
                                                      shape=(2, forces[j].shape[0]), dtype='f')
                    ds[0, :] = forces[j]
                    ds[1, :] = skins[j]
                    j += 1
                for i in range(testset_size):
                    ds = self._storage.create_dataset(self._record_path('train', taxel, i),
                                                      shape=(2, forces[j].shape[0]), dtype='f')
                    ds[0, :] = forces[j]
                    ds[1, :] = skins[j]
                    j += 1

    def _load_directory(self, directory):
        forces = []
        skins = []
        deltat = None

        files = glob.glob(os.path.join(directory, "*.fsd.npz"))
        files = sorted(files)
        for filename in files:
            d = np.load(filename)
            forces.append(d['force'])
            skins.append(d['skin'])

            dt = np.diff(d['time'])
            assert np.all(dt == dt[0])
            if not deltat:
                deltat = dt
            else:
                assert deltat == dt

        return forces, skins, deltat



if __name__ == '__main__':
    directory = sys.argv[1]
    print "Building dataset from %s" % directory
    ds = SkinDataset(directory=directory)
    print "Dataset contains taxels: ", ds.available_taxels()


