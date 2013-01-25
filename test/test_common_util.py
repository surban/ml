import rbm.util
import common.util

def test_pack_in_batches():
    for batch in common.util.pack_in_batches(rbm.util.all_states(5), 3):
        print batch

def test_myrand():
    common.util.myrand.seed(0)

    print "uint32s:"
    for i in range(10):
        print common.util.myrand.get_uint32()

    print "floats:"
    for i in range(10):
        print "%.20f" % common.util.myrand.get_float()
