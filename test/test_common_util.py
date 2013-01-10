import rbm.util
import common.util

def test_pack_in_batches():
    for batch in common.util.pack_in_batches(rbm.util.all_states(5), 3):
        print batch


