# -*- coding: utf-8 -*-

import rbm.util

def test_all_states():
    size = 4
    l = 0
    for state in rbm.util.all_states(4):
        print state
        l += 1
    assert l == 2**size


