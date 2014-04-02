import numpy as np


def check_gradient(func, grad_func, x, direction=None, always_output=False):
    epsilon = 0.0001
    tolerance = 0.01

    if direction is None:
        direction = np.random.random_integers(0, 1, size=x.shape)
    for s in range(direction.shape[1]):
        if np.sum(direction[:, s]) < 0.001:
            direction[0, s] = 1
    assert direction.shape == x.shape
    dx = epsilon * direction

    # print "dx:", dx, dx.shape
    # print "grad_func:", grad_func(x), grad_func(x).shape

    num_grad = (func(x + dx) - func(x)) / epsilon
    sym_grad = np.sum(dx * grad_func(x), axis=0) / epsilon

    if always_output or not np.all(np.abs(num_grad - sym_grad) < tolerance):
        gfx = grad_func(x)

        print "checking gradient failed at x="
        print x
        print "with dx="
        print dx

        print "func(x)=", func(x)
        print "func(x + dx)=", func(x + dx)
        print "(func(x + dx) - func(x)) / epsilon="
        print (func(x + dx) - func(x)) / epsilon
        print "grad_func(x)="
        print gfx

        print "numeric gradient:"
        print num_grad
        print "symbolic gradient:"
        print sym_grad

    assert np.all(np.abs(num_grad - sym_grad) < tolerance)

