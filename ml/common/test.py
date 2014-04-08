import numpy as np

epsilon = 1e-6
default_tolerance = 1e-2


def check_gradient(func, grad_func, x, always_output=False, columns_are_samples=False, tolerance=default_tolerance):
    assert x.ndim == 1
    n_directions = x.shape[0]

    if not columns_are_samples:
        def func_wrapper(f, xi):
            if xi.ndim == 2:
                retvals = [f(xi[:, i]) for i in range(xi.shape[1])]
                return np.asarray(retvals)
            else:
                return f(xi)
        orig_func = func
        func = lambda xi: func_wrapper(orig_func, xi)

    xb = np.tile(x, (n_directions, 1)).T
    xe = xb.copy()
    for d in range(n_directions):
        xe[d, d] += epsilon

    num_grad = (func(xe) - func(xb)) / epsilon
    sym_grad = grad_func(x)

    if always_output or not np.all(np.abs(num_grad - sym_grad) < tolerance):
        print "checking gradient at x="
        print x
        print "func(x)=", func(x)
        print "func(x + dx)=", func(xe)
        print "(func(x + dx) - func(x)) / epsilon="
        print num_grad
        print "grad_func(x)="
        print sym_grad

    assert np.all(np.abs(num_grad - sym_grad) < tolerance)


def check_directional_gradient(func, grad_func, x, direction=None, always_output=False, tolerance=default_tolerance):
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

        print "checking gradient at x="
        print x
        print "with dx="
        print dx

        print "func(x)=", func(x)
        print "func(x + dx)=", func(x + dx)
        print "grad_func(x)="
        print gfx

        print "(func(x + dx) - func(x)) / epsilon="
        print num_grad
        print "dx * grad(func)(x) / epsilon="
        print sym_grad

    assert np.all(np.abs(num_grad - sym_grad) < tolerance)



