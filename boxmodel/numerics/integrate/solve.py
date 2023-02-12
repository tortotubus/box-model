from ._ivp import (OdeSolver, RKF45)

import numpy as np

METHODS = {'RKF45': RKF45}

MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}

def _indenter(s, n=0):
    """
    Ensures that lines after the first are indented by the specified amount
    """
    split = s.split("\n")
    indent = " "*n
    return ("\n" + indent).join(split)

def _float_formatter_10(x):
    """
    Returns a string representation of a float with exactly ten characters
    """
    if np.isposinf(x):
        return "       inf"
    elif np.isneginf(x):
        return "      -inf"
    elif np.isnan(x):
        return "       nan"
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)

def _dict_formatter(d, n=0, mplus=1, sorter=None):
    """
    Pretty printer for dictionaries
    `n` keeps track of the starting indentation;
    lines are indented by this much after a line break.
    `mplus` is additional left padding applied to keys
    """
    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                       _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
                       for k, v in sorter(d)])  # +2 for ': '
    else:
        # By default, NumPy arrays print with linewidth=76. `n` is
        # the indent at which a line begins printing, so it is subtracted
        # from the default to avoid exceeding 76 characters total.
        # `edgeitems` is the number of elements to include before and after
        # ellipses when arrays are not shown in full.
        # `threshold` is the maximum number of elements for which an
        # array is shown in full.
        # These values tend to work well for use with OptimizeResult.
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)
    return s

class OdeResult(dict):
    """ 
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    `OptimizeResult` may have additional attributes not listed here depending
    on the specific solver being used. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin']
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {'slack', 'con', 'crossover_nit'}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def solve_ivp(fun, t_span, y0: float, method='RKF45', t_eval=None, atol=1e-6, max_step=1e10, min_step=0., args=None, **options):

    """
    """

    if method not in METHODS:
        raise ValueError("`method` must be one of {} or OdeSolver class.".format(METHODS))
    
    t0, tf = map(float, t_span)

    if args is not None:
        pass

    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, t0, tf, y0, atol, max_step, min_step)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    else:
        ts = []
        ys = []

    status = None

    while status is None:
        solver.step()

        y = solver.get_y()
        t = solver.get_t()

        if t_eval is None:
            ys.append(y)
            ts.append(t)

        if solver.is_complete():
            status = 0
        elif solver.has_failed():
            status = -1
            break

    message = MESSAGES.get(status)

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    sol = None

    return OdeResult(t=ts, y=ys, sol=sol, status=status, message=message, success=status >= 0)

