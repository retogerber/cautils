import numba
import numpy as np
import scipy
import skimage
from numba.experimental import jitclass

CONNECTIVITY_ROOK = 1
CONNECTIVITY_QUEEN = 2

GPVAL_TYPE_HOT = 1
GPVAL_TYPE_COLD = 2
GPVAL_TYPE_BOTH = 4


KERNEL_WEIGHT_NONE = 0
KERNEL_WEIGHT_EXP = 1

@numba.njit(parallel=False, cache=True)
def get_circular_kernel(distance, include_center=True, weight=KERNEL_WEIGHT_NONE):
    diameter = int(np.floor(distance) * 2 + 1)
    mid = (diameter - 1) // 2
    distancesxy = np.indices((diameter, diameter)) - np.array([mid, mid])[:, np.newaxis, np.newaxis]
    kernel = np.zeros((diameter, diameter), dtype=np.float64)
    for i in range(diameter):
        for j in range(diameter):
            tmpdist = np.sqrt(distancesxy[0, i, j] ** 2 + distancesxy[1, i, j] ** 2)
            if weight == KERNEL_WEIGHT_EXP:
                if tmpdist <= distance:
                    kernel[i, j] = np.exp(-tmpdist)
                else:
                    kernel[i, j] = 0
            else:
                kernel[i, j] = tmpdist <= distance
    if not include_center:
        kernel[mid, mid] = 0
    return kernel

# get_circular_kernel(1, weight=KERNEL_WEIGHT_EXP)
# KERNEL_QUEEN = get_circular_kernel(1.5)
# KERNEL_ROOK2 = get_circular_kernel(2)
# KERNEL_ROOK4 = get_circular_kernel(2.5)
# KERNEL_QUEEN2 = get_circular_kernel(2.9)


spec = [
    ('include_center', numba.boolean),
    ('diameter', numba.int32),
    ('distance', numba.float64),
    ('kernel', numba.float64[:,:]),
    ('w', numba.float64),
    ('n', numba.uint64),
    ('offsets', numba.int64[:,:]),
    ('kernel_flat', numba.float64[:])
]
@jitclass(spec)
class Kernel:
    def __init__(self, distance, include_center=True, weight=KERNEL_WEIGHT_EXP, normalize=False):
        self.include_center = include_center
        self.distance = distance
        self.diameter = int(np.floor(distance) * 2 + 1)
        self.kernel = get_circular_kernel(self.distance, self.include_center, weight=weight)
        w = np.sum(self.kernel)
        if normalize:
            if w > 0:
                self.kernel /= w
            else:
                self.kernel = np.zeros_like(self.kernel)
        self.w = np.sum(self.kernel)
        self.n = np.sum(self.kernel>0)
        indices = np.indices(self.kernel.shape)# - np.array([self.kernel.shape[0] // 2, self.kernel.shape[1] // 2])[:, np.newaxis, np.newaxis]
        self.offsets = np.zeros((self.n, 2), dtype=np.int64)
        self.kernel_flat = np.zeros(self.n, dtype=np.float64)
        idx = 0
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                if self.kernel[i, j]:
                    self.offsets[idx] = indices[:, i, j]
                    self.kernel_flat[idx] = self.kernel[i, j]
                    idx += 1

# ker = Kernel(2.5, include_center=False, weight=KERNEL_WEIGHT_EXP, normalize=True)
# ker.kernel


@numba.njit(parallel=False, cache=True)
def man_pad(x, distance=1):
    xm = np.zeros((x.shape[0] + 2*distance, x.shape[1] + 2*distance), dtype=x.dtype)
    xm[distance:-distance, distance:-distance] = x

    for d in range(distance):
        xm[distance:-distance, d] = x[:, 0]
        xm[distance:-distance, -d-1] = x[:, -1]
        xm[d, distance:-distance] = x[0, :]
        xm[-d-1, distance:-distance] = x[-1, :]
    xm[:distance, :distance] = x[0, 0]
    xm[:distance, -distance:] = x[0, -1]
    xm[-distance:, :distance] = x[-1, 0]
    xm[-distance:, -distance:] = x[-1, -1]
    return xm
# man_pad(np.random.rand(5, 5), distance=2)

@numba.njit(parallel=True, cache=True)
def neighborhood_sum(x, kernel: Kernel, xp=None):
    if xp is None:
        xp = np.copy(x)
        for i in range(kernel.diameter//2):
            xp = man_pad(xp)

    border_offset = kernel.diameter // 2 * 2
    xcomb = np.zeros_like(x, dtype=np.float64)
    for i in range(kernel.n):
        xcomb += xp[kernel.offsets[i, 0]:kernel.offsets[i, 0] + xp.shape[0]-border_offset, kernel.offsets[i, 1]:kernel.offsets[i, 1] + xp.shape[1]-border_offset] * kernel.kernel_flat[i]
    return xcomb

# x = np.random.rand(100, 100)
# xp = man_pad(man_pad(x))
# ker = Kernel(2, False)
# neighborhood_sum(x, ker, xp=xp)


@numba.njit(parallel=True, cache=True)
def G_classical(
        x, 
        connectivity=CONNECTIVITY_QUEEN, 
        normalize=True,
        Gstar=True
    ):
    x_sum = np.float64(np.sum(x))

    # neighborhood sums
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xns = (
            xp[:-2, 1:-1]
            + xp[1:-1, :-2]
            + xp[2:, 1:-1]
            + xp[1:-1, 2:]
            + xp[:-2, :-2]
            + xp[:-2, 2:]
            + xp[2:, :-2]
            + xp[2:, 2:]
        )
    else:
        xns = (
            xp[:-2, 1:-1] + xp[1:-1, :-2] + xp[2:, 1:-1] + xp[1:-1, 2:]
        )
    if normalize:
        if Gstar:
            xns = xns+x
            w1 = 4 * connectivity + 1
            n = np.float64(x.size)
            x_mean = x_sum / n
            x2_sum = np.sum(x**2)
            s2 = x2_sum / n - x_mean**2
            Gout = (xns - x_mean * w1) / np.sqrt(s2 * ((n * w1 - w1**2) / (n - 1)))
        else:
            w1 = 4 * connectivity
            n = np.float64(x.size)-1
            x_mean = (x_sum-x) / n
            x2 = x**2
            x2_sum = np.sum(x2)- x2
            s2 = x2_sum / n - x_mean**2
            Gout = (xns - x_mean * w1) / np.sqrt(s2 * ((n * w1 - w1**2) / (n - 1)))
    else:
        if Gstar:
            Gout = (xns+x) / x_sum
        else:
            Gout = xns / (x_sum-x)
    return Gout



@numba.njit(parallel=True, cache=True)
def G_classical_new(
        x, 
        kernel=Kernel(1.5, include_center=True, weight=KERNEL_WEIGHT_EXP, normalize=True),
        normalize=True
    ):
    x_sum = np.float64(np.sum(x))

    # neighborhood sums
    xp = man_pad(x, kernel.diameter // 2)
    xns = neighborhood_sum(x, kernel, xp=xp)
    w1 = kernel.w
    G_indicator = int(not kernel.include_center)
    if normalize:
        n = np.float64(x.size)-G_indicator
        x_mean = (x_sum-x*G_indicator) / n
        x2 = x**2
        x2_sum = np.sum(x2)- x2*G_indicator
        s2 = x2_sum / n - x_mean**2
        Gout = (xns - x_mean * w1) / np.sqrt(s2 * ((n * w1 - w1**2) / (n - 1)))
    else:
        Gout = xns / (x_sum - x * G_indicator)
    return Gout

# x = np.random.rand(100, 100)
# x[:2,:2] = 100
# out = G_classical(x, connectivity=CONNECTIVITY_QUEEN, normalize=True, Gstar=True)
# out2 = G_classical_new(x, kernel=Kernel(1.9, include_center=True, weight=KERNEL_WEIGHT_NONE, normalize=False))

# out[:5,:5]
# out2[:5,:5]
# np.all(out-out2< 1e-10)


# x = np.random.rand(1000, 1000)
# import timeit
# import time

# timeit.timeit(lambda: G_classical(x, connectivity=CONNECTIVITY_QUEEN, normalize=True, Gstar=True), number=100)
# time.sleep(2)
# kernel = Kernel(1.9, include_center=True, weight=KERNEL_WEIGHT_NONE, normalize=False)
# timeit.timeit(lambda: G_classical_new(x, kernel=kernel), number=100)

# np.max(out)
# np.min(out)


@numba.njit(cache=True)
def split_GPtype(GPtype):
    # decompose GPtype into its components
    which_test = np.zeros(3, dtype=np.uint8)
    tmpGPtype = GPtype
    for i, t in enumerate([GPVAL_TYPE_BOTH, GPVAL_TYPE_COLD, GPVAL_TYPE_HOT]):
        which_test[i] = tmpGPtype // t
        tmpGPtype -= (tmpGPtype // t) * t
    return which_test

@numba.njit(parallel=True, cache=True)
def G_permutation(
    x, connectivity=CONNECTIVITY_QUEEN, n_iter=99, seed=42, GPtype=GPVAL_TYPE_BOTH, Gstar=True
):
    """
    Calculate the G statistic and its p-value for a given input array x using permutation testing.

    Parameters
    ----------
    x : np.ndarray
        Input array for which the G statistic and p-value are calculated.
    connectivity : int, optional
        Connectivity type for neighborhood calculation. Default is CONNECTIVITY_QUEEN.
    n_iter : int, optional
        Number of iterations for permutation testing. Default is 99.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    GPtype : int, optional
        Type of G statistic to calculate. Can be either GPVAL_TYPE_BOTH, GPVAL_TYPE_COLD, or GPVAL_TYPE_HOT, or the sum of any of those three. For example GPVAL_TYPE_HOT+GPVAL_TYPE_COLD will return both hot and cold p-values.
        Default is GPVAL_TYPE_BOTH.
    Returns
    -------
    Gi : np.ndarray
        G statistic for the input array x.
    GPi : np.ndarray
        P-value for the G statistic.
    """
    np.random.seed(seed)
    x_sum = np.float64(np.sum(x))

    # neighborhood sums
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xns = (
            xp[:-2, 1:-1]
            + xp[1:-1, :-2]
            + xp[2:, 1:-1]
            + xp[1:-1, 2:]
            + xp[:-2, :-2]
            + xp[:-2, 2:]
            + xp[2:, :-2]
            + xp[2:, 2:]
        )
    else:
        xns = xp[:-2, 1:-1] + xp[1:-1, :-2] + xp[2:, 1:-1] + xp[1:-1, 2:]
    if Gstar:
        Gi = (xns+x) / x_sum
    else:
        Gi = xns / (x_sum-x)

    # decompose GPtype into its components
    which_test = split_GPtype(GPtype)

    # number of neighbors to sample
    n_samples = 8 if connectivity == CONNECTIVITY_QUEEN else 4

    # calculate the probability of duplicate neighbors (birthday problem approximation)
    p_duplicate = 1 - np.exp(-(4*connectivity)**2/(2*x.size))
    p_center_pixel = 1 / x.size
    expected_n_invalid = n_iter*p_duplicate + n_iter*p_center_pixel
    do_test_invalid = expected_n_invalid > 0.5
    n_additional = np.ceil(n_iter*p_duplicate + n_iter*p_center_pixel)+5
    xrindsarr = np.random.randint(0, x.shape[0], (x.shape[0], x.shape[1], int(n_iter * n_samples + n_additional)))
    yrindsarr = np.random.randint(0, x.shape[1], (x.shape[0], x.shape[1], int(n_iter * n_samples + n_additional)))

    perm_G_counts = np.zeros((3, Gi.shape[0], Gi.shape[1]), dtype=np.uint16)
    for xi in numba.prange(x.shape[0]):
        for yi in numba.prange(x.shape[1]):
           # permute over neighbors
            neighbor_sums = np.zeros((n_iter), dtype=np.float64)
            currind = 0
            for it in numba.prange(n_iter):
                for nei in range(n_samples):
                    if do_test_invalid:
                        while xrindsarr[xi, yi, currind] == xi and yrindsarr[xi, yi, currind] == yi:
                            # if the center pixel is selected as a neighbor, skip it
                            currind += 1
                    # sum over neighbors
                    neighbor_sums[it] += x[xrindsarr[xi,yi,currind], yrindsarr[xi,yi,currind]]
                    currind += 1

            permGi = (x[xi, yi]*Gstar + neighbor_sums) / (x_sum-x[xi,yi]*Gstar)
            # compare with observed Gi
            perm_G_counts[0, xi, yi] = np.sum(np.abs(permGi) > np.abs(Gi[xi, yi]))
            perm_G_counts[1, xi, yi] = np.sum(permGi < Gi[xi, yi])
            perm_G_counts[2, xi, yi] = np.sum(permGi > Gi[xi, yi])
    # only return the requested p-values
    GPi = (1 + perm_G_counts[np.where(which_test)[0], :, :]) / (n_iter + 1)
    return Gi, GPi

@numba.njit(parallel=True, cache=True)
def G_permutation_fast(
    x, connectivity=CONNECTIVITY_QUEEN, n_iter=99, seed=42, GPtype=GPVAL_TYPE_BOTH, Gstar=True
):
    """
    Calculate the G statistic and its p-value for a given input array x using permutation testing.

    Parameters
    ----------
    x : np.ndarray
        Input array for which the G statistic and p-value are calculated.
    connectivity : int, optional
        Connectivity type for neighborhood calculation. Default is CONNECTIVITY_QUEEN.
    n_iter : int, optional
        Number of iterations for permutation testing. Default is 99.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    GPtype : int, optional
        Type of G statistic to calculate. Can be either GPVAL_TYPE_BOTH, GPVAL_TYPE_COLD, or GPVAL_TYPE_HOT, or the sum of any of those three. For example GPVAL_TYPE_HOT+GPVAL_TYPE_COLD will return both hot and cold p-values.
        Default is GPVAL_TYPE_BOTH.
    Returns
    -------
    Gi : np.ndarray
        G statistic for the input array x.
    GPi : np.ndarray
        P-value for the G statistic.
    """
    np.random.seed(seed)
    x_sum = np.float64(np.sum(x))

    # neighborhood sums
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xns = (
            xp[:-2, 1:-1]
            + xp[1:-1, :-2]
            + xp[2:, 1:-1]
            + xp[1:-1, 2:]
            + xp[:-2, :-2]
            + xp[:-2, 2:]
            + xp[2:, :-2]
            + xp[2:, 2:]
        )
    else:
        xns = xp[:-2, 1:-1] + xp[1:-1, :-2] + xp[2:, 1:-1] + xp[1:-1, 2:]
    if Gstar:
        Gi = (xns+x) / x_sum
    else:
        # remove the center pixel from the total sum
        Gi = xns / (x_sum-x)

    # decompose GPtype into its components
    which_test = split_GPtype(GPtype)

    perm_G_counts = np.zeros((6, Gi.shape[0], Gi.shape[1]), dtype=np.uint16)
    for rep in numba.prange(n_iter):
        # random permutation of x
        xrp_test = man_pad(np.random.permutation(x.ravel()).reshape(x.shape))
        # since xrp_test is a permutation of x, we can use the same neighborhood means
        # x_test_sum = np.float64(np.sum(xrp_test[1:-1,1:-1]))
        for xi in numba.prange(x.shape[0]):
            for yi in numba.prange(x.shape[1]):
                # new neighborhood mean
                if connectivity == CONNECTIVITY_QUEEN:
                    neighbor_sums = (
                        xrp_test[xi + 1 - 1, yi + 1]
                        + xrp_test[xi + 1, yi + 1 - 1]
                        + xrp_test[xi + 1 + 1, yi + 1]
                        + xrp_test[xi + 1, yi + 1 + 1]
                        + xrp_test[xi + 1 - 1, yi + 1 - 1]
                        + xrp_test[xi + 1 + 1, yi + 1 - 1]
                        + xrp_test[xi + 1 - 1, yi + 1 + 1]
                        + xrp_test[xi + 1 + 1, yi + 1 + 1]
                    )
                else:
                    neighbor_sums = (
                        xrp_test[xi + 1 - 1, yi + 1]
                        + xrp_test[xi + 1, yi + 1 - 1]
                        + xrp_test[xi + 1 + 1, yi + 1]
                        + xrp_test[xi + 1, yi + 1 + 1]
                    )
                Gir = (x[xi, yi]*Gstar + neighbor_sums) / (x_sum-x[xi,yi]*Gstar)

                # compare with observed Gi
                perm_G_counts[np.int64(np.abs(Gir) > np.abs(Gi[xi, yi])), xi, yi] += 1
                perm_G_counts[np.int64(Gir < Gi[xi, yi]) + 2, xi, yi] += 1
                perm_G_counts[np.int64(Gir > Gi[xi, yi]) + 4, xi, yi] += 1

    # only return the requested p-values
    GPi = (1 + perm_G_counts[np.where(which_test)[0] * 2 + 1, :, :]) / (n_iter + 1)
    return Gi, GPi


@numba.njit(parallel=True, cache=True)
def _G_variable_permutation(
    x,
    x_sum,
    kernel,
    connectivity=CONNECTIVITY_QUEEN,
    n_iter=99,
    seed=42,
    GPtype=GPVAL_TYPE_BOTH,
):
    np.random.seed(seed)

    # number of neighbors to sample
    n_samples = 8 if connectivity == CONNECTIVITY_QUEEN else 4

    # neighborhood sums
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xns = (
            xp[:-2, 1:-1]
            + xp[1:-1, :-2]
            + xp[2:, 1:-1]
            + xp[1:-1, 2:]
            + xp[:-2, :-2]
            + xp[:-2, 2:]
            + xp[2:, :-2]
            + xp[2:, 2:]
        )
    else:
        xns = xp[:-2, 1:-1] + xp[1:-1, :-2] + xp[2:, 1:-1] + xp[1:-1, 2:]
    # observed Gi
    Gi = xns / x_sum

    kw = kernel.shape[0] // 2
    kh = kernel.shape[1] // 2

    kernel[kw, kh] = 0  # center pixel is not included in the kernel

    # decompose GPtype into its components
    which_test = split_GPtype(GPtype)

    # initialize counts for permutation
    perm_G_counts = np.zeros((3, Gi.shape[0], Gi.shape[1]), dtype=np.uint16)

    # subset of x to radius of kernel + borders
    xt0lskb = np.clip(np.arange(x.shape[0]) - kw, 0, x.shape[0])
    xt1lskb = np.clip(np.arange(x.shape[0]) + kw + 1, 0, x.shape[0])
    yt0lskb = np.clip(np.arange(x.shape[1]) - kh, 0, x.shape[1])
    yt1lskb = np.clip(np.arange(x.shape[1]) + kh + 1, 0, x.shape[1])

    # subset of kernel because of borders
    xt0lsb = np.clip(kw - np.arange(x.shape[0]), 0, kernel.shape[0])
    xt1lsb = np.clip(kw + np.arange(x.shape[0], 0, -1), 0, kernel.shape[0])
    yt0lsb = np.clip(kh - np.arange(x.shape[1]), 0, kernel.shape[1])
    yt1lsb = np.clip(kh + np.arange(x.shape[1], 0, -1), 0, kernel.shape[1])

    # iterate over all pixels
    for xi in numba.prange(x.shape[0]):
        for yi in numba.prange(x.shape[1]):
            xtmp = x[xt0lskb[xi] : xt1lskb[xi], yt0lskb[yi] : yt1lskb[yi]]
            kerneltmp = kernel[xt0lsb[xi] : xt1lsb[xi], yt0lsb[yi] : yt1lsb[yi]]
            # extract values from xtmp that are in the kernel
            xtmptmp = xtmp.ravel()[np.flatnonzero(kerneltmp)]

            # create random choice indices for sampling neighbors
            rindsarr = np.random.randint(0, len(xtmptmp), (n_iter, n_samples))
            for ni in numba.prange(n_iter):
                uq = np.unique(rindsarr[ni])
                if len(uq) < n_samples:
                    # resample indices
                    max_tries = 10
                    while len(uq) < n_samples and max_tries > 0:
                        rindsarr[ni] = np.random.randint(0, len(xtmptmp), n_samples)
                        uq = np.unique(rindsarr[ni])
                        max_tries -= 1

            # permute over neighbors
            neighbor_sums = np.sum(
                xtmptmp[rindsarr.ravel()].reshape(n_iter, n_samples), axis=1
            )
            permGi = (x[xi, yi] + neighbor_sums) / x_sum[xi, yi]
            # compare with observed Gi
            perm_G_counts[0, xi, yi] = np.sum(np.abs(permGi) > np.abs(Gi[xi, yi]))
            perm_G_counts[1, xi, yi] = np.sum(permGi < Gi[xi, yi])
            perm_G_counts[2, xi, yi] = np.sum(permGi > Gi[xi, yi])

    # only return the requested p-values
    GPi = (1 + perm_G_counts[np.arange(3)[np.where(which_test)[0]], :, :]) / (
        n_iter + 1
    )
    return Gi, GPi


def G_variable_permutation(
    x,
    radius=50,
    n_iter=99,
    connectivity=CONNECTIVITY_QUEEN,
    seed=42,
    GPtype=GPVAL_TYPE_BOTH,
):
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    rr, cc = skimage.draw.disk((radius, radius), radius + 0.5)
    kernel[rr, cc] = 1
    x_sum = scipy.signal.fftconvolve(x, kernel, mode="same")

    return _G_variable_permutation(
        x,
        x_sum,
        kernel,
        connectivity=connectivity,
        n_iter=n_iter,
        seed=seed,
        GPtype=GPtype,
    )


def G_variable_permutation_multiple(
    x,
    radius=[10, 25, 50, 100],
    n_iter=99,
    connectivity=CONNECTIVITY_QUEEN,
    seed=42,
    GPtype=GPVAL_TYPE_BOTH,
):
    which_test = split_GPtype(GPtype)
    Gimult = np.zeros((len(radius), x.shape[0], x.shape[1]), dtype=np.float64)
    GPimult = np.zeros(
        (len(radius), np.sum(which_test), x.shape[0], x.shape[1]), dtype=np.float64
    )
    for i, r in enumerate(radius):
        if r == -1:
            Gimult[i], GPimult[i] = G_permutation(
                x, n_iter=n_iter, connectivity=connectivity, seed=seed
            )
        else:
            Gimult[i], GPimult[i] = G_variable_permutation(
                x, radius=r, n_iter=n_iter, connectivity=connectivity, seed=seed
            )
    return Gimult, GPimult


# G_variable_multiple(x, n_iter=9)

# GPic = np.stack((GPi, GPi100, GPi50, GPi25, GPi10), axis=0)
# GPim = np.min(GPic, axis=0)  # take the minimum p-value across all resolutions


def G(
    x,
    n_iter=0,
    radius=None,
    # kernel=Kernel(1.5, include_center=True, weight=KERNEL_WEIGHT_EXP, normalize=True),
    connectivity=CONNECTIVITY_QUEEN,
    seed=42,
    aggregation="min",
    GPtype=GPVAL_TYPE_BOTH,
    fast=True,
    Gstar=True
):
    """
    Calculate the G statistic and its p-value (default: hot spots) for a given input array x.
    Parameters
    ----------
    x : np.ndarray
        Input array for which the G statistic and p-value are calculated.
    n_iter : int, optional
        Number of iterations for permutation testing. If 0, the classical G statistic is calculated.
    radius : int or list of int, optional
        Radius for variable permutation. If None, the classical G statistic is calculated.
    connectivity : int, optional
        Connectivity type for neighborhood calculation. Default is CONNECTIVITY_QUEEN.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    aggregation : str, optional
        Aggregation method for multiple radius values. Can be 'min' or 'mean'. Default is 'min'.
    GPtype : str, optional
        Type of G statistic to calculate. Can be 'hot', 'cold', or 'both'. Default is 'hot'.
    Returns
    -------
    G : np.ndarray
        G statistic for the input array x.
    GP : np.ndarray
        P-value for the G statistic.
    """
    assert aggregation in ["min", "mean"], "Aggregation must be either 'min' or 'mean'."
    which_test = split_GPtype(GPtype)
    all_tests = np.array([GPVAL_TYPE_BOTH, GPVAL_TYPE_COLD, GPVAL_TYPE_HOT])

    # check for possible overflow
    maxval = np.max(x).astype(np.float64)
    required_capacity = max((4 * connectivity + 1) * maxval, maxval**2)
    dtype_limits = [(np.uint16, np.iinfo(np.uint16).max), (np.uint32, np.iinfo(np.uint32).max),(np.uint64, np.iinfo(np.uint64).max)]
    for dtype, limit in dtype_limits:
        if required_capacity <= limit:
            x = x.astype(dtype) if x.dtype != dtype else x
            break

    if n_iter > 0:
        if radius is not None:
            if isinstance(radius, (list, tuple)):
                Gm, GPm = G_variable_permutation_multiple(
                    x,
                    radius=radius,
                    n_iter=n_iter,
                    connectivity=connectivity,
                    seed=seed,
                    GPtype=GPtype,
                )
                if aggregation == "mean":
                    GP = np.mean(GPm, axis=0)
                    G = np.mean(Gm, axis=0)
                else:
                    inds = np.argmin(GPm, axis=0)
                    GP = GPm[
                        inds, :, np.arange(x.shape[0])[:, None], np.arange(x.shape[1])
                    ]
                    G = Gm[inds, np.arange(x.shape[0])[:, None], np.arange(x.shape[1])]
            else:
                G, GP = G_variable_permutation(
                    x,
                    radius=radius,
                    n_iter=n_iter,
                    connectivity=connectivity,
                    seed=seed,
                    GPtype=GPtype,
                )
        else:
            if fast:
                G, GP = G_permutation_fast(
                    x, connectivity=connectivity, n_iter=n_iter, seed=seed, GPtype=GPtype,
                    Gstar=Gstar
                )
            else:
                G, GP = G_permutation(
                    x, connectivity=connectivity, n_iter=n_iter, seed=seed, GPtype=GPtype,
                    Gstar=Gstar
                )
    else:
        G = G_classical(x, connectivity=connectivity, normalize=True, Gstar=Gstar)
        GPtmp = (1.0 - scipy.stats.norm.cdf(np.abs(G))) * 2  # scale to [0,1]
        GP = np.zeros((np.sum(which_test), x.shape[0], x.shape[1]), dtype=np.float64)
        if GPVAL_TYPE_BOTH in all_tests[np.where(which_test == 1)]:
            GP[0] = GPtmp
        if (
            GPVAL_TYPE_COLD in all_tests[np.where(which_test == 1)]
            or GPVAL_TYPE_HOT in all_tests[np.where(which_test == 1)]
        ):
            # scale to [0,0.5] and invert
            GP_cold = 1 - GPtmp.copy() / 2
            # values where G is negative are scaled to [0.5,1] : cold spots
            GP_cold[G < 0] = 0.5 + (0.5 - GP_cold[G < 0])
            if GPVAL_TYPE_COLD in all_tests[np.where(which_test == 1)]:
                GP[GPVAL_TYPE_COLD == all_tests[np.where(which_test == 1)]] = GP_cold
            # invert to get hot spots
            if GPVAL_TYPE_HOT in all_tests[np.where(which_test == 1)]:
                GP[GPVAL_TYPE_HOT == all_tests[np.where(which_test == 1)]] = 1 - GP_cold
    return G, GP


# x = np.random.rand(1000, 1000)
# x[10:20,10:20] += 100
# x[80:120,60:130] += 200
# x[10:20,150:180] += 100


# _, GPi = G_permutation(x, connectivity=CONNECTIVITY_QUEEN, n_iter = 999, seed=42)
# _, GPi100 = G_variable_permutation(x, 100, connectivity=CONNECTIVITY_QUEEN, n_iter = 999, seed=42)
# _, GPi50 = G_variable_permutation(x, 50, connectivity=CONNECTIVITY_QUEEN, n_iter = 999, seed=42)
# _, GPi25 = G_variable_permutation(x, 25, connectivity=CONNECTIVITY_QUEEN, n_iter = 999, seed=42)
# _, GPi10 = G_variable_permutation(x, 10, connectivity=CONNECTIVITY_QUEEN, n_iter = 999, seed=42)
# _, GPcl = G(x)

# GPic = np.stack((GPi, GPi100, GPi50, GPi25, GPi10), axis=0)
# GPim = np.min(GPic, axis=0)  # take the minimum p-value across all resolutions
# import matplotlib.pyplot as plt
# plt.close()
# fig, ax = plt.subplots(2,3,figsize=(15, 10))
# ax[0,0].imshow(GPi<0.05, cmap='hot', vmin=0, vmax=1)
# ax[0,0].set_title('complete permuted')
# ax[1,0].imshow(GPcl<0.05, cmap='hot', vmin=0, vmax=1)
# ax[1,0].set_title('complete analytic')
# ax[0,1].imshow(GPi25<0.05, cmap='hot', vmin=0, vmax=1)
# ax[0,1].set_title('tile size 50x50')
# ax[1,1].imshow(GPi10<0.05, cmap='hot', vmin=0, vmax=1)
# ax[1,1].set_title('tile size 20x20')
# ax[1,2].imshow(GPim<0.05, cmap='hot', vmin=0, vmax=1)
# ax[1,2].set_title('minimum')
# ax[0,2].imshow(x, cmap='hot', vmin=0, vmax=200)
# ax[0,2].set_title('x')
# plt.savefig('GPim.png')

# x = np.random.rand(100, 100)
# x[10:20,10:20] += 1000
# x[40:,40:] += 100000
# Gi, GPi = G_variable(x, n_iter=999, connectivity=CONNECTIVITY_QUEEN, seed=42, min_range=3, max_range=None, n_ranges=10)


@numba.njit(parallel=True, cache=True)
def H_classical(x, connectivity=CONNECTIVITY_QUEEN, normalize=True, return_var=False):
    w1 = 4 * connectivity + 1

    # neighborhood means
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xnm = (
            x
            + xp[:-2, 1:-1]
            + xp[1:-1, :-2]
            + xp[2:, 1:-1]
            + xp[1:-1, 2:]
            + xp[:-2, :-2]
            + xp[:-2, 2:]
            + xp[2:, :-2]
            + xp[2:, 2:]
        ) / w1
    else:
        xnm = (x + xp[:-2, 1:-1] + xp[1:-1, :-2] + xp[2:, 1:-1] + xp[1:-1, 2:]) / w1

    # observed Hi
    xresid = man_pad((x - xnm) ** 2)
    denom = np.mean(xresid[1:-1, 1:-1]) * w1
    if connectivity == CONNECTIVITY_QUEEN:
        Hi = (
            xresid[1:-1, 1:-1]
            + xresid[:-2, 1:-1]
            + xresid[1:-1, :-2]
            + xresid[2:, 1:-1]
            + xresid[1:-1, 2:]
            + xresid[:-2, :-2]
            + xresid[:-2, 2:]
            + xresid[2:, :-2]
            + xresid[2:, 2:]
        ) / denom
    else:
        Hi = (
            xresid[1:-1, 1:-1]
            + xresid[:-2, 1:-1]
            + xresid[1:-1, :-2]
            + xresid[2:, 1:-1]
            + xresid[1:-1, 2:]
        ) / denom

    if normalize:
        h1 = denom / w1
        h2 = np.mean(xresid[1:-1, 1:-1] ** 2)
        n = np.float64((x.shape[0] * x.shape[1]))
        w2 = w1
        VarHi = 1 / (n - 1) * (1 / denom) ** 2 * (h2 - h1**2) * ((n * w2) - (w1**2))
        HZi = (2 * Hi) / VarHi
        if return_var:
            return HZi, VarHi
        else:
            return HZi, None
    else:
        return Hi, None

@numba.njit(parallel=True, cache=True)
def H_classical_new(
        x, 
        kernel=Kernel(1.5, include_center=True, weight=KERNEL_WEIGHT_EXP, normalize=True),
        normalize=True,
        return_var=False,
        a=2.0
    ):

    # neighborhood sums
    xp = man_pad(x, kernel.diameter // 2)
    xns = neighborhood_sum(x, kernel, xp=xp)
    w1 = kernel.w
    xresid = (x - xns/w1) ** a
    xresidp = man_pad(xresid, kernel.diameter // 2)
    denom = np.mean(xresid) * w1
    Hi = neighborhood_sum(xresid, kernel, xp=xresidp)/denom

    if normalize:
        h1 = denom / w1
        h2 = np.mean(xresid ** 2)
        n = np.float64((x.shape[0] * x.shape[1]))
        w2 = w1
        VarHi = 1 / (n - 1) * (1 / denom) ** 2 * (h2 - h1**2) * ((n * w2) - (w1**2))
        HZi = (2 * Hi) / VarHi
        if return_var:
            return HZi, VarHi
        else:
            return HZi, None
    else:
        return Hi, None


# x = np.random.rand(20, 20)
# # x = np.zeros((10, 10), dtype=np.float64)
# x[1:6, 1:6] += 10
# Kernel(1.5, include_center=True, weight=KERNEL_WEIGHT_NONE, normalize=False).kernel
# out2,var2 = H_classical_new(x, kernel=Kernel(2, include_center=True, weight=KERNEL_WEIGHT_NONE, normalize=False), return_var=True)
# dof = 2 / var2
# HP = 1 - scipy.stats.chi2.cdf(out2, dof)
# (x>5).astype(int)
# np.round(out2,1)
# np.round(HP,1)

# H(x, connectivity=CONNECTIVITY_QUEEN, normalize=False, return_var=False)


@numba.njit(parallel=True, cache=True)
def H_permutation(x, connectivity=CONNECTIVITY_QUEEN, n_iter=99, seed=42):
    np.random.seed(seed)
    w1 = 4 * connectivity + 1

    # neighborhood means
    xp = man_pad(x)
    if connectivity == CONNECTIVITY_QUEEN:
        xnm = (
            x
            + xp[:-2, 1:-1]
            + xp[1:-1, :-2]
            + xp[2:, 1:-1]
            + xp[1:-1, 2:]
            + xp[:-2, :-2]
            + xp[:-2, 2:]
            + xp[2:, :-2]
            + xp[2:, 2:]
        ) / w1
    else:
        xnm = (x + xp[:-2, 1:-1] + xp[1:-1, :-2] + xp[2:, 1:-1] + xp[1:-1, 2:]) / w1

    # observed Hi
    xresid = man_pad((x - xnm) ** 2)
    denom = np.mean(xresid[1:-1, 1:-1]) * w1
    if connectivity == CONNECTIVITY_QUEEN:
        Hi = (
            xresid[1:-1, 1:-1]
            + xresid[:-2, 1:-1]
            + xresid[1:-1, :-2]
            + xresid[2:, 1:-1]
            + xresid[1:-1, 2:]
            + xresid[:-2, :-2]
            + xresid[:-2, 2:]
            + xresid[2:, :-2]
            + xresid[2:, 2:]
        ) / denom
    else:
        Hi = (
            xresid[1:-1, 1:-1]
            + xresid[:-2, 1:-1]
            + xresid[1:-1, :-2]
            + xresid[2:, 1:-1]
            + xresid[1:-1, 2:]
        ) / denom

    perm_H_counts = np.zeros((2, Hi.shape[0], Hi.shape[1]), dtype=np.uint16)
    for rep in numba.prange(n_iter):
        # random permutation of x
        xrp_test = man_pad(np.random.permutation(x.flatten()).reshape(x.shape))
        if connectivity == CONNECTIVITY_QUEEN:
            xrnm_init = (
                xrp_test[1:-1, 1:-1]
                + xrp_test[:-2, 1:-1]
                + xrp_test[1:-1, :-2]
                + xrp_test[2:, 1:-1]
                + xrp_test[1:-1, 2:]
                + xrp_test[:-2, :-2]
                + xrp_test[:-2, 2:]
                + xrp_test[2:, :-2]
                + xrp_test[2:, 2:]
            ) / w1
        else:
            xrnm_init = (
                xrp_test[1:-1, 1:-1]
                + xrp_test[:-2, 1:-1]
                + xrp_test[1:-1, :-2]
                + xrp_test[2:, 1:-1]
                + xrp_test[1:-1, 2:]
            ) / w1
        xresidr_init = (xrp_test[1:-1, 1:-1] - xrnm_init) ** 2
        n = x.shape[0] * x.shape[1]
        denomr_sum_init = np.sum(xresidr_init) * w1
        for xi, yi in np.ndindex(x.shape):
            init_rval = xrp_test[xi + 1, yi + 1]
            xrp_test[xi + 1, yi + 1] = x[xi, yi]
            xstart = max(0, xi - 1)
            xstart_offset = xstart - (xi - 1)
            xend = min(xi + 2, x.shape[0])
            ystart = max(0, yi - 1)
            ystart_offset = ystart - (yi - 1)
            yend = min(yi + 2, x.shape[1])

            # partial sum for normalization
            xresidr_init_initsubarray_sum = (
                np.sum(xresidr_init[xstart:xend, ystart:yend]) * w1
            )
            # new neighborhood means
            xrnm_init_replacement_array = np.zeros((xend - xstart, yend - ystart))
            for ii, i in enumerate(range(xstart, xend)):
                for jj, j in enumerate(range(ystart, yend)):
                    # +1 everywhere because of the padding
                    ip = i + 1
                    jp = j + 1
                    if connectivity == CONNECTIVITY_QUEEN:
                        xrnm_init_replacement_array[ii, jj] = (
                            xrp_test[ip, jp]
                            + xrp_test[ip - 1, jp]
                            + xrp_test[ip, jp - 1]
                            + xrp_test[ip + 1, jp]
                            + xrp_test[ip, jp + 1]
                            + xrp_test[ip - 1, jp - 1]
                            + xrp_test[ip + 1, jp - 1]
                            + xrp_test[ip - 1, jp + 1]
                            + xrp_test[ip + 1, jp + 1]
                        ) / w1
                    else:
                        xrnm_init_replacement_array[ii, jj] = (
                            xrp_test[ip, jp]
                            + xrp_test[ip - 1, jp]
                            + xrp_test[ip, jp - 1]
                            + xrp_test[ip + 1, jp]
                            + xrp_test[ip, jp + 1]
                        ) / w1
            # new residuals
            xresidr_init_replacement_array = np.zeros((xend - xstart, yend - ystart))
            for ii, i in enumerate(range(xstart, xend)):
                for jj, j in enumerate(range(ystart, yend)):
                    # +1 everywhere because of the padding
                    ip = i + 1
                    jp = j + 1
                    xresidr_init_replacement_array[ii, jj] = (
                        xrp_test[ip, jp] - xrnm_init_replacement_array[ii, jj]
                    ) ** 2

            # update the denominator, remove initial partial sum and add the new partial sum, and calculate the mean
            denomr = (
                denomr_sum_init
                - xresidr_init_initsubarray_sum
                + np.sum(xresidr_init_replacement_array) * w1
            ) / n

            # pad for boundaries
            xresidr_init_replacement_array = man_pad(xresidr_init_replacement_array)

            # calculate Hi for the replacement, xstart_offset and ystart_offset are for the boundaries
            if connectivity == CONNECTIVITY_QUEEN:
                Hir = (
                    xresidr_init_replacement_array[2 - xstart_offset, 2 - ystart_offset]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset - 1, 2 - ystart_offset
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset, 2 - ystart_offset - 1
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset + 1, 2 - ystart_offset
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset, 2 - ystart_offset + 1
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset - 1, 2 - ystart_offset - 1
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset + 1, 2 - ystart_offset - 1
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset - 1, 2 - ystart_offset + 1
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset + 1, 2 - ystart_offset + 1
                    ]
                ) / denomr
            else:
                Hir = (
                    xresidr_init_replacement_array[2 - xstart_offset, 2 - ystart_offset]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset - 1, 2 - ystart_offset
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset, 2 - ystart_offset - 1
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset + 1, 2 - ystart_offset
                    ]
                    + xresidr_init_replacement_array[
                        2 - xstart_offset, 2 - ystart_offset + 1
                    ]
                ) / denomr

            # compare with observed Hi
            perm_H_counts[np.int64(Hir > Hi[xi, yi]), xi, yi] += 1

            # restore the original permuted value
            xrp_test[xi + 1, yi + 1] = init_rval

    HPi = (1 + perm_H_counts[1]) / (n_iter + 1)
    return Hi, HPi


def H(x, n_iter=99, connectivity=CONNECTIVITY_QUEEN, seed=42):
    H, VarH = H_classical(x, connectivity=connectivity, normalize=True, return_var=True)
    if n_iter > 0:
        HP = H_permutation(x, connectivity=connectivity, n_iter=n_iter, seed=seed)[1]
    else:
        dof = 2 / VarH
        HP = 1 - scipy.stats.chi2.cdf(H, dof)
    return H, HP


# x = np.random.normal(size=(100, 100))
# # x = np.random.random(size=(100, 100))
# x = np.random.binomial(p=0.25,n=10,size=(100, 100)).astype(np.float64)
# x[:2, :] += 10
# houtiter = H(x, n_iter=999)[1]
# houtcl = H(x, n_iter=0)[1]
# # import scipy.stats
# # houtiter = scipy.stats.false_discovery_control(houtiter)
# # houtcl = scipy.stats.false_discovery_control(houtcl)
# np.round(x[:5,3:8],2)
# np.round(houtcl[:5,3:8],2)
# np.round(houtiter[:5,3:8],2)
# import matplotlib.pyplot as plt
# plt.close()
# fig, ax = plt.subplots(3, 1, figsize=(10, 15))
# ax[0].hist(houtiter[1:3,:].flatten(), bins=20, label='H permutation')
# ax[0].set_title('H permutation distribution')
# ax[0].set_xlabel('H permutation values')
# ax[0].set_ylabel('Frequency')
# # ax[0].set_xlim(0, 1)
# ax[1].hist(houtcl[1:3,:].flatten(), bins=20, label='H permutation')
# ax[1].set_title('H classical distribution')
# ax[1].set_xlabel('H classical values')
# ax[1].set_ylabel('Frequency')
# # ax[1].set_xlim(0, 1)
# ax[2].scatter(houtcl[1:3,:].flatten(),houtiter[1:3,:].flatten(), label='H permutation', s=1)
# ax[2].set_title('H permutation vs H classical')
# ax[2].set_xlabel('H classical values')
# ax[2].set_ylabel('H permutation values')
# # ax[2].set_xlim(0, 1)
# # ax[2].set_ylim(0, 1)
# plt.savefig('H_permutation.png')


# import timeit
# import time
# iterations=10
# times1 = []
# times2 = []
# times3 = []
# # ns = [50, 100,500, 1000,5000, 10000,20000]
# # ns = [1000,5000,10000,20000]
# # ns = [50,100,200,500,750,1000,1500,2000,2500,5000,7500]
# ns = [10]
# for i in range(15):
#     ns.append(np.round(np.sqrt(ns[-1]**2*2)).astype(int))
# ns = np.array(ns)
# # ns = [10,20,30,40,50,75,100,200,500,750,1000,1500,2000,2500,5000,7500]
# for n in ns:
#     tmp_iterations = np.ceil(np.max([1/(n/np.max(ns)),1])).astype(int)*10
#     print(f"n: {n}, iterations: {tmp_iterations}")
#     x = np.random.rand(n,n)
#     print(f"\tnumba serial ")
#     time.sleep(0.5)
#     ts = np.array([timeit.timeit("G_and_H_numba_serial(x)", number=1, globals=globals()) for i in range(tmp_iterations)])
#     print(f"\t\tlen: {len(ts)}, median: {np.median(ts):.6f}")
#     times1.append(np.median(ts))
#     print(f"\tnumba parallel ")
#     time.sleep(0.5)
#     ts = np.array([timeit.timeit("G_and_H_numba(x)", number=1, globals=globals()) for i in range(tmp_iterations)])
#     print(f"\t\tlen: {len(ts)}, median: {np.median(ts):.6f}")
#     times2.append(np.median(ts))
#     print(f"\tnumba variable")
#     time.sleep(0.5)
#     ts = np.array([timeit.timeit("G_and_H(x)", number=1, globals=globals()) for i in range(tmp_iterations)])
#     print(f"\t\tlen: {len(ts)}, median: {np.median(ts):.6f}")
#     times3.append(np.median(ts))

# # tmp_iterations=10000
# # timeit.timeit("G_and_H(x)", number=tmp_iterations, globals=globals())/tmp_iterations
# # timeit.timeit("G_and_H(x)", number=tmp_iterations, globals=globals())/tmp_iterations

# import matplotlib.pyplot as plt
# fig,ax = plt.subplots(1,2,figsize=(10, 5))
# ax[0].plot(ns**2, times1, label='numba serial')
# ax[0].plot(ns**2, times2, label='numba parallel')
# ax[0].plot(ns**2, times3, label='numba variable', c="black")
# ax[0].set_xlabel('n')
# ax[0].set_ylabel('Time (seconds)')
# ax[0].set_title('Time taken to compute G and H')
# ax[0].legend()
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[1].plot(ns**2, np.array(times2)/np.array(times1), label='numba parallel vs. numba serial')
# ax[1].plot(ns**2, np.array(times3)/np.array(times1), label='numba variable vs. numba serial')
# ax[1].plot(ns**2, np.array(times3)/np.array(times2), label='numba variable vs. numba parallel',c="black")
# ax[1].set_xlabel('n')
# ax[1].set_ylabel('Speedup')
# ax[1].set_title('Speedup of numba vs ne')
# ax[1].set_xscale('log')
# ax[1].legend()
# plt.savefig('G_and_H.png')
