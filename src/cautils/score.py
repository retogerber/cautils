import numba
import numpy as np
import scipy
import polars as pl

# from nyxus import Nyxus
from cautils.gettisord import G

MARKER_LOCATIONS = {"Nuclear": 0, "Membrane": 1, "Whole_cell": 2}

# x = np.arange(1000)/250-2
# y = np.array([-np.abs(np.random.normal(i, 0.1, 1)) for i in x]).flatten()
# y = y - np.min(y)
# get_score(x,y)


@numba.njit(cache=True)
def get_smooth_max(x, y, nmaxdist=10, nmindist=25, minp=0.01):
    if len(x) == 0 or len(y) == 0:
        return np.array([np.nan, np.nan])
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.array([np.nan, np.nan])
    yn = y[~np.isnan(y)]
    xn = x[~np.isnan(y)]
    y_order = np.argsort(yn)
    y_sorted = yn[y_order]
    x_sorted = xn[y_order]
    xq99 = np.quantile(x_sorted, 0.99)
    max_distance = xq99 / nmaxdist
    min_distance = xq99 / nmindist
    tmp = x_sorted[int(np.round(len(y_sorted) * 0.95)) :]
    initx = np.partition(tmp, tmp.size // 2)[tmp.size // 2]
    tdist = np.abs(initx - x_sorted)
    used_max_distance = np.nanmax(
        [np.nanmin([np.nanquantile(tdist, minp), max_distance]), min_distance]
    )
    ws = 1 - np.clip(tdist / used_max_distance, 0, 1)
    return np.array([initx, np.sum(y_sorted * ws) / np.sum(ws)])


def do_smooth(x, y, nmaxdist=10, nmindist=25, minp=0.01):
    yn = y[~np.isnan(y)]
    xn = x[~np.isnan(y)]
    max_distance = np.nanquantile(xn, 0.99) / nmaxdist
    min_distance = np.nanquantile(xn, 0.99) / nmindist
    y_smooth = np.zeros_like(y)
    for i in range(len(x)):
        if np.isnan(y[i]):
            y_smooth[i] = np.nan
        else:
            tdist = scipy.spatial.distance.cdist(
                x[[i], np.newaxis], x[:, np.newaxis], metric="euclidean"
            )[0, ~np.isnan(y)]
            used_max_distance = np.nanmax(
                [np.nanmin([np.nanquantile(tdist, minp), max_distance]), min_distance]
            )
            ws = 1 - np.clip(tdist / used_max_distance, 0, 1)
            y_smooth[i] = np.sum(yn * ws) / np.sum(ws)
    # is_outlier = np.abs((y_smooth-y))>0.05
    # y_smooth = np.zeros_like(y)
    # for i in range(len(x)):
    #     tdist = scipy.spatial.distance.cdist(x[[i],np.newaxis], x[:,np.newaxis], metric='euclidean')
    #     ws = 1-np.clip(tdist/max_distance,0,1)
    #     ws[0,is_outlier] = 0
    #     y_smooth[i] = np.sum(y*ws)/np.sum(ws)
    return y_smooth


@numba.njit(cache=True)
def get_score(x, y, xy_max=None):
    # y_smooth =  do_smooth(x, y)
    # max_idx = np.argmax(y_smooth)
    # x_max = x[max_idx]
    # y_max = y_smooth[max_idx]
    if xy_max is None:
        xy_max = get_smooth_max(x, y, nmaxdist=10, nmindist=25, minp=0.01)
    if np.isnan(xy_max[0]) or np.isnan(xy_max[1]):
        return np.ones_like(y) * np.nan

    y_new = y.copy()
    y_new[np.logical_or(y > xy_max[1], x < xy_max[0])] = xy_max[1]
    y_new = 1 - y_new / np.nanmax(y_new)
    return y_new


@numba.njit(cache=True)
def bbox_label(mask):
    maxval = int(np.max(mask) + 1)
    labels = np.zeros((maxval, 2), dtype=np.uint32)
    labels[:, 0] = np.arange(maxval)
    bboxs = np.zeros((maxval, 4), dtype=np.int64)
    bboxs[:, 0] = np.max(np.array(mask.shape)) + 1
    bboxs[:, 1] = np.max(np.array(mask.shape)) + 1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                labels[mask[i, j], 1] = 1
                if i < bboxs[mask[i, j], 0]:
                    bboxs[mask[i, j], 0] = i
                if i + 1 > bboxs[mask[i, j], 2]:
                    bboxs[mask[i, j], 2] = i + 1
                if j < bboxs[mask[i, j], 1]:
                    bboxs[mask[i, j], 1] = j
                if j + 1 > bboxs[mask[i, j], 3]:
                    bboxs[mask[i, j], 3] = j + 1

    keep = labels[:, 1] == 1
    labels = labels[keep, 0]
    bboxs = bboxs[keep, :]
    return labels, bboxs[:, :]


@numba.njit(parallel=False, cache=True)
def man_pad_zero(x):
    xm = np.zeros((x.shape[0] + 2, x.shape[1] + 2), dtype=x.dtype)
    xm[1:-1, 1:-1] = x
    return xm


@numba.njit(parallel=False, cache=True)
def get_perimeter_and_not_perimeter_values(is_peri, submask, subimage):
    not_is_peri = np.logical_and(~is_peri, submask != 0)
    edgevals = np.empty((subimage.shape[0], np.sum(is_peri)), dtype=np.float64)
    corevals = np.empty((subimage.shape[0], np.sum(not_is_peri)), dtype=np.float64)
    pind = 0
    npind = 0
    for i in range(subimage.shape[1]):
        for j in range(subimage.shape[2]):
            if is_peri[i, j]:
                edgevals[:, pind] = subimage[:, i, j]
                pind += 1
            if not_is_peri[i, j]:
                corevals[:, npind] = subimage[:, i, j]
                npind += 1
    return edgevals, corevals


@numba.njit(parallel=False, cache=True)
def permute_core_edge_mean_diff(is_peri, submask, subimage, n_iter=99, seed=42):
    np.random.seed(seed)

    edgevals, corevals = get_perimeter_and_not_perimeter_values(
        is_peri, submask, subimage
    )
    n_edgevals = edgevals.shape[1]
    n_corevals = corevals.shape[1]
    if n_edgevals == 0 or n_corevals == 0:
        return np.ones(subimage.shape[0]) * np.nan, np.ones(subimage.shape[0]) * np.nan

    vals_concat = np.empty(
        (subimage.shape[0], n_edgevals + n_corevals), dtype=np.float64
    )
    vals_concat[:, :n_corevals] = corevals
    vals_concat[:, n_corevals:] = edgevals

    observed_diff = np.empty(subimage.shape[0], dtype=np.float64)
    for i in range(subimage.shape[0]):
        observed_diff[i] = np.mean(corevals[i]) - np.mean(edgevals[i])
    abs_observed_diff = np.abs(observed_diff)

    abs_is_larger = np.zeros(subimage.shape[0], dtype=np.uint16)
    for iter in numba.prange(n_iter):
        for i in range(subimage.shape[0]):
            perm = np.random.permutation(vals_concat[i, :])
            abs_is_larger[i] += (
                np.abs(np.mean(perm[:n_corevals]) - np.mean(perm[n_corevals:]))
                > abs_observed_diff[i]
            )
    pvals = (1 + abs_is_larger) / (1 + n_iter)
    return observed_diff, pvals


@numba.njit(cache=True)
def get_perimeter(submask):
    tmpsubmask = man_pad_zero(submask)
    is_peri = (
        tmpsubmask[1:-1, 1:-1]
        - (
            tmpsubmask[:-2, 1:-1]
            + tmpsubmask[2:, 1:-1]
            + tmpsubmask[1:-1, :-2]
            + tmpsubmask[1:-1, 2:]
        )
        // (4 * np.max(tmpsubmask))
        == 1
    )
    return is_peri


@numba.njit(cache=True)
def _HodgesLehmannEstimator(x, y):
    diffarr = np.empty((len(x), len(y)), dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(y)):
            diffarr[i, j] = x[i] - y[j]
    return np.median(diffarr)


@numba.njit(cache=True)
def HodgesLehmannEstimator(is_peri, submask, subimage):
    edgevals, corevals = get_perimeter_and_not_perimeter_values(
        is_peri, submask, subimage
    )
    hle = np.empty(subimage.shape[0], dtype=np.float64)
    for i in range(subimage.shape[0]):
        if edgevals.size == 0 or corevals.size == 0:
            hle[i] = np.nan
        else:
            hle[i] = _HodgesLehmannEstimator(corevals[i], edgevals[i])
    return hle


@numba.njit(cache=True)
def get_intensities_perimeter(subimage, is_peri, qs=np.array([0, 0.25, 0.5, 0.75, 1])):
    tmpimg = np.empty(
        (subimage.shape[0], subimage.shape[1] + 2, subimage.shape[2] + 2),
        dtype=subimage.dtype,
    )
    for j in range(subimage.shape[0]):
        tmpimg[j, :, :] = man_pad_zero(subimage[j, :])
    perin = np.sum(is_peri)
    perivals = np.empty((subimage.shape[0], perin), dtype=np.float64)
    outsums = np.zeros((tmpimg.shape[0],), dtype=np.float64)
    ind = 0
    for i in range(1, tmpimg.shape[1] - 1):
        for j in range(1, tmpimg.shape[2] - 1):
            if is_peri[i - 1, j - 1]:
                for ll in range(tmpimg.shape[0]):
                    outsums[ll] += tmpimg[ll, i, j]
                    perivals[ll, ind] = tmpimg[ll, i, j]
                ind += 1
    outqs = np.empty((tmpimg.shape[0], len(qs)), dtype=np.float64)
    iqmean = np.zeros((tmpimg.shape[0],), dtype=np.float64)
    if perin > 0:
        for ll in range(tmpimg.shape[0]):
            outqs[ll, :] = np.quantile(perivals[ll, :], qs)
            q1, q3 = np.quantile(perivals[ll, :], [0.25, 0.75])
            iqmean[ll] = np.mean(
                perivals[ll, :][perivals[ll, :] <= (q3 + (q3 - q1) * 1.5)]
            )
    return outsums, perin, outqs, iqmean


@numba.njit(parallel=True, cache=True)
def _get_features(
    image,
    mask,
    qs=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    calc_hle=True,
    calc_coreedgediff=True,
):
    label, bbox = bbox_label(mask)
    edge_sums_arr = np.zeros((bbox.shape[0], image.shape[0]), dtype=np.float64)
    edge_area_arr = np.zeros(bbox.shape[0], dtype=np.int64)
    sum_arr = np.zeros((bbox.shape[0], image.shape[0]), dtype=np.float64)
    area_arr = np.zeros(bbox.shape[0], dtype=np.int64)
    iqmeans = np.zeros((bbox.shape[0], image.shape[0]), dtype=np.float64)
    outqs = np.zeros((bbox.shape[0], image.shape[0], len(qs)), dtype=np.float64)
    hle = np.zeros((bbox.shape[0], image.shape[0]), dtype=np.float64)
    coreedgediff = np.zeros((bbox.shape[0], image.shape[0]), dtype=np.float64)
    coreedgediff_pval = np.zeros((bbox.shape[0], image.shape[0]), dtype=np.float64)
    for i in numba.prange(label.size):
        submask = (
            mask[bbox[i, 0] : bbox[i, 2], bbox[i, 1] : bbox[i, 3]] == label[i]
        ).astype(np.uint8)
        is_peri = get_perimeter(submask)
        subimage = image[:, bbox[i, 0] : bbox[i, 2], bbox[i, 1] : bbox[i, 3]]
        edge_sums_arr[i, :], edge_area_arr[i], outqs[i, :, :], iqmeans[i, :] = (
            get_intensities_perimeter(subimage, is_peri=is_peri, qs=qs)
        )
        if calc_hle:
            hle[i, :] = HodgesLehmannEstimator(is_peri, submask, subimage)
        if calc_coreedgediff:
            coreedgediff[i, :], coreedgediff_pval[i, :] = permute_core_edge_mean_diff(
                is_peri, submask, subimage, n_iter=99
            )

        area_arr[i] = np.sum(submask)
        for ch in range(image.shape[0]):
            sum_arr[i, ch] = np.sum(subimage[ch, :, :] * submask)
    return (
        label,
        sum_arr,
        area_arr,
        edge_sums_arr,
        edge_area_arr,
        outqs,
        iqmeans,
        hle,
        coreedgediff,
        coreedgediff_pval,
    )


def get_features(
    image,
    mask,
    channelnames,
    qs=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    calc_hle=False,
    calc_coreedgediff=False,
):
    (
        label,
        sum_arr,
        area_arr,
        edge_sums_arr,
        edge_area_arr,
        outqs,
        iqmeans,
        hle,
        coreedgediff,
        coreedgediff_pval,
    ) = _get_features(
        image, mask, qs=qs, calc_hle=calc_hle, calc_coreedgediff=calc_coreedgediff
    )
    df = pl.DataFrame(
        {
            **{
                "ObjectNumber": label,
                "AREA": area_arr,
                "AREA_EDGE": edge_area_arr,
                "AREA_CORE": area_arr - edge_area_arr,
            },
            **{f"{ch}_SUM": sum_arr[:, i] for i, ch in enumerate(channelnames)},
            **{
                f"{ch}_EDGE_SUM": edge_sums_arr[:, i]
                for i, ch in enumerate(channelnames)
            },
            **{
                f"{ch}_EDGE_Q{int(qs[j] * 100)}": outqs[:, i, j]
                for i, ch in enumerate(channelnames)
                for j in range(len(qs))
            },
            **{f"{ch}_EDGE_IQMEAN": iqmeans[:, i] for i, ch in enumerate(channelnames)},
        }
    ).lazy()
    if calc_hle:
        df = df.with_columns(
            **{f"{ch}_HLE_CORE_EDGE": hle[:, i] for i, ch in enumerate(channelnames)}
        )
    if calc_coreedgediff:
        df = df.with_columns(
            {
                **{
                    f"{ch}_CORE_EDGE_diff": coreedgediff[:, i]
                    for i, ch in enumerate(channelnames)
                },
                **{
                    f"{ch}_CORE_EDGE_diff_pval": coreedgediff_pval[:, i]
                    for i, ch in enumerate(channelnames)
                },
            }
        )
    df = (
        df.with_columns(
            **{
                "AREA_CORE": pl.col("AREA") - pl.col("AREA_EDGE"),
            }
        )
        .with_columns(
            **{
                f"{ch}_MEAN": pl.col(f"{ch}_SUM") / pl.col("AREA")
                for i, ch in enumerate(channelnames)
            }
        )
        .with_columns(
            **{
                f"{ch}_EDGE_MEAN": pl.col(f"{ch}_EDGE_SUM") / pl.col("AREA_EDGE")
                for i, ch in enumerate(channelnames)
            }
        )
        .with_columns(
            **{
                f"{ch}_CORE_SUM": pl.col(f"{ch}_SUM") - pl.col(f"{ch}_EDGE_SUM")
                for i, ch in enumerate(channelnames)
            }
        )
        .with_columns(
            **{
                f"{ch}_CORE_MEAN": pl.col(f"{ch}_CORE_SUM") / pl.col("AREA_CORE")
                for i, ch in enumerate(channelnames)
            }
        )
        .with_columns(
            **{  # replace NaN values in CORE_MEAN with 0
                f"{ch}_CORE_MEAN": pl.when(pl.col(f"{ch}_CORE_MEAN").is_nan())
                .then(0)
                .otherwise(pl.col(f"{ch}_CORE_MEAN"))
                for ch in channelnames
            }
        )
        .with_columns(
            **{  # replace NaN values in CORE_MEAN with 0
                f"{ch}_CORE_MEAN": pl.when(pl.col(f"{ch}_CORE_MEAN").is_infinite())
                .then(0)
                .otherwise(pl.col(f"{ch}_CORE_MEAN"))
                for ch in channelnames
            }
        )
        .drop([f"{ch}_SUM" for ch in channelnames])
        .drop([f"{ch}_CORE_SUM" for ch in channelnames])
        .drop([f"{ch}_EDGE_SUM" for ch in channelnames])
    )
    return df.collect()


def calculate_features(
    image,
    mask,
    channelnames=None,
    fdr_control=True,
    n_iter=0,
    radius=None,
    min_pval=0.1,
    offset=0.1,
    weighted=False,
    include_weights=False,
    include_scores=False,
    qs=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    calc_hle=False,
    calc_coreedgediff=False,
):
    assert image.ndim == 3, "Image must be a 3D numpy array (channels, height, width)"
    assert mask.ndim == 2, "Mask must be a 2D numpy array (height, width)"
    assert image.shape[1:] == mask.shape, "Image and mask dimensions must match"
    if channelnames is None:
        channelnames = np.array([f"Channel_{i}" for i in range(image.shape[0])])
    elif len(channelnames) != image.shape[0]:
        raise ValueError(
            "Length of channelnames must match the number of channels in the image"
        )

    image_G = np.zeros_like(image, dtype=float)
    image_GP = np.zeros_like(image, dtype=float)
    image_GPhot = np.zeros_like(image, dtype=float)
    for chind in range(len(channelnames)):
        GZi, GP = G(
            image[chind, :, :].astype(float), n_iter=n_iter, radius=radius, GPtype=5
        )
        image_G[chind, :, :] = GZi
        image_GP[chind, :, :] = GP[0, :, :]
        image_GPhot[chind, :, :] = GP[1, :, :]
    if fdr_control:
        for chind in range(len(channelnames)):
            image_GPhot[chind, :, :] = scipy.stats.false_discovery_control(
                image_GPhot[chind, :, :]
            )
        for chind in range(len(channelnames)):
            image_GP[chind, :, :] = scipy.stats.false_discovery_control(
                image_GP[chind, :, :]
            )

    channelnames_ls = (
        channelnames.tolist()
        + [f"{name}_GP" for name in channelnames.tolist()]
        + [f"{name}_GPhot" for name in channelnames.tolist()]
    )
    df = get_features(
        np.vstack([image, image_GP, image_GPhot]),
        mask,
        channelnames=channelnames_ls,
        qs=qs,
        calc_hle=calc_hle,
        calc_coreedgediff=calc_coreedgediff,
    )

    if weighted or include_weights:
        weights = 1 - np.clip(image_GPhot, 0, min_pval)
        weights = np.clip((weights - (1 - min_pval)) * (1 / min_pval), offset, 1)
        if weighted:
            df_weighted = get_features(
                image * weights,
                mask,
                channelnames=[f"{name}_weighted" for name in channelnames.tolist()],
            ).drop(["AREA", "AREA_EDGE", "AREA_CORE"])
            df = df.join(df_weighted, on="ObjectNumber", how="left")
        if include_weights:
            df_weights = get_features(
                weights,
                mask,
                channelnames=[f"{name}_weights" for name in channelnames.tolist()],
            ).drop(["AREA", "AREA_EDGE", "AREA_CORE"])
            df = df.join(df_weights, on="ObjectNumber", how="left")

    if include_scores:
        xy_max_ls = {
            ch: get_smooth_max(
                df[f"{ch}_MEAN"].to_numpy(), df[f"{ch}_GP_MEAN"].to_numpy()
            )
            for ch in channelnames
        }

        df = (
            df.with_columns(
                **{  # Calculate scores for each channel
                    f"{ch}_GP_{m}_score": get_score(
                        df[f"{ch}_{m}"].to_numpy(),
                        df[f"{ch}_GP_{m}"].to_numpy(),
                        xy_max=xy_max_ls[ch],
                    )
                    for ch in channelnames
                    for m in ["MEAN", "EDGE_MEAN", "CORE_MEAN"]
                }
            )
            .with_columns(
                **{  # Calculate differences between scores Edge and Core
                    f"{ch}_GP_MEAN_diff_score": pl.col(f"{ch}_GP_CORE_MEAN_score")
                    - pl.col(f"{ch}_GP_EDGE_MEAN_score")
                    for ch in channelnames
                }
            )
            .with_columns(  # weights
                pl.min_horizontal("AREA_EDGE", "AREA_CORE").alias("AREA_MIN_EDGE_CORE")
            )
        )
    return df


def calculate_score(
    image,
    mask,
    channelnames=None,
    fdr_control=True,
    n_iter=0,
    radius=None,
    min_pval=0.1,
    offset=0.1,
    weighted=False,
    include_weights=False,
    qs=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
):
    return calculate_features(
        image,
        mask,
        channelnames=channelnames,
        fdr_control=fdr_control,
        n_iter=n_iter,
        radius=radius,
        min_pval=min_pval,
        offset=offset,
        weighted=weighted,
        include_weights=include_weights,
        include_scores=True,
        qs=qs,
        calc_hle=False,
        calc_coreedgediff=False,
    )


@numba.njit(cache=True)
def weighted_proportion(values, wts):
    ps = 0
    for i in range(len(values)):
        if values[i] > 0:
            ps += wts[i]
    if ps > 0:
        return ps / np.sum(wts)
    else:
        return 0.0


@numba.njit(cache=True, parallel=True)
def bootstrap_weighted_proportion(data, weights, n_bootstrap=1000, alpha=0.05):
    """Bootstrap approach for weighted proportion testing"""

    # Original weighted proportion
    orig_prop = weighted_proportion(data, weights)

    # Bootstrap resampling
    bootstrap_props = np.empty(n_bootstrap)
    for i in numba.prange(n_bootstrap):
        indices = np.random.randint(0, data.size, size=data.size)
        bootstrap_props[i] = weighted_proportion(data[indices], weights[indices])

    # Confidence interval
    ci_lower = np.quantile(bootstrap_props, alpha / 2)
    ci_upper = np.quantile(bootstrap_props, 1 - alpha / 2)

    return orig_prop, ci_lower, ci_upper, bootstrap_props


def get_marker_location_df(
    df,
    channelnames,
    min_GP_score=0.5,
    max_GP_score=1,
    alpha=0.05,
    min_p=0.5,
    n_bootstrap=1000,
    return_all=False,
    min_area=None,
    marker_base_name="_GP_MEAN_diff_score",
):
    marker_locations = []
    for ch in channelnames:
        # Filter based on GP scores
        dfsub1 = df.filter(
            (pl.col(f"{ch}_GP_EDGE_MEAN_score") > min_GP_score)
            | (pl.col(f"{ch}_GP_CORE_MEAN_score") > min_GP_score)
        ).filter(
            (pl.col(f"{ch}_GP_EDGE_MEAN_score") < max_GP_score)
            & (pl.col(f"{ch}_GP_CORE_MEAN_score") < max_GP_score)
        )
        if min_area is not None:
            dfsub1 = dfsub1.filter(pl.col("AREA_MIN_EDGE_CORE") > min_area)
        data = dfsub1[f"{ch}{marker_base_name}"].to_numpy()
        weights = (
            dfsub1["AREA_MIN_EDGE_CORE"].to_numpy()
            / np.sum(dfsub1["AREA_MIN_EDGE_CORE"].to_numpy())
        ) * dfsub1.shape[0]

        prop, ci_low, ci_high, boot_props = bootstrap_weighted_proportion(
            data, weights, n_bootstrap=n_bootstrap, alpha=alpha
        )
        if ci_low > min_p:
            marker = "Nuclear"
        elif ci_high < 1 - min_p:
            marker = "Membrane"
        else:
            marker = "Whole_cell"
        if return_all:
            marker_locations.append(
                pl.DataFrame(
                    {
                        "channel": ch,
                        "location": marker,
                        "n": dfsub1.shape[0],
                        "proportion": prop,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
            )
        else:
            marker_locations.append(marker)
    if return_all:
        return pl.concat(marker_locations)
    else:
        return marker_locations


def get_marker_location(
    image,
    mask,
    channelnames=None,
    min_GP_score=0.5,
    alpha=0.05,
    min_p=0.75,
    n_bootstrap=1000,
):
    df = calculate_score(image, mask, channelnames=channelnames)
    marker_locations = get_marker_location_df(
        df,
        channelnames,
        min_GP_score=min_GP_score,
        alpha=alpha,
        min_p=min_p,
        n_bootstrap=n_bootstrap,
    )
    return df, marker_locations
