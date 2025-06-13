import numba
import numpy as np
import scipy
import polars as pl
from nyxus import Nyxus
from cautils.gettisord import G

MARKER_LOCATIONS = {
    "Nuclear": 0,
    "Membrane": 1,
    "Whole_cell": 2
}

# x = np.arange(1000)/250-2
# y = np.array([-np.abs(np.random.normal(i, 0.1, 1)) for i in x]).flatten()
# y = y - np.min(y)
# get_smooth_max(x,y)
# get_score(x,y)

@numba.njit(cache=True)
def get_smooth_max(x,y, nmaxdist=10, nmindist=25, minp=0.01):
    y_order = np.argsort(y)
    y_sorted = y[y_order]
    x_sorted = x[y_order]
    xq99 = np.quantile(x_sorted, 0.99)
    max_distance = xq99 / nmaxdist 
    min_distance = xq99 / nmindist 
    tmp=x_sorted[int(np.round(len(y_sorted)*0.95)):]
    initx = np.partition(tmp, tmp.size//2)[tmp.size//2]
    tdist = np.abs(initx-x_sorted)
    used_max_distance = np.nanmax([np.nanmin([np.nanquantile(tdist, minp), max_distance]), min_distance])
    ws = 1-np.clip(tdist/used_max_distance,0,1)
    return np.array([initx, np.sum(y_sorted*ws)/np.sum(ws)])


def do_smooth(x,y, nmaxdist=10, nmindist=25, minp=0.01):
    max_distance = np.quantile(x, 0.99) / nmaxdist 
    min_distance = np.quantile(x, 0.99) / nmindist 
    y_smooth = np.zeros_like(y)
    for i in range(len(x)):
        tdist = scipy.spatial.distance.cdist(x[[i],np.newaxis], x[:,np.newaxis], metric='euclidean')
        used_max_distance = np.nanmax([np.nanmin([np.nanquantile(tdist, minp), max_distance]), min_distance])
        ws = 1-np.clip(tdist/used_max_distance,0,1)
        y_smooth[i] = np.sum(y*ws)/np.sum(ws)
    # is_outlier = np.abs((y_smooth-y))>0.05
    # y_smooth = np.zeros_like(y)
    # for i in range(len(x)):
    #     tdist = scipy.spatial.distance.cdist(x[[i],np.newaxis], x[:,np.newaxis], metric='euclidean')
    #     ws = 1-np.clip(tdist/max_distance,0,1)
    #     ws[0,is_outlier] = 0
    #     y_smooth[i] = np.sum(y*ws)/np.sum(ws)
    return y_smooth

@numba.njit(cache=True)
def get_score(x,y):
    # y_smooth =  do_smooth(x, y)
    # max_idx = np.argmax(y_smooth)
    # x_max = x[max_idx]
    # y_max = y_smooth[max_idx]
    xy_max = get_smooth_max(x,y, nmaxdist=10, nmindist=25, minp=0.01)

    y_new = y.copy()
    y_new[np.logical_or(y>xy_max[1],x<xy_max[0])] = xy_max[1]
    y_new = 1-y_new/y_new.max()
    return y_new

def calculate_score(image, mask, channelnames=None, fdr_control=True, n_iter=0, radius=None):
    assert image.ndim == 3, "Image must be a 3D numpy array (channels, height, width)"
    assert mask.ndim == 2, "Mask must be a 2D numpy array (height, width)"
    assert image.shape[1:] == mask.shape, "Image and mask dimensions must match"
    if channelnames is None:
        channelnames = [f"Channel_{i}" for i in range(image.shape[0])]
    elif len(channelnames) != image.shape[0]:
        raise ValueError("Length of channelnames must match the number of channels in the image")

    image_G = np.zeros_like(image, dtype=float)
    image_GP = np.zeros_like(image, dtype=float)
    for chind in range(len(channelnames)):
        GZi, GP = G(image[chind,:,:].astype(float), n_iter=n_iter, radius=radius)
        image_G[chind,:,:] = GZi
        image_GP[chind,:,:] = GP
    image_GP_hot = 1-image_GP.copy()/2
    image_GP_hot[image_G<0] = 0.5+(0.5-image_GP_hot[image_G<0])
    image_GP_hot = 1-image_GP_hot
    if fdr_control:
        for chind in range(len(channelnames)):
            image_GP_hot[chind,:,:] = scipy.stats.false_discovery_control(image_GP_hot[chind,:,:])
        for chind in range(len(channelnames)):
            image_GP[chind,:,:] = scipy.stats.false_discovery_control(image_GP[chind,:,:])

    featurels = ["MEAN","EDGE_MEAN_INTENSITY","AREA_PIXELS_COUNT","PERIMETER"]
    to_rescale_features = ["MEAN", "EDGE_MEAN_INTENSITY"]
    index_features = [n for n in featurels if n not in to_rescale_features]
    nyx = Nyxus(featurels, n_feature_calc_threads=16 )
    channelnames_ls = channelnames.tolist() + [
        f"{name}_GP" for name in channelnames.tolist()
    ] + [
        f"{name}_GPhot" for name in channelnames.tolist()
    ]
    features1 = nyx.featurize(
        np.vstack([image*1e6,image_GP*1e6,image_GP_hot*1e6]), # mutliply by 1e6 to avoid numerical issues
        np.stack([mask for i in range(3*image.shape[0])]),
        intensity_names=channelnames_ls,
        label_names=channelnames_ls
        )
    df = pl.from_pandas(features1).pivot( 
        "intensity_image",
        index=["ROI_label"] + index_features,
        values=to_rescale_features,
    ).rename({
        "ROI_label": "ObjectNumber"
    }).rename({
        f"{fe}_{ch}": f"{ch}_{fe}"
        for fe in to_rescale_features for ch in channelnames_ls
    }).with_columns(**{ # rescale features
        f"{ch}_{fe}": pl.col(f"{ch}_{fe}") * 1e-6
        for fe in to_rescale_features for ch in channelnames_ls
    }).rename({
        f"{ch}_EDGE_MEAN_INTENSITY": f"{ch}_EDGE_MEAN"
        for ch in channelnames_ls
    }).with_columns(**{ # sum intensity
        f"{ch}_SUM": pl.col(f"{ch}_MEAN") * pl.col("AREA_PIXELS_COUNT")
        for ch in channelnames_ls
    }).with_columns(**{ # core mean intensity
        f"{ch}_CORE_MEAN": (pl.col(f"{ch}_SUM")-(pl.col(f"{ch}_EDGE_MEAN") * pl.col("PERIMETER"))) / pl.col("AREA_PIXELS_COUNT")
        for ch in channelnames_ls
    })
    df = df.with_columns(**{ # Calculate scores for each channel
        f"{ch}_GP_{m}_score": get_score(df[f'{ch}_{m}'].to_numpy(), df[f'{ch}_GP_{m}'].to_numpy()) for ch in channelnames for m in ["MEAN", "EDGE_MEAN", "CORE_MEAN"]
    }).with_columns(**{ # Calculate differences between scores Edge and Core
            f'{ch}_GP_MEAN_diff_score': pl.col(f'{ch}_GP_CORE_MEAN_score') - pl.col(f'{ch}_GP_EDGE_MEAN_score') for ch in channelnames
    }).with_columns(**{ # pixels count
            'AREA': pl.col('AREA_PIXELS_COUNT'),
            'AREA_EDGE': pl.col('PERIMETER'),
            'AREA_CORE': pl.col('AREA_PIXELS_COUNT') - pl.col('PERIMETER'),
    }).with_columns( # weights
        pl.min_horizontal("AREA_EDGE", "AREA_CORE").alias("AREA_MIN_EDGE_CORE")
    )
    return df




def bootstrap_weighted_proportion(data, weights, n_bootstrap=1000, alpha=0.05):
    """Bootstrap approach for weighted proportion testing"""
    
    def weighted_proportion(values, wts):
        indicators = (values > 0).astype(float)
        return np.sum(indicators * wts) / np.sum(wts)
    
    # Original weighted proportion
    orig_prop = weighted_proportion(data, weights)
    
    # Bootstrap resampling
    bootstrap_props = []
    for _ in range(n_bootstrap):
        # Sample with replacement using weights as probabilities
        prob_weights = weights / np.sum(weights)
        indices = np.random.choice(len(data), size=len(data), 
                                 p=prob_weights, replace=True)
        
        boot_data = data[indices]
        boot_weights = weights[indices]
        
        boot_prop = weighted_proportion(boot_data, boot_weights)
        bootstrap_props.append(boot_prop)
    
    # Confidence interval
    ci_lower = np.quantile(bootstrap_props, alpha/2)
    ci_upper = np.quantile(bootstrap_props, 1-alpha/2)
    
    return orig_prop, ci_lower, ci_upper, bootstrap_props



def get_marker_location_df(df, channelnames, min_GP_score=0.5, alpha=0.05, min_p=0.75,n_bootstrap=1000):
    marker_locations = []
    for ch in channelnames:
        # Filter based on GP scores
        dfsub1 = df.filter(
                (pl.col(f'{ch}_GP_EDGE_MEAN_score')>min_GP_score) | (pl.col(f'{ch}_GP_CORE_MEAN_score')>min_GP_score)
            ).select(
                pl.col('ObjectNumber'),
                pl.col('AREA_MIN_EDGE_CORE'),
                pl.col(f'{ch}_GP_MEAN_diff_score')
        )
        data = dfsub1[f"{ch}_GP_MEAN_diff_score"].to_numpy()
        weights = (dfsub1["AREA_MIN_EDGE_CORE"].to_numpy()/np.sum(dfsub1["AREA_MIN_EDGE_CORE"].to_numpy()))*dfsub1.shape[0]

        prop, ci_low, ci_high, boot_props = bootstrap_weighted_proportion(data, weights, n_bootstrap=n_bootstrap, alpha=alpha)
        if ci_low > min_p:
            marker_locations.append("Nuclear")
        elif 1-ci_high > min_p:
            marker_locations.append("Membrane")
        else:
            marker_locations.append("Whole_cell")
    return marker_locations



def get_marker_location(image, mask, channelnames=None, min_GP_score=0.5, alpha=0.05, min_p=0.75,n_bootstrap=1000):
    df = calculate_score(image, mask, channelnames=channelnames)
    marker_locations = get_marker_location_df(df, channelnames, min_GP_score=min_GP_score, alpha=alpha, min_p=min_p, n_bootstrap=n_bootstrap)
    return df, marker_locations