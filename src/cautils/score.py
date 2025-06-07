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


def do_smooth(x,y):
    max_distance = np.quantile(x, 0.99) / 25
    y_smooth = np.zeros_like(y)
    for i in range(len(x)):
        tdist = scipy.spatial.distance.cdist(x[[i],np.newaxis], x[:,np.newaxis], metric='euclidean')
        ws = 1-np.clip(tdist/max_distance,0,1)
        y_smooth[i] = np.sum(y*ws)/np.sum(ws)
    is_outlier = np.abs((y_smooth-y))>0.05
    y_smooth = np.zeros_like(y)
    for i in range(len(x)):
        tdist = scipy.spatial.distance.cdist(x[[i],np.newaxis], x[:,np.newaxis], metric='euclidean')
        ws = 1-np.clip(tdist/max_distance,0,1)
        ws[0,is_outlier] = 0
        y_smooth[i] = np.sum(y*ws)/np.sum(ws)
    return y_smooth

def get_score(x,y):
    y_smooth =  do_smooth(x, y)
    max_idx = np.argmax(y_smooth)
    x_max = x[max_idx]
    y_max = y_smooth[max_idx]

    y_new = y.copy()
    y_new[np.logical_or(y>y_max,x<x_max)] = y_max
    y_new = 1-y_new/y_new.max()
    return y_new

def calculate_score(image, mask, channelnames=None, fdr_control=True):
    assert image.ndim == 3, "Image must be a 3D numpy array (channels, height, width)"
    assert mask.ndim == 2, "Mask must be a 2D numpy array (height, width)"
    assert image.shape[1:] == mask.shape, "Image and mask dimensions must match"
    if channelnames is None:
        channelnames = [f"Channel_{i}" for i in range(image.shape[0])]
    elif len(channelnames) != image.shape[0]:
        raise ValueError("Length of channelnames must match the number of channels in the image")

    image_G = np.zeros_like(image, dtype=float)
    image_GP = np.zeros_like(image, dtype=float)
    for chind in range(3):
        GZi, GP = G(image[chind,:,:].astype(float), n_iter=0)
        image_G[chind,:,:] = GZi
        image_GP[chind,:,:] = GP
    image_GP_hot = 1-image_GP.copy()/2
    image_GP_hot[image_G<0] = 0.5+(0.5-image_GP_hot[image_G<0])
    image_GP_hot = 1-image_GP_hot
    if fdr_control:
        for chind in range(3):
            image_GP_hot[chind,:,:] = scipy.stats.false_discovery_control(image_GP_hot[chind,:,:])
        for chind in range(3):
            image_GP[chind,:,:] = scipy.stats.false_discovery_control(image_GP[chind,:,:])

    nyx = Nyxus(["MEAN","EDGE_MEAN_INTENSITY","AREA_PIXELS_COUNT","INTEGRATED_INTENSITY","PERIMETER"], n_feature_calc_threads=16 )
    df = None
    for i in range(len(channelnames)):
        features1 = nyx.featurize(image[i,:,:], mask, [channelnames[i]])
        features2 = nyx.featurize(image_GP[i,:,:]*1e6, mask, [channelnames[i]])
        features3 = nyx.featurize(image_GP_hot[i,:,:]*1e6, mask, [channelnames[i]])
        featuresdf = pl.DataFrame({
                "ObjectNumber": features1["ROI_label"].astype(np.int64),
                f"{channelnames[i]}_AREA": features1["AREA_PIXELS_COUNT"],
                f"{channelnames[i]}_PERIMETER": features1["PERIMETER"],
                f"{channelnames[i]}_MEAN": features1["MEAN"],
                f"{channelnames[i]}_SUM": features1["INTEGRATED_INTENSITY"],
                f"{channelnames[i]}_EDGE_MEAN": features1["EDGE_MEAN_INTENSITY"],
                f"{channelnames[i]}_CORE_MEAN": (features1["INTEGRATED_INTENSITY"]-(features1["EDGE_MEAN_INTENSITY"]*features1["PERIMETER"])) / features1["AREA_PIXELS_COUNT"],
                f"{channelnames[i]}_GP_MEAN": features2["MEAN"]*1e-6,
                f"{channelnames[i]}_GP_SUM": features2["INTEGRATED_INTENSITY"]*1e-6,
                f"{channelnames[i]}_GP_EDGE_MEAN": features2["EDGE_MEAN_INTENSITY"]*1e-6,
                f"{channelnames[i]}_GP_CORE_MEAN": (features2["INTEGRATED_INTENSITY"]-(features2["EDGE_MEAN_INTENSITY"]*features2["PERIMETER"])) / features2["AREA_PIXELS_COUNT"]*1e-6,
                f"{channelnames[i]}_GPhot_MEAN": features3["MEAN"]*1e-6,
                f"{channelnames[i]}_GPhot_SUM": features3["INTEGRATED_INTENSITY"]*1e-6,
                f"{channelnames[i]}_GPhot_EDGE_MEAN": features3["EDGE_MEAN_INTENSITY"]*1e-6,
                f"{channelnames[i]}_GPhot_CORE_MEAN": (features3["INTEGRATED_INTENSITY"]-(features3["EDGE_MEAN_INTENSITY"]*features3["PERIMETER"])) / features3["AREA_PIXELS_COUNT"]*1e-6,
        })
        if df is None:
            df = featuresdf
        else:
            df = df.join(featuresdf, on="ObjectNumber", how="inner")

    # Calculate scores for each channel
    df = df.with_columns(
        **{f"{ch}_GP_{m}_score": get_score(df[f'{ch}_{m}'].to_numpy(), df[f'{ch}_GP_{m}'].to_numpy()) for ch in channelnames for m in ["MEAN", "EDGE_MEAN", "CORE_MEAN"]}
    )
    # Calculate differences between scores Edge and Core
    df = df.with_columns(**{
            f'{ch}_GP_MEAN_diff_score': pl.col(f'{ch}_GP_CORE_MEAN_score') - pl.col(f'{ch}_GP_EDGE_MEAN_score') for ch in channelnames
        })
    # pixels count
    df = df.with_columns(**{
            'AREA': pl.col(f'{channelnames[0]}_AREA'),
            'AREA_EDGE': pl.col(f'{channelnames[0]}_PERIMETER'),
            'AREA_CORE': pl.col(f'{channelnames[0]}_AREA') - pl.col(f'{channelnames[0]}_PERIMETER'),
        })
    # weights
    df = df.with_columns(
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