import numpy as np
import skfda
import scipy
import matplotlib.pyplot as plt
import cautils.permutations as cauperm
import cautils.radial_intensities as caucirc
import cautils.intensities as cauint
import polars as pl
import numpy.typing as npt

class Experiment:
    def __init__(self, image: npt.NDArray, mask: npt.NDArray, mask_nuc: None | npt.NDArray = None, channelnames: None | list[str] = None, resolution: float = 1.0):
        assert mask.ndim == 2, f"mask has to be 2 dimensional, current dimensions = {mask.ndim}"
        assert image.ndim == 3, f"image has to be 3 dimensional, current dimensions = {image.ndim}"
        assert mask.shape == image.shape[1:], f"mask and image have to have the same shape"
        if mask_nuc is None:
            mask_nuc = mask.copy()
        else:
            assert mask_nuc.ndim == 2, f"mask_nuc has to be 2 dimensional, current shape = {mask_nuc.shape}"
            assert mask_nuc.shape == mask.shape, f"mask_nuc and mask have to have the same shape"
        if channelnames is None:
            channelnames = [f"C{i}" for i in range(image.shape[0])]
        else:
            assert len(channelnames) == image.shape[0], f"len(channelnames) = {len(channelnames)}, image.shape[0] = {image.shape[0]}"
        self.image: npt.ArrayLike = image
        self.mask: npt.ArrayLike = mask
        self.mask_nuc: npt.ArrayLike = mask_nuc
        self.channelnames: list[str] = channelnames
        self.resolution: float = resolution

class Intensity:
    def __init__(self, experiment: Experiment, intensitytype: str = "mean"):
        self.intensities = cauint.get_cell_intensities(experiment.image, experiment.mask, experiment.channelnames, type = intensitytype)
        self.intensitytype = intensitytype

class RadialIntensity:
    def __init__(self, rmax: float = 12, nangles : int = 8, scale: int = 1, resolution: float = 1.0):
        self.create_kernel_list(rmax=rmax, nangles=nangles, scale=scale, resolution=resolution)

    def create_kernel_list(self, rmax: float = 12, nangles : int = 8, scale: int = 1, resolution: float = 1.0):
        
        rmax = int(rmax/resolution)
        phis = np.linspace(np.pi/nangles,2*np.pi-np.pi/nangles,nangles, dtype=np.float64)-np.pi/(nangles)
        rs = np.linspace(1,rmax*scale,int(np.round(rmax*resolution)), dtype=np.float64)
        epsphi = np.float64(np.pi/len(phis)-1e-6)
        tis = (rmax*scale*2+1, rmax*scale*2+1)

        tils = caucirc.create_kernel_list(tis=tis, rs=rs, phis=phis, epsphi=epsphi)
        tidls = caucirc.create_kernel_diff_list(tils = tils)
        tidnls = [(tidls[i]*(i+1)).astype(np.uint64) for i in range(len(tidls))]

        self.phis = phis
        self.rs = rs
        self.scale = scale
        self.rmax = rmax
        self.nangles = nangles
        self.kernel_list = np.sum(np.stack(tidnls, axis=0), axis=0) #tidn


    def calculate_radial_intensities(self, experiment: Experiment, intensity: Intensity):
        centroidarr, bboxarr = caucirc.bbox_centroids(experiment.mask_nuc)
        self.centroidarr = centroidarr
        self.bboxarr = bboxarr

        mask_all = np.stack([experiment.mask, experiment.mask, experiment.mask], axis=0)
        mask_all_nuc = np.stack([experiment.mask_nuc, experiment.mask_nuc, experiment.mask_nuc], axis=0)

        # create polar intensities
        radial_intensities = np.zeros((len(centroidarr), experiment.image.shape[0]+10, len(self.phis), len(self.rs)), dtype=np.float64)
        for ind, cellid in enumerate(intensity.intensities['ObjectNumber'].to_numpy()):
            subimg, p0 = caucirc.get_subimg(experiment.image, mask_all, mask_all_nuc, cellid, centroidarr[ind,:], bboxarr[ind,:], rmax=self.rmax, scale=self.scale)
            _ , radial_intensities[ind,:,:,:] = caucirc.calculate_radial_intensities(subimg, self.kernel_list, self.rs, self.phis, p0)
        self.intensities = radial_intensities


class Permutation:
    def __init__(self, nangles: int = 8, dmax: int = 3):
        self.nangles = nangles
        self.dmax = dmax
        self.create_permutation_template(nangles=self.nangles, dmax=self.dmax)
        self.calculate_permutation_template_stats()
    
    def create_permutation_template(self, nangles: int = 8, dmax: int = 3):
        assert dmax > 0, "dmax must be greater than 0"
        assert dmax < 10, "dmax must be less than 4"

        self.combs_arr = cauperm.get_combinations(list(range(nangles)), return_array=True)

        allowed_combinations_ls = []
        for i in range(1,dmax+1):
            allowed_combinations_ls.append(cauperm.get_combinations_lx_filter(self.combs_arr, i))
        self.allowed_combinations_ls = allowed_combinations_ls

        self.allowed_combinations = cauperm.get_combinations_lx_filter(self.combs_arr, dmax)
        self.perm_dmax = dmax

    def calculate_permutation_template_stats(self):
        combs = cauperm.get_combinations(list(range(self.nangles)), return_array=False)

        self.perm_mean_angles_full = np.array([ scipy.stats.circmean([c for ac in self.allowed_combinations[i] for c in combs[ac]], high=8-1e-12, low=0) for i in range(self.allowed_combinations.shape[0]) ])

        self.perm_std_angles_full = np.array([ scipy.stats.circstd([c for ac in self.allowed_combinations[i] for c in combs[ac]], high=8-1e-12, low=0) for i in range(self.allowed_combinations.shape[0]) ])

        self.perm_max_len_full = np.array([ sum([len(combs[ac])>0 for ac in self.allowed_combinations[i]]) for i in range(self.allowed_combinations.shape[0]) ])

        self.perm_n_pixels_full = np.array([ sum([len(combs[ac]) for ac in self.allowed_combinations[i]]) for i in range(self.allowed_combinations.shape[0]) ])

        def len_per_layer(combs, acl):
            return np.diff(np.array([len(combs[ac]) for ac in acl]+[0]))*-1
        self.perm_mean_len_full = np.array([np.sum(np.arange(1,self.allowed_combinations[i].shape[0]+1) * len_per_layer(combs, self.allowed_combinations[i]) )/sum(len_per_layer(combs, self.allowed_combinations[i])) for i in range(len(self.allowed_combinations))])

    def calculate_permutation(self, experiment: Experiment, radial_intensity: RadialIntensity, idx: int = 0, channelname: None | str | list[str] = None, normalize: str = "all", offset: int = 0, maxdist: None|int = None) -> np.ndarray:
        if channelname is None:
            channelname = experiment.channelnames
        if not isinstance(channelname, list):
            channelname = [channelname]
        if not all([ch in experiment.channelnames for ch in channelname]):
            raise ValueError("Not all channelnames are present")
        if maxdist is None:
            maxdist = self.perm_dmax
        assert maxdist != 0

        chind = np.array([i for ch,i in zip(experiment.channelnames, range(len(experiment.channelnames))) if ch in channelname])
        chind = np.concatenate([chind, np.arange(radial_intensity.intensities.shape[1]-10,radial_intensity.intensities.shape[1])])
        return cauperm.get_boundary_permutation_marker(radial_intensity.intensities[idx,chind,:,:], self.combs_arr, self.allowed_combinations_ls[abs(maxdist)-1], ch=-2, thr=1, maxdist=maxdist, normalize=normalize, offset=offset)


    def calculate_permutation_distributions(self, experiment: Experiment, radial_intensity: RadialIntensity, intensity: Intensity, channelname: None | str | list[str] = None, normalize: str = "all", offset: int = 0, maxdist: None|int = None) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(channelname, list):
            assert len(channelname)==1, "only a single channel can be specified"
        if channelname is None:
            channelname = [experiment.channelnames[0]]
        if not isinstance(channelname, list):
            channelname = [channelname]
        if not all([ch in experiment.channelnames for ch in channelname]):
            raise ValueError("Not all channelnames are present")
        if maxdist is None:
            maxdist = self.perm_dmax
        assert maxdist != 0

        chind = np.array([i for ch,i in zip(experiment.channelnames, range(len(experiment.channelnames))) if ch in channelname])
        chind = np.concatenate([chind, [radial_intensity.intensities.shape[1]-2]]).astype(int)

        hist_data, hist_grid_points = cauperm.get_boundary_permutation_marker_multiple(radial_intensity.intensities[:,chind,:,:], self.combs_arr, self.allowed_combinations, intensity.intensities[:,channelname[0]].to_numpy(), ch=-1, thr=1, maxdist=maxdist, normalize=normalize, offset=offset)

        return hist_data[:,1:], hist_grid_points[1:]
        # self.fd = skfda.FDataGrid(
        #     data_matrix=hist_data[:,1:],
        #     grid_points=hist_grid_points[1:],
        # )




class Sample:
    def __init__(self, image: np.ndarray, mask: np.ndarray, mask_nuc: None | np.ndarray = None, channelnames: None | list[str] = None, resolution: float = 1.0):
        self.experiment = Experiment(image=image, mask=mask, mask_nuc=mask_nuc, channelnames=channelnames, resolution=resolution)

        self.combs_arr = None
        self.allowed_combinations = None
        self.perm_dmax = None

        self.perm_mean_angles_full = None
        self.perm_std_angles_full = None
        self.perm_max_len_full = None
        self.perm_n_pixels_full = None


    def get_cell_image(self, ind, rmax:int=12, return_centroid: bool = False):
        mask_all = np.stack([self.experiment.mask, self.experiment.mask, self.experiment.mask], axis=0)
        mask_all_nuc = np.stack([self.experiment.mask_nuc, self.experiment.mask_nuc, self.experiment.mask_nuc], axis=0)
        return caucirc.get_subimg_all(ind, ind+1, self.experiment.image, mask_all, mask_all_nuc, rmax=rmax, scale=1, return_centroid=return_centroid)

    def calculate_cell_intensities(self, intensitytype = "mean"):
        self.intensity = Intensity(self.experiment, intensitytype=intensitytype)

    def calculate_radial_intensities(self, rmax: float = 12, nangles : int = 8, scale: int = 1):
        self.radial_intensity = RadialIntensity(rmax=rmax, nangles=nangles, scale=scale, resolution=self.experiment.resolution)
        self.radial_intensity.calculate_radial_intensities(self.experiment, self.intensity)

    def prepare_permutation(self, dmax: int = 3):
        self.permutation = Permutation(nangles=self.radial_intensity.nangles, dmax=dmax)

    def calculate_permutation(self, idx: int = 0, channelname: None | str | list[str] = None, normalize: str = "all", offset: int = 0, maxdist: None|int = None) -> np.ndarray:
        return self.permutation.calculate_permutation(experiment=self.experiment, radial_intensity=self.radial_intensity, idx=idx, channelname=channelname, normalize=normalize, offset=offset, maxdist=maxdist)

    def calculate_permutation_distributions(self, channelname: None | str = None, normalize: str = "all", offset: int = 0, maxdist: None|int = None) -> tuple[np.ndarray, np.ndarray]:
        return self.permutation.calculate_permutation_distributions(self.experiment, self.radial_intensity, self.intensity, channelname=channelname, normalize=normalize, offset=offset, maxdist=maxdist)
    
#     def plot_cell(self, cellind: int, channelname: str, offset: int = 0, extra=False):
#         from sklearn.preprocessing import SplineTransformer
#         from sklearn.linear_model import Ridge, LinearRegression
#         from sklearn.pipeline import make_pipeline
#         import cv2
#         import matplotlib.patches as patches

#         chind = [i for ch,i in zip(self.experiment.channelnames, range(len(self.experiment.channelnames))) if ch in channelname][0]

#         if extra:
#             fig, ax = plt.subplots(2,4)
#         else:
#             fig, ax = plt.subplots(2,3)
#         timg = self.get_cell_image(cellind, rmax=self.rmax)

#         obn = self.intensity.intensities.with_row_index().filter(pl.col("index")==cellind)["ObjectNumber"][0]

#         cell_bounds = cv2.findContours((timg[-10,:,:]==obn).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)
#         pcm = ax[0,0].imshow(timg[chind,:,:])
#         poly = patches.Polygon(cell_bounds, edgecolor='white', facecolor='none')
#         ax[0,0].add_patch(poly)
#         # fig.colorbar(pcm, ax=ax[0,0], location='bottom')
#         ax[0,1].imshow(timg[-10,:,:])
#         ax[1,0].imshow(self.radial_intensities[cellind,0,:,:])
#         ax[1,1].imshow(self.radial_intensities[cellind,-5,:,:])
#         ax[1,2].scatter(self.perm_mean_angles_full, self.calculate_permutation(cellind, offset=offset, channelname=channelname)[:,0], c=self.perm_mean_len_full, s=1)

#         x = self.perm_mean_angles_full.copy()
#         y = self.calculate_permutation(cellind, offset=offset, channelname=channelname)[:,0]

#         X_train = x.reshape(-1, 1)
#         y_train = y
#         X_train_ls = [x[self.perm_max_len_full>-offset].reshape(-1, 1),x[self.perm_max_len_full<=-offset].reshape(-1, 1)]
#         y_train_ls = [y[self.perm_max_len_full>-offset],y[self.perm_max_len_full<=-offset]]
#         X_plot = np.linspace(0, 8, 1000)[:, np.newaxis]
#         colors = ['blue', 'red']
#         for i, X_train, y_train in zip([0,1], X_train_ls, y_train_ls):
#             try:
#                 model = make_pipeline(SplineTransformer(n_knots=16, degree=6, extrapolation="periodic"), Ridge(alpha=1e-1))
#                 model.fit(X_train, y_train)
#                 y_plot = model.predict(X_plot)
#                 ax[0,2].scatter(X_train, y_train, s=1, c=colors[i])
#                 ax[0,2].plot(X_plot, y_plot, c=colors[i])
    
#                 # model = make_pipeline(SplineTransformer(n_knots=2, degree=1, extrapolation="periodic"), LinearRegression())
#                 model = make_pipeline(SplineTransformer(n_knots=2, degree=1, extrapolation="periodic"), Ridge(alpha=1e-1))
#                 model.fit(X_train, y_train)
#                 y_plot = model.predict(X_plot)
#                 ax[0,2].plot(X_plot, y_plot, c=colors[i])
#             except:
#                 pass
        
#         if extra:
#             import seaborn as sns
#             sns.scatterplot(self.intensity.intensities, x='ct',y=channelname, hue='area', ax=ax[1,3])
#             ax[1,3].scatter(self.intensity.intensities['ct'].to_numpy()[cellind],self.intensity.intensities[channelname].to_numpy()[cellind], color='red')

#         return fig



if __name__ == "__main__":
    import tifffile
    # read omexml, for channelnames only
    imgpath_imc = "/home/retger/Nextcloud/Projects/celltyping_imc/IMC-IF/ROI001_035_PS15.19650-B3/registered_IMC/registered_IMC.ome.tiff"
    from ome_types import from_tiff
    ome = from_tiff(imgpath_imc)
    channelnames=np.array([x.name for x in ome.images[0].pixels.channels])
    tb = np.array([ch in np.array(list(channelnames[:1]) + list(channelnames[8:-2])) for ch in channelnames])
    channelnames = channelnames[tb]
    channelnames[np.argmax(channelnames=="DNA")] = "DNA1"
    channelnames[np.argmax(channelnames=="DNA")] = "DNA2"
    channelnames[np.argmax(channelnames=="CD14")] = "CD14_1"
    channelnames[np.argmax(channelnames=="CD14")] = "CD14_2"
    part=6
    mask_path = f"/home/retger/Nextcloud/Projects/celltyping_imc/IMC-IF/ROI001_035_PS15.19650-B3/IMC_mask/mask_whole_{part}.tiff"
    mask_nuc_path = f"/home/retger/Nextcloud/Projects/celltyping_imc/IMC-IF/ROI001_035_PS15.19650-B3/IMC_mask/mask_nuc_{part}.tiff"
    image_path = f"/home/retger/Nextcloud/Projects/celltyping_imc/IMC-IF/ROI001_035_PS15.19650-B3/IMC/imc_{part}.tiff"

    # Read the TIFF file
    image = tifffile.imread(image_path)
    image = image[tb,:,:]
    mask_all = tifffile.imread(mask_path)[[0,2,3],:,:]
    mask = mask_all[0,:,:].astype(int)
    mask_all_nuc = tifffile.imread(mask_nuc_path)[[0,2,3],:,:]
    mask_nuc = mask_all_nuc[0,:,:].astype(int)

    sa = Sample(image, mask, mask, list(channelnames))
    # # sa = Sample(image, mask, mask_nuc, channelnames)

    sa.calculate_cell_intensities()
    sa.calculate_radial_intensities()
    sa.prepare_permutation(dmax=3)
    sa.calculate_permutation(0, "CD14_1")
    sa.calculate_permutation_distributions("CD14_1")
#     sa.create_permutation_template()
#     sa.calculate_permutation_template_stats()
#     perm = sa.calculate_permutation(2)

#     idx = 5
#     minshri = cauperm.get_max_shrinkage(sa.radial_intensities[idx,:,:,:])
#     minshri
#     perm = sa.calculate_permutation(idx, maxdist=-minshri)

#     perm = sa.calculate_permutation(971, "DNA1")
#     perminv = sa.calculate_permutation(971, "DNA1", invert=True)

#     plt.scatter(perm[:,0], perminv[:,0], c=sa.perm_mean_len_full)
#     plt.show()

#     plt.scatter(sa.perm_mean_angles_full, perm[:,0], c="blue")
#     plt.scatter(sa.perm_mean_angles_full, perminv[:,0], c="red")
#     plt.show()

#     # channelname = sa.channelnames[0]
#     # sa.calculate_permutation(0,channelname)
#     # sa.calculate_permutation_distributions(channelname)

#     # channelname = sa.channelnames[0]
#     # sa.calculate_permutation(0,channelname)
#     # sa.calculate_permutation_distributions(channelname)
#     # sa.fd.plot()
#     # plt.show()

#     import tifffile
#     image_path = "/home/retger/Nextcloud/Projects/celltyping_imc/simulation/example_data_182_countsimgprlr.tiff"
#     image = tifffile.imread(image_path)
#     image_path = "/home/retger/Nextcloud/Projects/celltyping_imc/simulation/example_data_182_ctimgprlr.tiff"
#     ctimage = tifffile.imread(image_path)
#     mask_path = "/home/retger/Nextcloud/Projects/celltyping_imc/simulation/example_data_182_padmatlrpr2.5.tiff"
#     mask = tifffile.imread(mask_path)
#     slice_nr = mask.shape[0]//2
#     mask = mask[slice_nr,:,:].astype(int)
#     import skimage
#     rps = skimage.measure.regionprops(mask, ctimage[0,:,:])
#     import polars as pl
#     ctinfo = pl.DataFrame({"ct": [rp.mean_intensity for rp in rps], "ObjectNumber": [rp.label for rp in rps]})

#     sa = Sample(image, mask, mask)

#     sa.calculate_cell_intensities()
#     sa.intensities = ctinfo.join(sa.intensities, on="ObjectNumber")
    
#     sa.create_kernel_list(rmax=12)
#     sa.calculate_radial_intensities()
#     print("create permutation template")
#     sa.create_permutation_template(4)
#     print("done")
#     import time
#     print("calculate permutation")
#     t1 = time.time()
#     sa.calculate_permutation(0)
#     t2 = time.time()
#     print(f"done in: {t2-t1}")
#     print("calculate permutation")
#     t1 = time.time()
#     sa.calculate_permutation(0)
#     t2 = time.time()
#     print(f"done in: {t2-t1}")
#     channelname = sa.channelnames[0]
#     print("calculate_permutation_distributions")
#     t1 = time.time()
#     sa.calculate_permutation_distributions(channelname)
#     t2 = time.time()
#     print(f"done in: {t2-t1}")
#     print("calculate_permutation_distributions")
#     t1 = time.time()
#     sa.calculate_permutation_distributions(channelname)
#     t2 = time.time()
#     print(f"done in: {t2-t1}")

#     sa.calculate_permutation_template_stats()

#     tmp = sa.intensities
#     tmp.columns
#     tmp = tmp.with_row_index()
#     tmp.filter(pl.col("ct")>1.5,pl.col("ct")<2, pl.col("area")>10, pl.col("C0")<5)

#     tmp.filter(pl.col("ObjectNumber")==3085)
    
#     cellind = 101
#     cellind = 88
#     cellind = 255
#     offset = -2
#     fig = sa.plot_cell(cellind, offset, extra=True)
#     plt.show()

#     import seaborn as sns
#     sns.scatterplot(sa.intensities, x='ct',y=channelname, hue='area')
#     plt.scatter(sa.intensities['ct'].to_numpy()[cellind],sa.intensities[channelname].to_numpy()[cellind], color='red')
#     plt.show()

#     plt.hist(sa.intensities['ct'].to_numpy(), bins=100)
#     plt.show()
#     plt.hist(sa.intensities[channelname].to_numpy(), bins=20)
#     plt.vlines(sa.intensities[channelname][cellind], 0, 100, color='red')
#     plt.show()



#     t = np.linspace(x.min(), x.max(), 100)
#     tmp = np.argsort(x)
#     x=x[tmp]
#     y=y[tmp]
#     spl = splrep(x, y, s=1)
#     bspl = BSpline(*spl)
#     # plt.scatter(x, y)
#     plt.plot(t, bspl(t))
#     plt.show()


#     import seaborn as sns
#     import pandas as pd
#     data = pd.DataFrame({"perm_mean_angles_full": sa.perm_mean_angles_full, "perm_max_len_full": sa.perm_max_len_full, "intensity": sa.calculate_permutation(cellind, offset=-1)[:,0]})
#     sns.lmplot(x="perm_mean_angles_full", y="intensity", data=data, hue="perm_max_len_full", lowess=True)
#     plt.show()

# # if False:
#     channelname = sa.channelnames[0]
#     sa.calculate_permutation_distributions(channelname)
#     sa.fd.plot()
#     plt.show()

#     # from skfda.misc.regularization import L2Regularization
#     # from skfda.misc.operators import LinearDifferentialOperator
#     # basis = skfda.representation.basis.BSplineBasis(n_basis=51)
#     # fd = fd.to_basis(basis)
#     # fd.plot()
#     # plt.show()

#     from skfda.preprocessing.smoothing import KernelSmoother
#     from skfda.misc.hat_matrix import (
#         KNeighborsHatMatrix,
#         LocalLinearRegressionHatMatrix,
#         NadarayaWatsonHatMatrix,
#     )
#     fd_os = KernelSmoother(
#         kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=0.05),
#     ).fit_transform(fd)
#     fd_os.plot()
#     plt.show()


#     from skfda.exploratory.visualization import FPCAPlot
#     from skfda.preprocessing.dim_reduction import FPCA
#     fpca_discretized = FPCA(n_components=3)
#     fpca_discretized.fit(fd_os)
#     scores = fpca_discretized.transform(fd_os)

#     fpca_discretized.components_.plot()
#     plt.show()

#     plt.scatter(np.arange(1, len(fpca_discretized.explained_variance_ratio_)+1), fpca_discretized.explained_variance_ratio_)
#     plt.show()

#     # plt.scatter(scores[:,0], scores[:,1], c=np.arcsinh(sa.intensities[channelname].to_list()))
#     plt.scatter(scores[:,0], scores[:,1], c=scores[:,2])
#     plt.show()
#     plt.scatter(scores[:,1], scores[:,2], c=scores[:,0])
#     plt.show()

#     fig, ax = plt.subplots(1,2)
#     fd_os[scores[:,0]>500].plot(axes=ax[0])
#     fd_os[scores[:,0]<-500].plot(axes=ax[1])
#     plt.show()

#     fd_smooth = fpca_discretized.inverse_transform(scores)
#     fd_smooth.plot()
#     plt.show()


#     from skfda.ml.clustering import FuzzyCMeans, KMeans
#     kmeans = KMeans(n_clusters=5, random_state=123)
#     kmeans.fit(fd_os)
#     print(kmeans.predict(fd_os))

#     from skfda.exploratory.visualization.clustering import (
#         ClusterMembershipLinesPlot,
#         ClusterMembershipPlot,
#         ClusterPlot,
#     )
#     ClusterPlot(kmeans, fd_os).plot()
#     plt.show()

#     fd_os[kmeans.predict(fd_os)==4].plot()
#     plt.show()

#     from skfda.ml.clustering import AgglomerativeClustering
#     clustering = AgglomerativeClustering(
#         linkage=AgglomerativeClustering.LinkageCriterion.COMPLETE,
#         n_clusters=5
#     )
#     clustering.fit(fd_os)
#     clustering.labels_.astype(np.int_)

#     fd_os.plot(group=clustering.labels_.astype(np.int_))
#     plt.show()

#     fuzzy_kmeans = FuzzyCMeans(n_clusters=5, random_state=123)
#     fuzzy_kmeans.fit(fd)
#     print(fuzzy_kmeans.predict_proba(fd))


#     fig, ax = plt.subplots(1,5)
#     for i in range(5):
#         ClusterMembershipPlot(fuzzy_kmeans, fd, sort=i,axes=ax[i]).plot()
#     plt.show()

#     FPCAPlot(
#     fd.mean(),
#     fpca_discretized.components_,
#     factor=30,
#     fig=plt.figure(figsize=(6, 2 * 4)),
#     n_rows=2,
#     ).plot()
#     plt.show()

#     skfda.misc.cosine_similarity_matrix(fd_os)

#     from skfda.misc.operators import SRSF
#     srsf = SRSF()
#     q = srsf.fit_transform(fd_os)
#     intens = np.arcsinh(sa.intensities[channelname].to_list())
#     fd_os.plot(gradient_criteria="l", linewidth=0.5, colormap='viridis')
#     plt.show()

#     import matplotlib as mpl

#     viridis = mpl.colormaps['viridis'].resampled(8)
#     normint = sa.intensities[channelname].to_numpy()/sa.intensities[channelname].max()
#     colors = viridis(normint)
#     colors = [mpl.colors.to_hex(colors[i]) for i in range(colors.shape[0])]
#     fd_os.plot(c=colors, linewidth=0.5)
#     plt.show()


