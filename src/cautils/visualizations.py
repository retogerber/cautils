import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats

from .radial_intensities import get_kernel_from_output, to_euclid_coords


def to_euclid_coords_plot(ind, imcinmat, ifinmat, imcchannelnames, ifchannelnames, tidn=None, scale=1, show_mask=True, show_tidn=False):
    assert sum([show_mask, show_tidn]) <= 1
    ifcelleu = to_euclid_coords(ind, ifinmat, tidn, scale)
    tmp = to_euclid_coords(ind, imcinmat, tidn, scale)
    imccelleu = np.zeros((tmp.shape[0], ifcelleu.shape[1], ifcelleu.shape[2]))
    for i in range(tmp.shape[0]):
        imccelleu[i,:,:] = cv2.resize(tmp[i,:,:], (ifcelleu.shape[2], ifcelleu.shape[1]), interpolation=cv2.INTER_NEAREST)
    ifcelleu[ifcelleu==0] = np.nan
    imccelleu[imccelleu==0] = np.nan

    if show_mask or show_tidn:
        fig, ax = plt.subplots(2,3)
    else:
        fig, ax = plt.subplots(2,2)
    ch=np.argmax(imcchannelnames=="CD45RO")
    pcm = ax[0,0].imshow(imccelleu[ch,:,:])
    ax[0,0].set_title("IMC - CD45RO")
    fig.colorbar(pcm, ax=ax[0,0], location='bottom')
    ch=np.argmax(imcchannelnames=="DNA1")
    pcm = ax[0,1].imshow(imccelleu[ch,:,:])
    ax[0,1].set_title("IMC - DNA")
    fig.colorbar(pcm, ax=ax[0,1], location='bottom')
    if show_mask:
        ax[0,2].imshow(imccelleu[-5,:,:])
        ax[0,2].set_title("IMC - Cell")
    if show_tidn:
        tidn = get_kernel_from_output(imcinmat, scale=scale)
        ax[0,2].imshow(tidn)
        ax[0,2].set_title("IMC - Kernel")
    ch=np.argmax(ifchannelnames=="CD45")
    pcm = ax[1,0].imshow(ifcelleu[ch,:,:])
    ax[1,0].set_title("IF - CD45")
    fig.colorbar(pcm, ax=ax[1,0], location='bottom')
    ch=np.argmax(ifchannelnames=="DNA")
    pcm = ax[1,1].imshow(ifcelleu[ch,:,:])
    ax[1,1].set_title("IF - DNA")
    fig.colorbar(pcm, ax=ax[1,1], location='bottom')
    if show_mask:
        ax[1,2].imshow(ifcelleu[-5,:,:])
        ax[1,2].set_title("IF - Cell")
    if show_tidn:
        ax[1,2].imshow(tidn)
        ax[1,2].set_title("IMC - Kernel")
    plt.show()



def to_euclid_coords_plot_all(ind, imc_tsubimg, if_tsubimg, imcinmat, ifinmat, imcinscmat, ifinscmat, imcchannelnames, ifchannelnames, tidn=None, scale=1, return_fig=False, ch2={"IMC": "CD45RO", "IF": "CD45"}):

    # tsubimgexp = cv2.morphologyEx(imc_tsubimg[-5,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    imc_pts_cell = cv2.findContours(imc_tsubimg[-5,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)
    # tsubimgexp2 = cv2.morphologyEx(imc_tsubimg[-2,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    imc_pts_nuc = cv2.findContours(imc_tsubimg[-2,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)

    # tsubimgexp = cv2.morphologyEx(if_tsubimg[-5,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    if_pts_cell = cv2.findContours(if_tsubimg[-5,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)
    # tsubimgexp2 = cv2.morphologyEx(if_tsubimg[-2,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    if_pts_nuc = cv2.findContours(if_tsubimg[-2,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)


    ifcelleu = to_euclid_coords(ind, ifinmat, tidn, scale)
    tmp = to_euclid_coords(ind, imcinmat, tidn, scale)
    imccelleu = np.zeros((tmp.shape[0], ifcelleu.shape[1], ifcelleu.shape[2]))
    for i in range(tmp.shape[0]):
        imccelleu[i,:,:] = cv2.resize(tmp[i,:,:], (ifcelleu.shape[2], ifcelleu.shape[1]), interpolation=cv2.INTER_NEAREST)

    ifsccelleu = to_euclid_coords(ind, ifinscmat, tidn, scale)
    tmp = to_euclid_coords(ind, imcinscmat, tidn, scale)
    imcsccelleu = np.zeros((tmp.shape[0], ifsccelleu.shape[1], ifsccelleu.shape[2]))
    for i in range(tmp.shape[0]):
        imcsccelleu[i,:,:] = cv2.resize(tmp[i,:,:], (ifsccelleu.shape[2], ifsccelleu.shape[1]), interpolation=cv2.INTER_NEAREST)

    # tsubimgexp = cv2.morphologyEx(imcsccelleu[-5,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    imc_pts_cell_round = cv2.findContours(imcsccelleu[-5,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)

    # tsubimgexp = cv2.morphologyEx(ifsccelleu[-5,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    if_pts_cell_round = cv2.findContours(ifsccelleu[-5,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)

    import matplotlib.patches as patches
    fig, ax = plt.subplots(2, 6,figsize=(18, 8), dpi=100,layout='constrained')
    ch=np.argmax(imcchannelnames==ch2['IMC'])
    tsubimgdna = imc_tsubimg[ch,:,:].copy()#/np.max(imc_tsubimg[ch,:,:])
    pcm = ax[0,0].imshow(tsubimgdna)
    poly = patches.Polygon(imc_pts_cell, edgecolor='red', facecolor='none')
    ax[0,0].add_patch(poly)
    poly = patches.Polygon(imc_pts_nuc, edgecolor='lightblue', facecolor='none')
    ax[0,0].add_patch(poly)
    ax[0,0].set_title(f"IMC - {ch2['IMC']}")
    fig.colorbar(pcm, ax=ax[0,0], location='bottom')
    ch=np.argmax(imcchannelnames=="DNA1")
    tsubimgdna = imc_tsubimg[ch,:,:].copy()#/np.max(imc_tsubimg[ch,:,:])
    pcm = ax[0,1].imshow(tsubimgdna)
    poly = patches.Polygon(imc_pts_cell, edgecolor='red', facecolor='none')
    ax[0,1].add_patch(poly)
    poly = patches.Polygon(imc_pts_nuc, edgecolor='lightblue', facecolor='none')
    ax[0,1].add_patch(poly)
    ax[0,1].set_title("IMC - DNA")
    fig.colorbar(pcm, ax=ax[0,1], location='bottom')
    ch=np.argmax(ifchannelnames==ch2['IF'])
    tsubimgdna = if_tsubimg[ch,:,:].copy()#/np.max(if_tsubimg[ch,:,:])
    pcm = ax[1,0].imshow(tsubimgdna)
    poly = patches.Polygon(if_pts_cell, edgecolor='red', facecolor='none')
    ax[1,0].add_patch(poly)
    poly = patches.Polygon(if_pts_nuc, edgecolor='lightblue', facecolor='none')
    ax[1,0].add_patch(poly)
    ax[1,0].set_title(f"IF - {ch2['IF']}")
    fig.colorbar(pcm, ax=ax[1,0], location='bottom')
    ch=np.argmax(ifchannelnames=="DNA1")
    tsubimgdna = if_tsubimg[ch,:,:].copy()#/np.max(if_tsubimg[ch,:,:])
    pcm = ax[1,1].imshow(tsubimgdna)
    poly = patches.Polygon(if_pts_cell, edgecolor='red', facecolor='none')
    ax[1,1].add_patch(poly)
    poly = patches.Polygon(if_pts_nuc, edgecolor='lightblue', facecolor='none')
    ax[1,1].add_patch(poly)
    ax[1,1].set_title("IF - DNA")
    fig.colorbar(pcm, ax=ax[1,1], location='bottom')
    ch=np.argmax(imcchannelnames==ch2['IMC'])
    cellroundimgdna = imccelleu[ch,:,:].copy()#/np.max(imccelleu[ch,:,:])
    pcm = ax[0,2].imshow(cellroundimgdna)
    ax[0,2].set_title(f"IMC - {ch2['IMC']} polar")
    fig.colorbar(pcm, ax=ax[0,2], location='bottom')
    ch=np.argmax(imcchannelnames=="DNA1")
    cellroundimgdna = imccelleu[ch,:,:].copy()#/np.max(imccelleu[ch,:,:])
    pcm = ax[0,3].imshow(cellroundimgdna)
    ax[0,3].set_title("IMC - DNA polar")
    fig.colorbar(pcm, ax=ax[0,3], location='bottom')
    ch=np.argmax(ifchannelnames==ch2['IF'])
    cellroundimgdna = ifcelleu[ch,:,:].copy()#/np.max(ifcelleu[ch,:,:])
    pcm = ax[1,2].imshow(cellroundimgdna)
    ax[1,2].set_title(f"IF - {ch2['IF']} polar")
    fig.colorbar(pcm, ax=ax[1,2], location='bottom')
    ch=np.argmax(ifchannelnames=="DNA")
    cellroundimgdna = ifcelleu[ch,:,:].copy()#/np.max(ifcelleu[ch,:,:])
    pcm = ax[1,3].imshow(cellroundimgdna)
    ax[1,3].set_title("IF - DNA polar")
    fig.colorbar(pcm, ax=ax[1,3], location='bottom')

    ch=np.argmax(imcchannelnames==ch2['IMC'])
    cellroundimgdna = imcsccelleu[ch,:,:].copy()#/np.max(imcsccelleu[ch,:,:])
    pcm = ax[0,4].imshow(cellroundimgdna)
    ax[0,4].set_title(f"IMC - {ch2['IMC']} polar scaled")
    fig.colorbar(pcm, ax=ax[0,4], location='bottom')
    poly = patches.Polygon(imc_pts_cell_round, edgecolor='red', facecolor='none')
    ax[0,4].add_patch(poly)

    ch=np.argmax(imcchannelnames=="DNA1")
    cellroundimgdna = imcsccelleu[ch,:,:].copy()#/np.max(imcsccelleu[ch,:,:])
    pcm = ax[0,5].imshow(cellroundimgdna)
    ax[0,5].set_title("IMC - DNA polar scaled")
    fig.colorbar(pcm, ax=ax[0,5], location='bottom')
    poly = patches.Polygon(imc_pts_cell_round, edgecolor='red', facecolor='none')
    ax[0,5].add_patch(poly)

    ch=np.argmax(ifchannelnames==ch2['IF'])
    cellroundimgdna = ifsccelleu[ch,:,:].copy()#/np.max(ifsccelleu[ch,:,:])
    pcm = ax[1,4].imshow(cellroundimgdna)
    ax[1,4].set_title(f"IF - {ch2['IF']} polar scaled")
    fig.colorbar(pcm, ax=ax[1,4], location='bottom')
    poly = patches.Polygon(imc_pts_cell_round, edgecolor='red', facecolor='none')
    ax[1,4].add_patch(poly)

    ch=np.argmax(ifchannelnames=="DNA")
    cellroundimgdna = ifsccelleu[ch,:,:].copy()#/np.max(ifsccelleu[ch,:,:])
    pcm = ax[1,5].imshow(cellroundimgdna)
    ax[1,5].set_title("IF - DNA polar scaled")
    fig.colorbar(pcm, ax=ax[1,5], location='bottom')
    poly = patches.Polygon(imc_pts_cell_round, edgecolor='red', facecolor='none')
    ax[1,5].add_patch(poly)

    if return_fig:
        return fig
    plt.show()

def acf(x):
    return [scipy.stats.pearsonr(x[:], np.concatenate([x[sh:],x[:sh]])).statistic for sh in range(len(x))]
def acf2d(x):
    return np.stack([np.array(acf(x[:,i])) for i in range(x.shape[1])]).T

def cov(x):
    return [np.cov(np.stack([x[:], np.concatenate([x[sh:],x[:sh]])]))[1,0] for sh in range(len(x))]
def cov2d(x):
    return np.stack([np.array(cov(x[:,i])) for i in range(x.shape[1])]).T

def difl1(x):
    return [np.linalg.norm(x[:] - np.concatenate([x[sh:],x[:sh]]), ord=1) for sh in range(len(x))]
def difl12d(x):
    return np.stack([np.array(difl1(x[:,i])) for i in range(x.shape[1])]).T

def difl2(x):
    return [np.linalg.norm(x[:] - np.concatenate([x[sh:],x[:sh]]), ord=2) for sh in range(len(x))]
def difl22d(x):
    return np.stack([np.array(difl2(x[:,i])) for i in range(x.shape[1])]).T




def to_euclid_coords_plot_scaled(ind, imc_tsubimg, if_tsubimg, imcinmat, ifinmat, imcinscmat, ifinscmat, imcchannelnames, ifchannelnames, tidn=None, scale=1, return_fig=False, ch2={"IMC": "CD45RO", "IF": "CD45"}):

    # tsubimgexp = cv2.morphologyEx(imc_tsubimg[-5,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    imc_pts_cell = cv2.findContours(imc_tsubimg[-5,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)
    # tsubimgexp2 = cv2.morphologyEx(imc_tsubimg[-2,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    imc_pts_nuc = cv2.findContours(imc_tsubimg[-2,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)

    # tsubimgexp = cv2.morphologyEx(if_tsubimg[-5,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    if_pts_cell = cv2.findContours(if_tsubimg[-5,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)
    # tsubimgexp2 = cv2.morphologyEx(if_tsubimg[-2,:,:].astype(np.uint8), cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    if_pts_nuc = cv2.findContours(if_tsubimg[-2,:,:].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2)


    import matplotlib.patches as patches
    fig, ax = plt.subplots(2, 6,figsize=(18, 8), dpi=100,layout='constrained')
    ch=np.argmax(imcchannelnames==ch2['IMC'])
    pcm=ax[0,0].imshow(imc_tsubimg[ch,:,:])
    ax[0,0].set_title("IMC Image")
    fig.colorbar(pcm, ax=ax[0,0], location='bottom')
    poly = patches.Polygon(imc_pts_cell, edgecolor='red', facecolor='none')
    ax[0,0].add_patch(poly)
    poly = patches.Polygon(imc_pts_nuc, edgecolor='lightblue', facecolor='none')
    ax[0,0].add_patch(poly)
    pcm=ax[0,1].imshow(imcinscmat[ind,ch,:,:])
    ax[0,1].set_title(f"IMC {ch2['IMC']} scaled polar")
    fig.colorbar(pcm, ax=ax[0,1], location='bottom')
    pcm=ax[0,2].imshow(acf2d(imcinscmat[ind,ch,:,:])[:5,:])
    ax[0,2].set_title("ACF")
    fig.colorbar(pcm, ax=ax[0,2], location='bottom')
    pcm=ax[0,3].imshow(cov2d(imcinscmat[ind,ch,:,:])[:5,:])
    ax[0,3].set_title("Cov")
    fig.colorbar(pcm, ax=ax[0,3], location='bottom')
    pcm=ax[0,4].imshow(difl12d(imcinscmat[ind,ch,:,:])[:5,:])
    ax[0,4].set_title("L1 Norm")
    fig.colorbar(pcm, ax=ax[0,4], location='bottom')
    pcm=ax[0,5].imshow(difl22d(imcinscmat[ind,ch,:,:])[:5,:])
    ax[0,5].set_title("L2 Norm")
    fig.colorbar(pcm, ax=ax[0,5], location='bottom')


    ch=np.argmax(ifchannelnames==ch2['IF'])
    pcm=ax[1,0].imshow(if_tsubimg[ch,:,:])
    ax[1,0].set_title("IF Image")
    fig.colorbar(pcm, ax=ax[1,0], location='bottom')
    poly = patches.Polygon(if_pts_cell, edgecolor='red', facecolor='none')
    ax[1,0].add_patch(poly)
    poly = patches.Polygon(if_pts_nuc, edgecolor='lightblue', facecolor='none')
    ax[1,0].add_patch(poly)
    pcm=ax[1,1].imshow(ifinscmat[ind,ch,:,:])
    ax[1,1].set_title(f"IF {ch2['IF']}scaled polar")
    fig.colorbar(pcm, ax=ax[1,1], location='bottom')
    pcm=ax[1,2].imshow(acf2d(ifinscmat[ind,ch,:,:])[:5,:])
    ax[1,2].set_title("ACF")
    fig.colorbar(pcm, ax=ax[1,2], location='bottom')
    pcm=ax[1,3].imshow(cov2d(ifinscmat[ind,ch,:,:])[:5,:])
    ax[1,3].set_title("Cov")
    fig.colorbar(pcm, ax=ax[1,3], location='bottom')
    pcm=ax[1,4].imshow(difl12d(ifinscmat[ind,ch,:,:])[:5,:])
    ax[1,4].set_title("L1 Norm")
    fig.colorbar(pcm, ax=ax[1,4], location='bottom')
    pcm=ax[1,5].imshow(difl22d(ifinscmat[ind,ch,:,:])[:5,:])
    ax[1,5].set_title("L2 Norm")
    fig.colorbar(pcm, ax=ax[1,5], location='bottom')


    if return_fig:
        return fig
    plt.show()

