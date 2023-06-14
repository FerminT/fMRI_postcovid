import matplotlib.pyplot as plt
from nilearn import plotting, image, masking
from nilearn.interfaces import fmriprep
from nilearn.maskers import MultiNiftiMasker
from nilearn.decomposition import dict_learning


def extract_components(func_data, brain_masks, conf_strategy, n_components):
    brain_mask = masking.intersect_masks(brain_masks)
    masker = MultiNiftiMasker(mask_img=brain_mask,
                              high_pass=0.01,
                              low_pass=0.08,
                              t_r=2.,
                              smoothing_fwhm=6.,
                              mask_strategy='epi',
                              standardize=True,
                              detrend=True,
                              memory='nilearn_cache', memory_level=2)
    dict_learn = dict_learning.DictLearning(n_components=n_components,
                                            high_pass=0.01,
                                            low_pass=0.08,
                                            t_r=2.,
                                            smoothing_fwhm=6.,
                                            standardize=True,
                                            detrend=True,
                                            mask=masker,
                                            random_state=42,
                                            n_jobs=-1,
                                            memory='nilearn_cache', memory_level=2)
    confounds, _ = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    dict_learn.fit(func_data, confounds=confounds)

    return dict_learn.components_img_


def save_principal_components(clusters_data, output):
    cortices_coords = {'Motor cortex': [45, -35, 50], 'Auditory cortex': [50, -15, 12], 'Visual cortex': [0, -75, 4]}
    for cluster in clusters_data:
        components_img = clusters_data[cluster]['components_img']
        plotting.plot_prob_atlas(components_img,
                                 draw_cross=False,
                                 linewidths=None,
                                 cut_coords=[0, 0, 0],
                                 title=f'Cluster {cluster}')
        plt.savefig(output / f'maps_cluster_{cluster}.png')
        first_comp = components_img.slicer[..., :4]
        fig = plt.figure(figsize=(16, 3))
        for i, cur_img in enumerate(image.iter_img(first_comp)):
            ax = fig.add_subplot(1, 4, i + 1)
            plotting.plot_stat_map(cur_img, display_mode="z", title="PC %d" % i,
                                   cut_coords=1, colorbar=True, axes=ax)
        fig.savefig(output / f'components_cluster_{cluster}.png')
        plt.close(fig)

        for cortex in cortices_coords:
            fig = plt.figure(figsize=(16, 3))
            cut_coords = cortices_coords[cortex]
            plotting.plot_stat_map(components_img.slicer[..., 19], display_mode='ortho',
                                   cut_coords=cut_coords, colorbar=True, draw_cross=False,
                                   title=cortex)
            plt.savefig(output / f'{cortex}_cluster_{cluster}.png')
            plt.close(fig)