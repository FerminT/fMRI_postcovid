import matplotlib.pyplot as plt
from nilearn import plotting, image, masking
from nilearn.interfaces import fmriprep
from nilearn.maskers import MultiNiftiMasker
from nilearn.decomposition import dict_learning
from nilearn.regions import RegionExtractor


def extract_group_components(subjects_df, conf_strategy, n_components,
                             low_pass, high_pass, smoothing_fwhm, t_r,
                             output):
    groups_components = {group: None for group in subjects_df['group'].unique()}
    for group in groups_components:
        group_df = subjects_df[subjects_df['group'] == group]
        groups_components[group] = extract_components(group_df['func_path'].values,
                                                      group_df['mask_path'].values,
                                                      conf_strategy,
                                                      n_components,
                                                      low_pass, high_pass, smoothing_fwhm, t_r)

    save_principal_components(groups_components, output)


def extract_components(func_data, brain_masks, conf_strategy, n_components, low_pass, high_pass, smoothing_fwhm, t_r):
    brain_mask = masking.intersect_masks(brain_masks)
    masker = MultiNiftiMasker(mask_img=brain_mask,
                              smoothing_fwhm=smoothing_fwhm,
                              low_pass=low_pass,
                              high_pass=high_pass,
                              t_r=t_r,
                              mask_strategy='epi',
                              standardize=True,
                              detrend=True,
                              memory='cache', memory_level=2)
    dict_learn = dict_learning.DictLearning(n_components=n_components,
                                            smoothing_fwhm=smoothing_fwhm,
                                            low_pass=low_pass,
                                            high_pass=high_pass,
                                            t_r=t_r,
                                            standardize=True,
                                            detrend=True,
                                            mask=masker,
                                            random_state=42,
                                            n_jobs=-1,
                                            memory='cache', memory_level=2)
    confounds, _ = fmriprep.load_confounds(func_data, conf_strategy)
    dict_learn.fit(func_data, confounds=confounds)

    return dict_learn.components_img_


def extract_regions(components_img, threshold=0.5, thresholding_strategy='ratio_n_voxels', extractor='local_regions',
                    min_region_size=1350, standardize=True):
    regions_extractor = RegionExtractor(components_img,
                                        threshold=threshold,
                                        extractor=extractor,
                                        thresholding_strategy=thresholding_strategy,
                                        min_region_size=min_region_size,
                                        standardize=standardize)
    regions_extractor.fit()
    print(f'Extracted {regions_extractor.regions_img_.shape[-1]} regions')

    return regions_extractor


def save_principal_components(groups_components, output):
    output.mkdir(exist_ok=True)
    cortices_coords = {'Motor cortex': [45, -35, 50], 'Auditory cortex': [50, -15, 12], 'Visual cortex': [0, -75, 4]}
    for group in groups_components:
        components_img = groups_components[group]
        plotting.plot_prob_atlas(components_img,
                                 draw_cross=False,
                                 linewidths=None,
                                 cut_coords=[0, 0, 0],
                                 title=f'{group}')
        plt.savefig(output / f'maps_{group}.png')
        first_comp = components_img.slicer[..., :4]
        fig = plt.figure(figsize=(16, 3))
        for i, cur_img in enumerate(image.iter_img(first_comp)):
            ax = fig.add_subplot(1, 4, i + 1)
            plotting.plot_stat_map(cur_img, display_mode="z", title="PC %d" % i,
                                   cut_coords=1, colorbar=True, axes=ax)
        fig.savefig(output / f'components_{group}.png')
        plt.close(fig)

        for cortex in cortices_coords:
            fig = plt.figure(figsize=(16, 3))
            cut_coords = cortices_coords[cortex]
            plotting.plot_stat_map(components_img.slicer[..., 19], display_mode='ortho',
                                   cut_coords=cut_coords, colorbar=True, draw_cross=False,
                                   title=cortex)
            plt.savefig(output / f'{cortex}_{group}.png')
            plt.close(fig)
