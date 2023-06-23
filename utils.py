import pandas as pd
import numpy as np
from nilearn import datasets, image
from nilearn.interfaces import fmriprep
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker


def time_series(func_data, brain_mask, conf_strategy, atlas_maps, low_pass, high_pass, smoothing_fwhm, t_r):
    kwargs = {'mask_img': brain_mask, 'smoothing_fwhm': smoothing_fwhm, 'low_pass': low_pass, 'high_pass': high_pass,
              't_r': t_r, 'standardize': False, 'detrend': True, 'memory': 'nilearn_cache', 'memory_level': 2}
    atlas_maps_img = image.load_img(atlas_maps)
    if len(atlas_maps_img.shape) == 4:
        # Probabilistic atlas
        nifti_masker = NiftiMapsMasker(maps_img=atlas_maps,
                                       **kwargs)
    else:
        nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                         **kwargs)
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series


def timeseries_from_regions(regions_extractor, func_data, conf_strategy):
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    return regions_extractor.transform(func_data, confounds=confounds, sample_mask=sample_mask)


def pad_timeseries(timeseries, pad_value=np.nan):
    n_timepoints = timeseries.apply(lambda ts: ts.shape[0]).value_counts().index[0]
    timeseries = timeseries.apply(lambda ts: np.pad(ts, ((0, n_timepoints - ts.shape[0]), (0, 0)),
                                                    'constant', constant_values=pad_value))

    return timeseries


def extract_network_from_atlas(atlas, network_name):
    if atlas.name == 'msdl':
        network_img, network_labels = extract_network_from_msdl_atlas(atlas, network_name)
    else:
        raise ValueError(f'Can not extract networks from {atlas.name} atlas')

    return network_img, network_labels


def extract_network_from_msdl_atlas(atlas, network_name):
    if network_name not in atlas.networks:
        raise ValueError(f'Network {network_name} not in {atlas.name} atlas')

    network_indices = [idx for idx, network in enumerate(atlas.networks) if network == network_name]
    network_labels = [atlas.labels[idx] for idx in network_indices]

    indices_not_in_network = [idx for idx, label in enumerate(atlas.labels) if label not in network_labels]
    atlas_img = image.load_img(atlas.maps)
    atlas_affine, network_data = atlas_img.affine, atlas_img.get_fdata()
    network_data[:, :, :, indices_not_in_network] = 0
    network_img = image.new_img_like(atlas_img, network_data, affine=atlas_affine, copy_header=True)

    return network_img, network_labels


def load_atlas(atlas_name):
    if atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
        atlas.labels = pd.DataFrame({'name': atlas.labels})
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
        # Remove missing regions in atlas.maps from atlas.labels
        # (0 == 'background', 42 == 'L Medial_wall', 117 == 'R Medial_wall)
        atlas.labels = atlas.labels.drop([0, 42, 117]).reset_index(drop=True)
    elif atlas_name == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
        # Remove '7Networks_' prefix
        atlas.labels = pd.DataFrame({'name': [label[10:].decode() for label in atlas.labels]})
    elif atlas_name == 'msdl':
        atlas = datasets.fetch_atlas_msdl()
        atlas.labels = pd.DataFrame({'name': atlas.labels})
    else:
        raise ValueError(f'Atlas {atlas_name} not recognized')
    atlas.name = atlas_name

    return atlas


def load_clinical_data(clinical_datafile):
    cg = pd.read_csv(clinical_datafile)
    subjects_data = cg[~cg['AnonID'].isna()]
    subjects_data = subjects_data.astype({'AnonID': int})
    subjects_data = subjects_data.set_index('AnonID')

    # Remove invalid subjects (subj 29 has different data shapes)
    subjects_to_remove = [2, 17, 29]
    subjects_data = subjects_data.drop(subjects_to_remove)
    # Remove subjects with no cluster
    subjects_data = subjects_data[~subjects_data['cluster'].isna()]

    return subjects_data


def load_datapaths(subjects_paths, subjects_df):
    for subj_path in subjects_paths:
        subj_id = int(subj_path.name.split('-')[1])
        if subj_id in subjects_df.index:
            # Get the path to the preprocessed functional data
            func_path = subj_path / 'func'
            func_file = [f for f in func_path.glob('*.nii.gz') if 'preproc' in f.name][0]
            mask_file = [f for f in func_path.glob('*.nii.gz') if 'brain_mask' in f.name][0]
            subjects_df.loc[subj_id, 'func_path'] = str(func_file)
            subjects_df.loc[subj_id, 'mask_path'] = str(mask_file)


def apply_threshold(connectivity_matrix, threshold):
    percentile = np.percentile(connectivity_matrix, threshold)
    connectivity_matrix[connectivity_matrix < percentile] = 0
    return connectivity_matrix


def q_test(data, mean):
    # Upper triangulate the data
    data, mean = np.triu(data, k=1), np.triu(mean, k=1)
    q = np.sum(np.sum(np.square(data - mean)) / (len(data) - 1))
    return q
