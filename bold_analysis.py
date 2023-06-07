import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# NiLearn methods and classes
from nilearn import datasets, plotting, image, masking
from nilearn.interfaces import fmriprep
from nilearn.connectome import ConnectivityMeasure
from nilearn.decomposition import dict_learning
from nilearn.maskers import NiftiLabelsMasker, MultiNiftiMasker

TEMPLATE_SHAPE = [55, 65, 55]


def save_clusters_matrices(clusters_data, atlas, threshold, output):
    fig, axes = plt.subplots(nrows=1, ncols=len(clusters_data), figsize=(20, 20))
    for i, cluster in enumerate(clusters_data):
        cluster_conmatrix = clusters_data[cluster]['connectivity_matrix']
        # Apply first cluster label order to the rest of the clusters for better comparison
        if i > 0:
            label_order = [tick.get_text() for tick in axes[0].get_xticklabels()]
            regions_indices = [atlas.labels.index(region) for region in label_order]
            cluster_conmatrix = cluster_conmatrix[regions_indices, :][:, regions_indices]
        else:
            label_order = atlas.labels
        reorder = i == 0
        plot_matrix_on_axis(cluster_conmatrix, label_order, axes[i], threshold, reorder=reorder)
        axes[i].set_title(f'Cluster {cluster}')
    fig.savefig(output / 'clusters.png')


def save_principal_components(clusters_data, output):
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


def build_connectome(subjects_paths, conf_strategy, atlas_name, n_components, clinical_datafile, output, threshold=95):
    subjects_df = load_clinical_data(clinical_datafile)
    load_datapaths(subjects_paths, subjects_df)
    atlas = load_atlas(atlas_name)
    subjects_df['time_series'] = subjects_df.apply(lambda subj: time_series(subj['func_path'], subj['mask_path'],
                                                                            conf_strategy, atlas.maps), axis=1)

    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])

    # Compute mean connectivity matrix by cluster
    clusters_data = {cluster: {'connectivity_matrix': None, 'components_img': None}
                     for cluster in subjects_df['cluster'].unique()}
    for cluster in clusters_data:
        cluster_df = subjects_df[subjects_df['cluster'] == cluster]
        clusters_data[cluster]['connectivity_matrix'] = mean_connectivity_matrix(cluster_df['time_series'].values)
        clusters_data[cluster]['components_img'] = extract_components(cluster_df['func_path'].values,
                                                                      cluster_df['mask_path'].values,
                                                                      conf_strategy,
                                                                      n_components)

    output = output / 'connectivity_matrices'
    output.mkdir(exist_ok=True)
    save_connectivity_matrices(subjects_df, atlas, threshold, output)
    save_clusters_matrices(clusters_data, atlas, threshold, output)
    output = output.parent / 'components'
    output.mkdir(exist_ok=True)
    save_principal_components(clusters_data, output)


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


def save_connectivity_matrices(subjects_df, atlas, threshold, output, reorder=False):
    # Plot connectivity matrices and save them
    for subj in subjects_df['AnonID'].values:
        subj_df = subjects_df[subjects_df['AnonID'] == subj]
        connectivity_matrix = subj_df['connectivity_matrix'].values[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_matrix_on_axis(connectivity_matrix, atlas.labels, ax, threshold, reorder=reorder)
        fig.savefig(output / f'subj_{subj}.png')
        plt.close(fig)


def plot_matrix_on_axis(connectivity_matrix, labels, ax, threshold, reorder=False):
    # Compute percentile and apply threshold
    percentile = np.percentile(connectivity_matrix, threshold)
    connectivity_matrix[connectivity_matrix < percentile] = 0
    plotting.plot_matrix(connectivity_matrix,
                         tri='lower',
                         labels=labels,
                         colorbar=True,
                         vmax=0.8,
                         vmin=-0.8,
                         reorder=reorder,
                         axes=ax)


def mean_connectivity_matrix(time_series, kind='correlation'):
    connectivity_matrices, connectivity_measure = connectivity_matrix(time_series, kind)
    mean_connectivity_matrix = connectivity_measure.mean_

    q = q_test(connectivity_matrices, mean_connectivity_matrix)
    print(f'Q test: {q}')

    return mean_connectivity_matrix


def connectivity_matrix(time_series, kind='correlation'):
    connectivity_measure = ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform(time_series)

    return connectivity_matrix, connectivity_measure


def time_series(func_data, brain_mask, conf_strategy, atlas_maps):
    nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                     mask_img=brain_mask,
                                     smoothing_fwhm=6,
                                     low_pass=0.08,
                                     high_pass=0.01,
                                     t_r=2.,
                                     standardize=False,
                                     detrend=True,
                                     memory='nilearn_cache', memory_level=2)
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series


def q_test(data, mean):
    # Upper triangulate the data
    data, mean = np.triu(data, k=1), np.triu(mean, k=1)
    q = np.sum(np.sum(np.square(data - mean)) / (len(data) - 1))
    return q


def load_atlas(atlas_name):
    # Use nilearn datasets to fetch atlas
    if atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009()
    elif atlas_name == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018()
    else:
        raise ValueError(f'Atlas {atlas_name} not recognized')

    return atlas


def load_clinical_data(clinical_datafile):
    cg = pd.read_csv(clinical_datafile)
    subjects_data = cg[~cg['AnonID'].isna()]
    subjects_data = subjects_data.astype({'AnonID': int})

    # Remove invalid subjects (subj 29 has different data shapes)
    subjects_to_remove = [2, 17, 29]
    subjects_data = subjects_data.drop(subjects_data[subjects_data['AnonID'].isin(subjects_to_remove)].index)
    # Remove subjects with no cluster
    subjects_data = subjects_data[~subjects_data['cluster'].isna()]

    return subjects_data


def load_datapaths(subjects_paths, subjects_df):
    for subj_path in subjects_paths:
        subj_id = int(subj_path.name.split('-')[1])
        if subj_id in subjects_df['AnonID'].values:
            # Get the path to the preprocessed functional data
            func_path = subj_path / 'func'
            func_file = [f for f in func_path.glob('*.nii.gz') if 'preproc' in f.name][0]
            mask_file = [f for f in func_path.glob('*.nii.gz') if 'brain_mask' in f.name][0]
            subjects_df.loc[subjects_df['AnonID'] == subj_id, 'func_path'] = str(func_file)
            subjects_df.loc[subjects_df['AnonID'] == subj_id, 'mask_path'] = str(mask_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--subject', type=str, default='all')
    arg_parser.add_argument('-c', '--confounds', type=str, default='simple',
                            help='Strategy for loading fMRIPrep denoising strategies. \
                           Options: simple, compcor, srubbing, ica_aroma')
    arg_parser.add_argument('-a', '--atlas', type=str, default='aal', help='Atlas to use for brain parcellation')
    arg_parser.add_argument('-d', '--derivatives', type=str, default='neurocovid_derivatives',
                            help='Path to BIDS derivatives folder')
    arg_parser.add_argument('-o', '--output', type=str, default='analysis/functional_connectivity')
    arg_parser.add_argument('-n', '--n_components', type=int, default=5,
                            help='Number of components to use for DictLearning')
    arg_parser.add_argument('-t', '--threshold', type=int, default=95,
                            help='Activity threshold for connectome (percentile)')
    arg_parser.add_argument('-clinical', '--clinical', type=str, default='clinical_data.csv',
                            help='Path to file with subjects clinical data')

    args = arg_parser.parse_args()

    # Set up paths
    derivatives, output = Path(args.derivatives), Path(args.output) / args.atlas
    output.mkdir(parents=True, exist_ok=True)

    if args.subject == 'all':
        subjects = [sub for sub in derivatives.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [derivatives / f'sub-{args.subject.zfill(3)}']

    build_connectome(subjects, args.confounds, args.atlas, args.n_components, args.clinical, output, args.threshold)
