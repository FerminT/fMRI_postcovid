import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils

# NiLearn methods and classes
from nilearn import plotting
from nilearn.interfaces import fmriprep
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker


def build_connectome(subjects_df, conf_strategy, atlas_name,
                     threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                     output):
    atlas = utils.load_atlas(atlas_name)
    subjects_df['time_series'] = subjects_df.apply(lambda subj: time_series(subj['func_path'], subj['mask_path'],
                                                                            conf_strategy, atlas.maps,
                                                                            low_pass, high_pass, smoothing_fwhm,
                                                                            t_r), axis=1)

    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])

    # Compute mean connectivity matrix by cluster
    clusters_connectivity_matrix = {cluster: None for cluster in subjects_df['cluster'].unique()}
    for cluster in clusters_connectivity_matrix:
        cluster_df = subjects_df[subjects_df['cluster'] == cluster]
        clusters_connectivity_matrix[cluster] = mean_connectivity_matrix(cluster_df['time_series'].values)

    conn_output = output / 'connectivity_matrices'
    conn_output.mkdir(exist_ok=True)
    save_connectivity_matrices(subjects_df, atlas.labels, threshold, conn_output)
    save_clusters_matrices(clusters_connectivity_matrix, atlas.labels, threshold, conn_output)
    if atlas_name == 'schaefer':
        connmatrices_over_networks(clusters_connectivity_matrix, atlas.labels, threshold, output)


def connmatrices_over_networks(clusters_data, atlas_labels, threshold, output):
    # Only for Schaefer atlas
    diff_connmatrix = np.empty((len(atlas_labels), len(atlas_labels)))
    for i, cluster in enumerate(clusters_data):
        cluster_connmatrix = clusters_data[cluster]['connectivity_matrix']
        cluster_connmatrix = utils.apply_threshold(cluster_connmatrix, threshold)
        if i == 0:
            diff_connmatrix = cluster_connmatrix
        else:
            diff_connmatrix = np.abs(diff_connmatrix - cluster_connmatrix)
    networks, network_index = {}, 0
    all_atlas_labels = atlas_labels['name'].values
    for region in all_atlas_labels:
        network = region.split('_')[1]
        if network not in networks:
            networks[network] = {'index': network_index}
            network_index += 1
    networks_connmatrix = np.zeros((len(networks), len(networks)))
    terms_matrix = np.zeros((len(networks), len(networks)))
    for i, row_region in enumerate(all_atlas_labels):
        row_network = row_region.split('_')[1]
        idx_row_network = networks[row_network]['index']
        for j, col_region in enumerate(all_atlas_labels):
            col_network = col_region.split('_')[1]
            idx_col_network = networks[col_network]['index']
            networks_connmatrix[idx_row_network, idx_col_network] += diff_connmatrix[i, j]
            terms_matrix[idx_row_network, idx_col_network] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.matshow(terms_matrix)
    plt.colorbar()
    plt.savefig(output / f'terms_matrix_{len(all_atlas_labels)}rois.png')
    plt.close(fig)
    # Compute mean
    networks_connmatrix /= terms_matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    networks_labels = pd.DataFrame({'name': list(networks.keys())})
    plot_matrix_on_axis(networks_connmatrix, networks_labels, ax, threshold=0,
                        tri='full', vmin=-0.1, vmax=0.1)
    fig.savefig(output / f'networks_diff_{len(all_atlas_labels)}rois.png')
    plt.close(fig)


def save_clusters_matrices(clusters_data, atlas_labels, threshold, output):
    fig, axes = plt.subplots(nrows=1, ncols=len(clusters_data), figsize=(30, 30))
    for i, cluster in enumerate(clusters_data):
        cluster_connmatrix = clusters_data[cluster]['connectivity_matrix']
        # Apply first cluster label order to the rest of the clusters for better comparison
        if i > 0:
            label_order = [tick.get_text() for tick in axes[0].get_xticklabels()]
            regions_indices = [atlas_labels[atlas_labels.name == region].index[0] for region in label_order]
            cluster_connmatrix = cluster_connmatrix[regions_indices, :][:, regions_indices]
            atlas_labels['name'] = pd.Categorical(atlas_labels['name'], categories=label_order, ordered=True)
            atlas_labels = atlas_labels.sort_values('name')
        reorder = i == 0
        plot_matrix_on_axis(cluster_connmatrix, atlas_labels, axes[i], threshold, reorder=reorder)
        axes[i].set_title(f'Cluster {cluster}')
    fig.savefig(output / 'clusters.png')


def save_connectivity_matrices(subjects_df, atlas_labels, threshold, output, reorder=False):
    # Plot connectivity matrices and save them
    for subj in subjects_df['AnonID'].values:
        subj_df = subjects_df[subjects_df['AnonID'] == subj]
        connectivity_matrix = subj_df['connectivity_matrix'].values[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_matrix_on_axis(connectivity_matrix, atlas_labels, ax, threshold, reorder=reorder)
        fig.savefig(output / f'subj_{subj}.png')
        plt.close(fig)


def plot_matrix_on_axis(connectivity_matrix, atlas_labels, ax, threshold,
                        reorder=False, tri='lower', vmin=-0.8, vmax=0.8):
    matrix_to_plot = connectivity_matrix.copy()
    matrix_to_plot = utils.apply_threshold(matrix_to_plot, threshold)
    # Get labels in the correct format until plot_matrix is fixed
    labels = list(atlas_labels.name.values)
    plotting.plot_matrix(matrix_to_plot,
                         tri=tri,
                         labels=labels,
                         colorbar=True,
                         vmin=vmin,
                         vmax=vmax,
                         reorder=reorder,
                         axes=ax)


def mean_connectivity_matrix(time_series, kind='correlation'):
    connectivity_matrices, connectivity_measure = connectivity_matrix(time_series, kind)
    mean_connectivity_matrix = connectivity_measure.mean_

    q = utils.q_test(connectivity_matrices, mean_connectivity_matrix)
    print(f'Q test: {q}')

    return mean_connectivity_matrix


def connectivity_matrix(time_series, kind='correlation'):
    connectivity_measure = ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform(time_series)

    return connectivity_matrix, connectivity_measure


def time_series(func_data, brain_mask, conf_strategy, atlas_maps, low_pass, high_pass, smoothing_fwhm, t_r):
    nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                     mask_img=brain_mask,
                                     smoothing_fwhm=smoothing_fwhm,
                                     low_pass=low_pass,
                                     high_pass=high_pass,
                                     t_r=t_r,
                                     standardize=False,
                                     detrend=True,
                                     memory='nilearn_cache', memory_level=2)
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series
