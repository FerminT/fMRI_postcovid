import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils

# NiLearn methods and classes
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure


def build_connectome(subjects_df, conf_strategy, atlas,
                     threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                     output):
    subjects_df['time_series'] = subjects_df.apply(lambda subj: utils.time_series(subj['func_path'], subj['mask_path'],
                                                                                  conf_strategy, atlas.maps,
                                                                                  low_pass, high_pass, smoothing_fwhm,
                                                                                  t_r), axis=1)

    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])

    if atlas.name == 'schaefer':
        connmatrices_over_networks(subjects_df, atlas.labels, threshold, output)

    # Compute mean connectivity matrix by cluster
    clusters_connectivity_matrix = {cluster: None for cluster in subjects_df['cluster'].unique()}
    for cluster in clusters_connectivity_matrix:
        print(f'\nProcessing cluster {cluster} on {atlas.name}...')
        cluster_df = subjects_df[subjects_df['cluster'] == cluster]
        clusters_connectivity_matrix[cluster] = mean_connectivity_matrix(cluster_df['time_series'].values)
        utils.print_connectivity_metrics(clusters_connectivity_matrix[cluster], threshold)

    conn_output = output / 'connectivity_matrices'
    conn_output.mkdir(exist_ok=True, parents=True)
    save_connectivity_matrices(subjects_df, atlas.labels, threshold, conn_output)
    save_clusters_matrices(clusters_connectivity_matrix, atlas.labels, threshold, conn_output)
    save_clusters_connectomes(clusters_connectivity_matrix, atlas, threshold, conn_output)


def networks_connectivity_matrix(subj_connectivity_matrix, networks, all_atlas_labels):
    networks_connmatrix = np.zeros((len(networks), len(networks)))
    terms_matrix = np.zeros((len(networks), len(networks)))
    for i, row_region in enumerate(all_atlas_labels):
        row_network = row_region.split('_')[1]
        idx_row_network = networks[row_network]['index']
        for j, col_region in enumerate(all_atlas_labels):
            col_network = col_region.split('_')[1]
            idx_col_network = networks[col_network]['index']
            networks_connmatrix[idx_row_network, idx_col_network] += subj_connectivity_matrix[i, j]
            terms_matrix[idx_row_network, idx_col_network] += 1
    networks_connmatrix /= terms_matrix

    return networks_connmatrix


def connmatrices_over_networks(subjects_df, atlas_labels, threshold, output):
    # Only for Schaefer atlas
    networks = utils.get_schaefer_networks_indices(atlas_labels)
    all_atlas_labels = atlas_labels['name'].values
    # subjects_df['connectivity_matrix'] = subjects_df['connectivity_matrix'].apply(lambda conn_matrix:
    #                                                                               utils.apply_threshold(conn_matrix,
    #                                                                                                     threshold=threshold))
    subjects_df['networks_connmatrix'] = subjects_df['connectivity_matrix'].apply(lambda conn_matrix:
                                                                                  networks_connectivity_matrix(
                                                                                      conn_matrix, networks,
                                                                                      all_atlas_labels))
    networks_std = subjects_df['networks_connmatrix'].values.std()
    diff_connmatrix = np.empty((len(atlas_labels), len(atlas_labels)))
    for i, cluster in enumerate(subjects_df['cluster'].unique()):
        cluster_connmatrices = subjects_df[subjects_df['cluster'] == cluster]['networks_connmatrix'].values
        if i == 0:
            diff_connmatrix = cluster_connmatrices.mean()
        else:
            diff_connmatrix = (diff_connmatrix - cluster_connmatrices.mean()) / networks_std
    fig, ax = plt.subplots(figsize=(10, 8))
    networks_labels = pd.DataFrame({'name': list(networks.keys())})
    plot_matrix_on_axis(diff_connmatrix, networks_labels, ax, threshold=0,
                        tri='full')
    fig.savefig(output / f'networks_diff_{len(all_atlas_labels)}rois.png')
    plt.close(fig)


def save_clusters_connectomes(clusters_connectivity_matrix, atlas, threshold, conn_output):
    coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas.maps)
    for cluster in clusters_connectivity_matrix:
        correlation_matrix = clusters_connectivity_matrix[cluster]
        plotting.plot_connectome(correlation_matrix, coordinates, edge_threshold=threshold / 100,
                                 title=f'{atlas.name} cluster {cluster}', edge_cmap='YlOrBr', edge_vmin=0, edge_vmax=0.8)
        plt.savefig(conn_output / f'{atlas.name}_cluster_{cluster}_connectome.png')


def save_clusters_matrices(clusters_connectivity_matrices, atlas_labels, threshold, output):
    fig, axes = plt.subplots(nrows=1, ncols=len(clusters_connectivity_matrices), figsize=(30, 30))
    for i, cluster in enumerate(clusters_connectivity_matrices):
        cluster_connmatrix = clusters_connectivity_matrices[cluster]
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
    subjects_df.apply(lambda subj: save_connectivity_matrix(subj['connectivity_matrix'], subj.name,
                                                            atlas_labels, threshold, output, reorder=reorder), axis=1)


def save_connectivity_matrix(conn_matrix, subj_id, atlas_labels, threshold, output, reorder=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_matrix_on_axis(conn_matrix, atlas_labels, ax, threshold, reorder=reorder)
    fig.savefig(output / f'subj_{subj_id}.png')
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
