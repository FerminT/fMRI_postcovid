import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import utils
from .atlas_manager import get_schaefer_networks_indices

# NiLearn methods and classes
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure


def build_connectome(subjects_df, conf_strategy, atlas,
                     threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                     output):
    conn_output = output / 'connectivity_matrices'
    conn_output.mkdir(exist_ok=True, parents=True)

    subjects_df['time_series'] = subjects_df.apply(lambda subj: utils.time_series(subj['func_path'], subj['mask_path'],
                                                                                  conf_strategy, atlas.maps,
                                                                                  low_pass, high_pass, smoothing_fwhm,
                                                                                  t_r), axis=1)

    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])

    if atlas.name == 'schaefer':
        networks_diff, networks_labels = connmatrices_over_networks(subjects_df, atlas.labels)
        save_connectivity_matrix(networks_diff, f'networks_diff_{len(atlas.labels)}rois', networks_labels,
                                 tri='full', output=conn_output)

    subjects_df['connectivity_matrix'] = subjects_df['connectivity_matrix'].apply(lambda matrix:
                                                                                  utils.apply_threshold(matrix,
                                                                                                        threshold))

    groups_connectivity_matrix = {group: None for group in subjects_df['group'].unique()}
    for group in groups_connectivity_matrix:
        print(f'\nProcessing group {group} on {atlas.name}...')
        group_df = subjects_df[subjects_df['group'] == group]
        groups_connectivity_matrix[group] = mean_connectivity_matrix(group_df['connectivity_matrix'].values)
        utils.print_connectivity_metrics(groups_connectivity_matrix[group])

    save_connectivity_matrices(subjects_df, atlas.labels, conn_output)
    save_groups_matrices(groups_connectivity_matrix, atlas.labels, conn_output)
    save_groups_connectomes(groups_connectivity_matrix, atlas, conn_output)


def connmatrices_over_networks(subjects_df, atlas_labels):
    # Only for Schaefer atlas: compute the difference between groups connectivity matrices over networks
    networks = get_schaefer_networks_indices(atlas_labels)
    all_atlas_labels = atlas_labels['name'].values
    subjects_df['networks_connmatrix'] = subjects_df['connectivity_matrix'].apply(lambda conn_matrix:
                                                                                  networks_connectivity_matrix(
                                                                                    conn_matrix, networks,
                                                                                    all_atlas_labels))
    networks_std = subjects_df['networks_connmatrix'].values.std()
    diff_connmatrix = np.empty((len(atlas_labels), len(atlas_labels)))
    for i, group in enumerate(sorted(subjects_df['group'].unique())):
        group_connmatrices = subjects_df[subjects_df['group'] == group]['networks_connmatrix'].values
        if i == 0:
            diff_connmatrix = group_connmatrices.mean()
        else:
            diff_connmatrix = (diff_connmatrix - group_connmatrices.mean()) / networks_std
    networks_labels = pd.DataFrame({'name': list(networks.keys())})

    return diff_connmatrix, networks_labels


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


def mean_connectivity_matrix(connectivity_matrices):
    mean_connectivity_matrix = np.mean(connectivity_matrices.tolist(), axis=0)
    q, df = utils.q_test(connectivity_matrices.tolist(), mean_connectivity_matrix)
    print(f'Q test: {q}; degrees of freedom: {df}')

    return mean_connectivity_matrix


def connectivity_matrix(time_series, kind='correlation'):
    connectivity_measure = ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform(time_series)

    return connectivity_matrix, connectivity_measure


def save_groups_connectomes(groups_connectivity_matrix, atlas, conn_output):
    if utils.is_probabilistic_atlas(atlas.maps):
        coordinates = plotting.find_probabilistic_atlas_cut_coords(maps_img=atlas.maps)
    else:
        coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas.maps)
    for group in groups_connectivity_matrix:
        correlation_matrix = groups_connectivity_matrix[group]
        plotting.plot_connectome(correlation_matrix, coordinates, title=f'{atlas.name}, {group}',
                                 edge_cmap='coolwarm', edge_vmin=-0.8, edge_vmax=0.8)
        plt.savefig(conn_output / f'{atlas.name}_{group}_connectome.png')


def save_groups_matrices(groups_connectivity_matrices, atlas_labels, output):
    fig, axes = plt.subplots(nrows=1, ncols=len(groups_connectivity_matrices), figsize=(30, 30))
    for i, group in enumerate(groups_connectivity_matrices):
        group_connmatrix = groups_connectivity_matrices[group]
        # Apply first group label order to the rest of the groups for better comparison
        if i > 0:
            label_order = [tick.get_text() for tick in axes[0].get_xticklabels()]
            regions_indices = [atlas_labels[atlas_labels.name == region].index[0] for region in label_order]
            group_connmatrix = group_connmatrix[regions_indices, :][:, regions_indices]
            atlas_labels['name'] = pd.Categorical(atlas_labels['name'], categories=label_order, ordered=True)
            atlas_labels = atlas_labels.sort_values('name')
        reorder = i == 0
        utils.plot_matrix_on_axis(group_connmatrix, atlas_labels, axes[i], reorder=reorder)
        axes[i].set_title(f'{group}')
    fig.savefig(output / 'groups_comparison.png')


def save_connectivity_matrices(subjects_df, atlas_labels, output, reorder=False):
    subjects_df.apply(lambda subj: save_connectivity_matrix(subj['connectivity_matrix'], f'subj_{subj.name}',
                                                            atlas_labels, output, reorder=reorder), axis=1)


def save_connectivity_matrix(conn_matrix, fig_name, atlas_labels, output,
                             tri='lower', vmin=-0.8, vmax=0.8, reorder=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    utils.plot_matrix_on_axis(conn_matrix, atlas_labels, ax, tri=tri, vmin=vmin, vmax=vmax, reorder=reorder)
    fig.savefig(output / f'{fig_name}.png')
    plt.close(fig)
