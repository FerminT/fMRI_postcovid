import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import bct
from . import utils
from .atlas_manager import get_schaefer_networks_indices

# NiLearn methods and classes
from nilearn import plotting
from nilearn import connectome


def build_connectome(subjects_df, conf_strategy, atlas,
                     threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                     output):
    subjects_df = build_timeseries(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r)
    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])
    output.mkdir(exist_ok=True, parents=True)
    if 'schaefer' in atlas.name and not utils.is_network(atlas.name):
        groups_diff_over_networks(subjects_df, atlas.labels, output)

    subjects_df['connectivity_matrix'] = subjects_df['connectivity_matrix'].apply(lambda matrix:
                                                                                  apply_threshold(matrix,
                                                                                                  threshold))
    save_connectivity_matrices(subjects_df, atlas.labels, output)
    groups_connectome_analysis(subjects_df, atlas, threshold, output)


def build_timeseries(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r):
    subjects_df['time_series'] = subjects_df.apply(lambda subj: utils.time_series(subj['func_path'], subj['mask_path'],
                                                                                  conf_strategy, atlas.maps,
                                                                                  low_pass, high_pass, smoothing_fwhm,
                                                                                  t_r), axis=1)
    subjects_df.drop(columns=['func_path', 'mask_path'], inplace=True)

    return subjects_df


def groups_connectome_analysis(subjects_df, atlas, threshold, conn_output):
    groups_connectomes = {group: None for group in subjects_df['group'].unique()}
    for group in groups_connectomes:
        group_df = subjects_df[subjects_df['group'] == group]
        group_connectivity_matrices = group_df['connectivity_matrix'].values
        group_connectome = mean_connectivity_matrix(group_connectivity_matrices)
        global_connectivity_metrics(group, group_connectivity_matrices.tolist(),
                                    threshold, conn_output / f'global_metrics.csv')
        save_connectome(group_connectome, atlas,
                        f'{atlas.name}, {group}', f'{atlas.name}_{group}_connectome.png', conn_output)
        groups_connectomes[group] = group_connectome

    save_groups_matrices(groups_connectomes, atlas.labels, conn_output)


def groups_diff_over_networks(subjects_df, atlas_labels, conn_output):
    networks_diff, networks_labels = connmatrices_over_networks(subjects_df, atlas_labels)
    utils.networks_corrcoef_boxplot(subjects_df, 'networks_connmatrix', networks_labels,
                                    group_by='group', output=conn_output)
    utils.save_connectivity_matrix(networks_diff, f'networks_diff', networks_labels,
                                   tri='full', output=conn_output)


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
    connectivity_measure = connectome.ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform(time_series)

    return connectivity_matrix, connectivity_measure


def save_connectome(connectivity_matrix, atlas, fig_title, fig_name, conn_output):
    if utils.is_probabilistic_atlas(atlas.maps):
        coordinates = plotting.find_probabilistic_atlas_cut_coords(maps_img=atlas.maps)
    else:
        coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas.maps)
    plotting.plot_connectome(connectivity_matrix, coordinates, title=fig_title,
                             edge_cmap='coolwarm', edge_vmin=-0.8, edge_vmax=0.8)
    plt.savefig(conn_output / fig_name)


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
    subjects_df.apply(lambda subj: utils.save_connectivity_matrix(subj['connectivity_matrix'], f'subj_{subj.name}',
                                                                  atlas_labels, output, reorder=reorder), axis=1)


def global_connectivity_metrics(group, connectivity_matrices, threshold, filename):
    group_metrics = {'avg_clustering': [], 'avg_neighbor_degree': [], 'num_nodes': [], 'num_edges': []}
    for connectivity_matrix in connectivity_matrices:
        np.fill_diagonal(connectivity_matrix, 0)
        connectome = nx.from_numpy_array(connectivity_matrix)
        group_metrics['avg_clustering'].append(nx.average_clustering(connectome, weight='weight'))
        avg_neighbor_degree = nx.average_neighbor_degree(connectome, weight='weight')
        group_metrics['avg_neighbor_degree'].append(np.mean(list(avg_neighbor_degree.values())))
        group_metrics['num_nodes'].append(len(connectome.nodes))
        group_metrics['num_edges'].append(len(connectome.edges))

    mean_metrics = {'group': group, 'threshold': np.round(threshold, 4)}
    for metric in group_metrics:
        if len(group_metrics[metric]) > 0:
            mean_metrics[metric] = np.mean(group_metrics[metric])
            std = np.std(group_metrics[metric])
            if std > 0:
                mean_metrics[f'{metric}_std'] = np.std(group_metrics[metric])

    print(f'\nGlobal connectivity metrics on group {group}:')
    print(f'Average clustering coefficient: {mean_metrics["avg_clustering"]}')
    print(f'Average neighbor degree: {mean_metrics["avg_neighbor_degree"]}')
    print(f'Number of nodes: {mean_metrics["num_nodes"]}')
    print(f'Number of edges: {mean_metrics["num_edges"]}')

    utils.add_to_csv(mean_metrics, filename)


def apply_threshold(connectivity_matrix, threshold):
    lower_part = connectome.sym_matrix_to_vec(connectivity_matrix, discard_diagonal=True)
    n_connections = int(threshold * len(lower_part))
    max_nconnections_ind = np.argpartition(np.abs(lower_part), -n_connections)[-n_connections:]
    lower_part[~np.isin(np.arange(len(lower_part)), max_nconnections_ind)] = 0
    thresholded_matrix = connectome.vec_to_sym_matrix(lower_part, diagonal=np.diag(connectivity_matrix))

    return thresholded_matrix
