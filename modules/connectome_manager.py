import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nilearn import plotting, connectome

from modules import plot, utils, atlas_manager, export
from modules.graph_measures import compute_group_measures, get_num_nodes_edges


def build_connectome(subjects_df, conf_strategy, atlas,
                     thresholds, low_pass, high_pass, smoothing_fwhm, t_r,
                     force, no_plot, output):
    subjects_df = build_timeseries(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r)
    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])
    output.mkdir(exist_ok=True, parents=True)
    if 'schaefer' in atlas.name and not atlas_manager.is_network(atlas.name) and not no_plot:
        groups_diff_over_networks(subjects_df, atlas.labels, output)

    save_connectivity_matrices(subjects_df, atlas.labels, no_plot, output / 'connectivity_matrices')
    groups_analysis(subjects_df, atlas, thresholds, force, no_plot, output)


def build_timeseries(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r):
    subjects_df['time_series'] = subjects_df.apply(lambda subj: utils.time_series(subj['func_path'], subj['mask_path'],
                                                                                  conf_strategy, atlas.maps,
                                                                                  low_pass, high_pass, smoothing_fwhm,
                                                                                  t_r), axis=1)
    subjects_df.drop(columns=['func_path', 'mask_path'], inplace=True)

    return subjects_df


def groups_analysis(subjects_df, atlas, thresholds, force, no_plot, output):
    global_measures = {'avg_clustering': 'Mean Clustering Coefficient', 'global_efficiency': 'Global Efficiency',
                       'avg_local_efficiency': 'Mean Local Efficiency', 'modularity': 'Modularity',
                       'largest_cc': 'Largest Connected Component', 'avg_pc': 'Mean Participation Coefficient'}
    networks_nce = {'SalVentAttn': 'executive', 'DorsAttn': 'attention', 'Cont': 'memory', 'Default': 'global',
                    'Vis': 'visuoespatial', 'Global': 'global'}
    if atlas_manager.is_network(atlas.name):
        global_measures.pop('modularity')
    results_file = output / 'global_measures.csv'
    for threshold in thresholds:
        groups_analysis_at_threshold(subjects_df, atlas, threshold, global_measures, force, no_plot, output,
                                     results_file)
    utils.rank_sum(subjects_df['group'].unique(), global_measures, results_file)
    if not no_plot:
        plot.global_measures(subjects_df, output, global_measures, networks_nce, results_file, atlas.name)


def groups_analysis_at_threshold(subjects_df, atlas, threshold, global_measures, force, no_plot, output, metrics_file):
    threshold_output = output / f'density_{str(int(threshold * 100)).zfill(3)}'
    groups_connectomes = {group: None for group in subjects_df['group'].unique()}
    for group in groups_connectomes:
        group_df = subjects_df[subjects_df['group'] == group]
        thresholded_matrices = group_df['connectivity_matrix'].apply(lambda matrix:
                                                                     apply_threshold(matrix,
                                                                                     threshold))
        global_connectivity_measures(group, global_measures, thresholded_matrices.tolist(), np.round(threshold, 4),
                                     atlas, force, metrics_file)
        group_connectome = mean_connectivity_matrix(thresholded_matrices)
        if not no_plot:
            threshold_output.mkdir(exist_ok=True)
            save_connectome(group, group_connectome, atlas,
                            f'{atlas.name}, {group}', f'{group}_connectome.png', threshold_output)
        groups_connectomes[group] = group_connectome
    if not no_plot:
        save_groups_matrices(groups_connectomes, atlas.labels, threshold_output)


def apply_threshold(connectivity_matrix, threshold):
    lower_part = connectome.sym_matrix_to_vec(connectivity_matrix, discard_diagonal=True)
    n_connections = int(threshold * len(lower_part))
    max_nconnections_ind = np.argpartition(np.abs(lower_part), -n_connections)[-n_connections:]
    lower_part[~np.isin(np.arange(len(lower_part)), max_nconnections_ind)] = 0
    thresholded_matrix = connectome.vec_to_sym_matrix(lower_part, diagonal=np.diag(connectivity_matrix))
    return thresholded_matrix


def groups_diff_over_networks(subjects_df, atlas_labels, conn_output):
    networks_diff, networks_labels = connmatrices_over_networks(subjects_df, atlas_labels)
    plot.networks_corrcoef_boxplot(subjects_df, 'networks_connmatrix', networks_labels,
                                   group_by='group', output=conn_output)
    plot.connectivity_matrix(networks_diff, f'networks_diff', networks_labels,
                             tri='full', output=conn_output)


def connmatrices_over_networks(subjects_df, atlas_labels):
    # Only for Schaefer atlas: compute the difference between groups connectivity matrices over networks
    networks = atlas_manager.get_schaefer_networks_indices(atlas_labels)
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
    print(f'\nMean connectivity matrix: Q test: {q}; degrees of freedom: {df}')

    return mean_connectivity_matrix


def connectivity_matrix(time_series, kind='correlation'):
    connectivity_measure = connectome.ConnectivityMeasure(kind=kind)
    connectivity_matrix = connectivity_measure.fit_transform(time_series)

    return connectivity_matrix, connectivity_measure


def save_connectome(group_name, connectivity_matrix, atlas, fig_title, fig_name, conn_output):
    export.to_gephi(group_name, connectivity_matrix, atlas, conn_output)
    if atlas_manager.is_probabilistic_atlas(atlas.maps):
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
        plot.matrix_on_axis(group_connmatrix, atlas_labels, axes[i], reorder=reorder)
        axes[i].set_title(f'{group}')
    fig.savefig(output / 'groups_comparison.png')


def save_connectivity_matrices(subjects_df, atlas_labels, no_plot, output, reorder=False):
    if not output.exists():
        output.mkdir(parents=True)
    if not no_plot:
        subjects_df.apply(lambda subj: plot.connectivity_matrix(subj['connectivity_matrix'], f'subj_{subj.name}',
                                                                atlas_labels, output, reorder=reorder), axis=1)


def global_connectivity_measures(group, global_measures, connectivity_matrices, threshold, atlas, force, filename):
    if filename.exists() and not force:
        all_computed = utils.check_for_computed_metrics(group, threshold, filename)
        if all_computed:
            print(f'Group {group} on graph density {threshold} already computed')
            return
    group_measures = compute_group_measures(connectivity_matrices, global_measures, atlas)
    group_filename = filename.parent / f'{filename.stem}_{group}.pkl'
    utils.save_networks_pc(group, threshold, group_measures, filename, group_filename)
    utils.add_to_df(group, threshold, group_measures.copy().pop('avg_pc'), group_filename)

    num_nodes, num_edges = get_num_nodes_edges(connectivity_matrices[0])
    mean_measures = utils.compute_mean(group, threshold, group_measures, num_nodes, num_edges, filename)
    measures_values = set(mean_measures.keys()).intersection(global_measures.keys())
    print(f'\nGroup {group}; graph density {threshold}:')
    for measure_name in measures_values:
        print(f'{global_measures[measure_name]}: {mean_measures[measure_name]}')
