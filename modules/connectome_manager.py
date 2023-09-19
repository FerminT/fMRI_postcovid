import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import bct
from . import utils, atlas_manager

# NiLearn methods and classes
from nilearn import plotting
from nilearn import connectome


def build_connectome(subjects_df, conf_strategy, atlas,
                     thresholds, low_pass, high_pass, smoothing_fwhm, t_r,
                     force, no_plot, output):
    subjects_df = build_timeseries(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r)
    subjects_df['connectivity_matrix'] = subjects_df['time_series'].apply(lambda time_series:
                                                                          connectivity_matrix([time_series])[0][0])
    output.mkdir(exist_ok=True, parents=True)
    if 'schaefer' in atlas.name and not utils.is_network(atlas.name) and not no_plot:
        groups_diff_over_networks(subjects_df, atlas.labels, output)

    save_connectivity_matrices(subjects_df, atlas.labels, no_plot, output / 'connectivity_matrices')
    groups_connectome_analysis(subjects_df, atlas, thresholds, force, no_plot, output)


def build_timeseries(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r):
    subjects_df['time_series'] = subjects_df.apply(lambda subj: utils.time_series(subj['func_path'], subj['mask_path'],
                                                                                  conf_strategy, atlas.maps,
                                                                                  low_pass, high_pass, smoothing_fwhm,
                                                                                  t_r), axis=1)
    subjects_df.drop(columns=['func_path', 'mask_path'], inplace=True)

    return subjects_df


def groups_connectome_analysis(subjects_df, atlas, thresholds, force, no_plot, output):
    global_metrics = {'avg_clustering': 'Mean Clustering Coefficient', 'global_efficiency': 'Global Efficiency',
                      'avg_local_efficiency': 'Mean Local Efficiency', 'modularity': 'Modularity',
                      'largest_cc': 'Largest Connected Component', 'avg_pc': 'Mean Participation Coefficient'}
    metrics_file = output / 'global_metrics.csv'
    for threshold in thresholds:
        threshold_output = output / f'density_{str(int(threshold * 100)).zfill(3)}'
        threshold_output.mkdir(exist_ok=True)
        groups_connectomes = {group: None for group in subjects_df['group'].unique()}
        for group in groups_connectomes:
            group_df = subjects_df[subjects_df['group'] == group]
            thresholded_matrices = group_df['connectivity_matrix'].apply(lambda matrix:
                                                                         apply_threshold(matrix,
                                                                                         threshold))
            global_connectivity_metrics(group, global_metrics, thresholded_matrices.tolist(), threshold, atlas,
                                        force, metrics_file)
            group_connectome = mean_connectivity_matrix(thresholded_matrices)
            if not no_plot:
                save_connectome(group_connectome, atlas,
                                f'{atlas.name}, {group}', f'{group}_connectome.png', threshold_output)
            groups_connectomes[group] = group_connectome
        if not no_plot:
            save_groups_matrices(groups_connectomes, atlas.labels, threshold_output)
    utils.rank_sum(subjects_df['group'].unique(), global_metrics, metrics_file)
    if not no_plot:
        for metric in global_metrics:
            atlas_basename = atlas.name if not utils.is_network(atlas.name) else atlas.name.split('_')[0]
            atlas_networks = [dir_.name for dir_ in output.parent.iterdir() if
                              dir_.is_dir() and atlas_basename in dir_.name]
            utils.plot_measure(atlas_basename, atlas_networks, metric, global_metrics[metric],
                               output.parent, metrics_file)


def groups_diff_over_networks(subjects_df, atlas_labels, conn_output):
    networks_diff, networks_labels = connmatrices_over_networks(subjects_df, atlas_labels)
    utils.networks_corrcoef_boxplot(subjects_df, 'networks_connmatrix', networks_labels,
                                    group_by='group', output=conn_output)
    utils.save_connectivity_matrix(networks_diff, f'networks_diff', networks_labels,
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


def schaefer_networks_from_matrix(connectivity_matrix, atlas_labels):
    networks_names = atlas_manager.get_schaefer_networks_names(atlas_labels)
    networks = {network: {'connectome': None, 'nodes': []} for network in networks_names}
    for network in networks:
        network_indices = atlas_labels[atlas_labels.name.str.contains(network)].index.to_list()
        networks[network]['nodes'] = network_indices
        networks[network]['connectome'] = connectivity_matrix[network_indices, :][:, network_indices]

    return networks


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


def save_connectivity_matrices(subjects_df, atlas_labels, no_plot, output, reorder=False):
    if not output.exists():
        output.mkdir(parents=True)
    if not no_plot:
        subjects_df.apply(lambda subj: utils.save_connectivity_matrix(subj['connectivity_matrix'], f'subj_{subj.name}',
                                                                      atlas_labels, output, reorder=reorder), axis=1)


def global_efficiency(connectivity_matrix):
    e_glob = bct.efficiency_wei(connectivity_matrix, local=False)
    return e_glob


def mean_local_efficiency(connectivity_matrix):
    e_loc = bct.efficiency_wei(connectivity_matrix, local=True)
    mean_e_loc = np.mean(e_loc)
    return mean_e_loc


def modularity(connectome):
    partitions = nx.community.louvain_communities(connectome, weight='weight')
    q = nx.community.modularity(connectome, partitions, weight='weight')
    return q


def largest_connected_component(connectome):
    largest_cc = max(nx.connected_components(connectome), key=len)
    return len(largest_cc) / len(connectome.nodes)


def mean_participation_coefficient(connectome, module_partition, modules_pc):
    for module in module_partition:
        module_subgraph = set(module_partition[module]['nodes'])
        nodes_pc = []
        for node in module_subgraph:
            degree = float(nx.degree(G=connectome, nbunch=node))
            # intramodule degree of node
            wm_degree = float(sum([1 for u in module_subgraph if (u, node) in connectome.edges()]))

            # The participation coefficient is 1 - the square of
            # the ratio of the within module degree and the total degree
            if degree == 0:
                nodes_pc.append(0)
            else:
                nodes_pc.append(1 - (wm_degree / degree) ** 2)
        modules_pc[module].append(np.mean(nodes_pc))

    return modules_pc


def global_connectivity_metrics(group, global_metrics, connectivity_matrices, threshold, atlas, force, filename):
    if filename.exists() and not force:
        computed_thresholds = pd.read_csv(filename, index_col=0)
        if group in computed_thresholds['group'].unique():
            group_thresholds = computed_thresholds[computed_thresholds['group'] == group]['threshold'].values
            if np.round(threshold, 4) in group_thresholds:
                print(f'Group {group} on graph density {threshold} already computed')
                return
    group_metrics = {metric: [] for metric in global_metrics}
    if 'schaefer' in atlas.name and not utils.is_network(atlas.name):
        group_metrics['avg_pc'] = {network: [] for network in atlas_manager.get_schaefer_networks_names(atlas.labels)}
    num_nodes, num_edges = 0, 0
    for connectivity_matrix in connectivity_matrices:
        np.fill_diagonal(connectivity_matrix, 0)
        abs_connectivity_matrix = np.abs(connectivity_matrix)
        connectome = nx.from_numpy_array(abs_connectivity_matrix)
        if not utils.is_network(atlas.name):
            group_metrics['modularity'].append(modularity(connectome))
            if 'schaefer' in atlas.name:
                networks = schaefer_networks_from_matrix(abs_connectivity_matrix, atlas.labels)
                mean_participation_coefficient(connectome, networks, group_metrics['avg_pc'])
        group_metrics['avg_clustering'].append(nx.average_clustering(connectome, weight='weight'))
        group_metrics['largest_cc'].append(largest_connected_component(connectome))
        group_metrics['global_efficiency'].append(global_efficiency(abs_connectivity_matrix))
        group_metrics['avg_local_efficiency'].append(mean_local_efficiency(abs_connectivity_matrix))
        num_nodes, num_edges = len(connectome.nodes), len(connectome.edges)

    group_filename = filename.parent / f'{filename.stem}_{group}.pkl'
    for network in group_metrics['avg_pc']:
        network_path = filename.parents[1] / f'{filename.parent.name}_{network}'
        values = group_metrics['avg_pc'][network]
        if network_path.exists():
            network_file = network_path / group_filename.name
            utils.add_to_df(group, threshold, {'avg_pc': values}, network_file)
    group_metrics_cp = group_metrics.copy()
    group_metrics_cp.pop('avg_pc')
    utils.add_to_df(group_metrics_cp, threshold, group_metrics, group_filename)
    mean_metrics = utils.compute_mean(group, threshold, group_metrics, num_nodes, num_edges, filename)

    print(f'\nGroup {group}; graph density {np.round(threshold, 4)}:')
    print(f'Average clustering coefficient: {mean_metrics["avg_clustering"]}')
    print(f'Average global efficiency: {mean_metrics["global_efficiency"]}')
    print(f'Average local efficiency: {mean_metrics["avg_local_efficiency"]}')
    print(f'Average modularity: {mean_metrics["modularity"]}')
    print(f'Average largest cc: {mean_metrics["largest_cc"]}')


def apply_threshold(connectivity_matrix, threshold):
    lower_part = connectome.sym_matrix_to_vec(connectivity_matrix, discard_diagonal=True)
    n_connections = int(threshold * len(lower_part))
    max_nconnections_ind = np.argpartition(np.abs(lower_part), -n_connections)[-n_connections:]
    lower_part[~np.isin(np.arange(len(lower_part)), max_nconnections_ind)] = 0
    thresholded_matrix = connectome.vec_to_sym_matrix(lower_part, diagonal=np.diag(connectivity_matrix))

    return thresholded_matrix
