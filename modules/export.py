import numpy as np
import pandas as pd
import seaborn as sns

from modules import atlas_manager
from modules.utils import is_network


def to_gephi(group_name, connectivity_matrix, atlas, output):
    # Connectivity matrix and atlas.labels follow the same order
    n_rois = connectivity_matrix.shape[0]
    if not is_network(atlas.name) and 'schaefer' in atlas.name:
        networks = atlas_manager.get_schaefer_networks_names(atlas.labels)
        network_mapping = {network: i for i, network in enumerate(networks)}
        networks_ids = [network_mapping[region.split('_')[1]] for region in atlas.labels.name]
    else:
        networks_ids = np.zeros(n_rois, dtype=int)
    save_gephi_nodes(group_name, n_rois, networks_ids, output)
    save_gephi_edges(group_name, connectivity_matrix, output)


def save_gephi_nodes(group_name, n_rois, networks_ids, output):
    nodes_colors = []
    ids, zeros = np.arange(n_rois), np.zeros(n_rois, dtype=int)
    for roi in range(n_rois):
        color = sns.color_palette()[networks_ids[roi]]
        nodes_colors.append('%.03f,%.03f,%.03f' % (color[0] * 255, color[1] * 255, color[2] * 255))
    items = np.transpose([ids, zeros, zeros, networks_ids, nodes_colors, zeros, zeros, zeros])
    nodes_df = pd.DataFrame(items, columns=['Id', 'Label', 'Interval', 'Network', 'Color', 'Hub1', 'Hub2', 'Hub3'])
    nodes_df.to_csv(output / f'gephi_nodes_{group_name}.csv', index=False)


def save_gephi_edges(group_name, connectivity_matrix, output):
    connectivity_matrix = np.triu(connectivity_matrix, k=1)
    edges_indices = np.where(connectivity_matrix != 0)
    n_edges = len(edges_indices[0])
    source, target = edges_indices[0], edges_indices[1]
    weights = connectivity_matrix[edges_indices]
    types = np.full(n_edges, 'Undirected')
    ids = np.arange(n_edges)
    zeros, ones = np.zeros(n_edges), np.ones(n_edges)
    items = np.transpose([source, target, types, ids, zeros, zeros, weights, ones, ones, ones])
    edges_df = pd.DataFrame(items, columns=['Source', 'Target', 'Type', 'Id', 'Label', 'Interval', 'Weight', 'Hub1',
                                            'Hub2', 'Hub3'])
    edges_df.to_csv(output / f'gephi_edges_{group_name}.csv', index=False)
