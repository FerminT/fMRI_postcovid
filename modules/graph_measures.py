import numpy as np
import networkx as nx
import bct
from . import utils, atlas_manager
from .connectome_manager import schaefer_networks_from_matrix


def build_graph(connectivity_matrix):
    np.fill_diagonal(connectivity_matrix, 0)
    abs_connectivity_matrix = np.abs(connectivity_matrix)
    return nx.from_numpy_array(abs_connectivity_matrix)


def get_num_nodes_edges(connectivity_matrix):
    connectome = build_graph(connectivity_matrix)
    return len(connectome.nodes), len(connectome.edges)


def add_subject_measures(connectivity_matrix, group_metrics, atlas):
    connectome = build_graph(connectivity_matrix)
    abs_connectivity_matrix = np.abs(connectivity_matrix)
    if not utils.is_network(atlas.name):
        group_metrics['modularity'].append(modularity(connectome))
        if 'schaefer' in atlas.name:
            networks = schaefer_networks_from_matrix(abs_connectivity_matrix, atlas.labels)
            mean_participation_coefficient(connectome, networks, group_metrics['avg_pc'])
    group_metrics['avg_clustering'].append(average_clustering(connectome))
    group_metrics['largest_cc'].append(largest_connected_component(connectome))
    group_metrics['global_efficiency'].append(global_efficiency(abs_connectivity_matrix))
    group_metrics['avg_local_efficiency'].append(mean_local_efficiency(abs_connectivity_matrix))


def average_clustering(connectome):
    return nx.average_clustering(connectome, weight='weight')


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


def compute_group_measures(connectivity_matrices, global_metrics, atlas):
    group_measures = {metric: [] for metric in global_metrics}
    if 'schaefer' in atlas.name and not utils.is_network(atlas.name):
        group_measures['avg_pc'] = {network: [] for network in atlas_manager.get_schaefer_networks_names(atlas.labels)}
    for connectivity_matrix in connectivity_matrices:
        add_subject_measures(connectivity_matrix, group_measures, atlas)

    return group_measures
