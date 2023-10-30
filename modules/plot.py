import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nilearn import plotting
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import auc

from modules.utils import score_to_bins
from modules.atlas_manager import is_network


def rdm(rdm, subjects_df, title, output, method='TSNE', clinical_score='global', draw_labels=False):
    embedding = initialize_embedding(method)
    embeddings = embedding.fit_transform(rdm)
    rdm_df = subjects_df.copy()
    rdm_df['emb_x'], rdm_df['emb_y'] = embeddings[:, 0], embeddings[:, 1]
    rdm_df, sizes = score_to_bins(rdm_df, clinical_score)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    sns.scatterplot(rdm_df, x='emb_x', y='emb_y', hue='group', style='cluster', size=clinical_score, sizes=sizes,
                    hue_order=sorted(rdm_df['group'].unique()), ax=ax)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    if draw_labels:
        for i, txt in enumerate(rdm_df.index.to_list()):
            ax.annotate(txt, (embeddings[i, 0], embeddings[i, 1]), alpha=0.6)

    title = title.replace(' ', '_') + f'_{method}'
    ax.set_title(title)
    ax.axes.xaxis.set_visible(False), ax.axes.yaxis.set_visible(False)
    plt.show()

    output.mkdir(exist_ok=True, parents=True)
    fig.savefig(output / f'{title}.png')

    return embeddings


def initialize_embedding(method):
    if method == 'MDS':
        embedding = MDS(n_components=2,
                        dissimilarity='precomputed',
                        random_state=42)
    elif method == 'TSNE':
        embedding = TSNE(n_components=2,
                         perplexity=20,
                         random_state=42)
    elif method == 'ISOMAP':
        embedding = Isomap(n_components=2,
                           n_neighbors=10,
                           n_jobs=-1)
    elif method == 'PCA':
        embedding = PCA(n_components=2)
    else:
        raise NotImplementedError(f'Method {method} not implemented')

    return embedding


def global_measures(subjects_df, output, global_measures, networks_nce, results_file, atlas_name):
    atlas_basename = atlas_name if not is_network(atlas_name) else atlas_name.split('_')[0]
    atlas_networks = [dir_.name for dir_ in output.parent.iterdir() if dir_.is_dir() and atlas_basename in dir_.name]
    for measure in global_measures:
        plot_measure(atlas_basename, atlas_networks, measure, global_measures[measure],
                     output.parent, results_file)
        plot_measure_to_nce(atlas_basename, atlas_networks, subjects_df, measure, global_measures[measure],
                            networks_nce, output.parent, results_file)


def plot_measure(atlas_basename, networks, measure_label, measure_desc, output, filename):
    fig, axes = plt.subplots(figsize=(15, 15), nrows=len(networks) // 2 + 1, ncols=2)
    aucs = {network: {} for network in networks}
    for i, network in enumerate(networks):
        measures_values = pd.read_csv(output / network / filename.name, index_col=0)
        ax = axes[i // 2, i % 2]
        groups = sorted(measures_values['group'].unique())
        for color_index, group in enumerate(groups):
            group_values = measures_values[measures_values['group'] == group]
            densities = group_values['threshold'].values
            if measure_label not in group_values.columns:
                continue
            measure_values = group_values[measure_label].values
            lower_error, upper_error = group_values[measure_label] - group_values[f'{measure_label}_ste'], \
                                       group_values[measure_label] + group_values[f'{measure_label}_ste']
            sorted_densities = np.argsort(densities)
            aucs[network][group] = auc(densities[sorted_densities], measure_values[sorted_densities])
            add_curve(densities, measure_values, lower_error, upper_error, group, color_index, ax)
        if f'{measure_label}_p' in measures_values.columns:
            p_at_thresholds = measures_values[['threshold', f'{measure_label}_p']].drop_duplicates().set_index(
                'threshold')
            add_statistical_significance(p_at_thresholds, ax, significance_levels=[0.01])
        network_name = network.strip(f'{atlas_basename}_') if is_network(network) else 'Global'
        ax.set_title(f'{network_name}')
        ax.set_xlabel('Graph density')
        ax.set_ylabel(measure_desc)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(measure_desc)
    fig.savefig(output / f'{measure_label}.png')
    plt.show()

    return aucs


def add_statistical_significance(p_at_thresholds, ax, significance_levels, eps=1e-4):
    pvalues = p_at_thresholds[p_at_thresholds.columns[0]]
    labels = ['*' * i for i in range(len(significance_levels), 0, -1)]
    significance_levels.insert(0, 0.0)
    significance_levels.append(1.)
    labels.append('ns')
    categorized_pvalues = pd.cut(pvalues, significance_levels, right=False, labels=labels)
    spacing = pvalues.index[1] - pvalues.index[0] + eps

    significance_bar(ax, categorized_pvalues, labels, spacing)


def plot_measure_to_nce(atlas_basename, networks, subjects_df, measure_label, measure_desc, networks_nce,
                        output, filename):
    fig, axes = plt.subplots(figsize=(15, 15), nrows=len(networks) // 2 + 1, ncols=2)
    for i, network in enumerate(networks):
        ax = axes[i // 2, i % 2]
        network_name = network.strip(f'{atlas_basename}_') if is_network(network) else 'Global'
        if network_name not in networks_nce:
            continue
        network_nce = networks_nce[network_name]
        groups = sorted(subjects_df['group'].unique())
        for group in groups:
            group_df = subjects_df[subjects_df['group'] == group]
            group_network_measures = pd.read_pickle(output / network / f'{filename.stem}_{group}.pkl')
            measures_at_threshold = group_network_measures.sort_values(by='threshold').iloc[-1]
            if measure_label not in measures_at_threshold.index:
                continue
            ax.scatter(measures_at_threshold[measure_label], group_df[network_nce].values, label=group)
        ax.set_title(f'{network_name}')
        ax.set_xlabel(f'{measure_desc}')
        ax.set_ylabel(f'{network_nce} score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(measure_desc)
    fig.savefig(output / f'{measure_label}_to_NCE.png')
    plt.show()


def networks_corrcoef_boxplot(subjects_df, attribute, networks_labels, group_by, output):
    nrows = len(networks_labels) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(20, 20))
    for i, network in enumerate(networks_labels.name):
        ax = axes[i // nrows, i % nrows]
        network_means, groups = {}, subjects_df[group_by].unique()
        for group in groups:
            networks_connmatrix = subjects_df[subjects_df[group_by] == group][attribute].to_list()
            network_means[group] = [subj_connmatrix[i, i] for subj_connmatrix in networks_connmatrix]
        df = pd.DataFrame.from_dict(data=network_means, orient='index').transpose()
        sns.boxplot(data=df, order=sorted(groups), ax=ax)
        ax.set_title(network)
        ax.set_ylabel('Mean correlation coefficient')
    fig.suptitle(f'Mean correlation coefficients by network and group', fontsize=20)
    fig.savefig(output / f'networks_mean_corrcoef.png')
    plt.show()


def matrix_on_axis(connectivity_matrix, atlas_labels, ax,
                   tri='lower', vmin=-0.8, vmax=0.8, reorder=False):
    # Get labels in the correct format until plot_matrix is fixed
    labels = list(atlas_labels.name.values)
    plotting.plot_matrix(connectivity_matrix,
                         tri=tri,
                         labels=labels,
                         colorbar=True,
                         vmin=vmin,
                         vmax=vmax,
                         reorder=reorder,
                         axes=ax)


def add_curve(graph_densities, measure, lower_error, upper_error, group, color_index, ax):
    ax.plot(graph_densities, measure, label=group, color=f'C{color_index}')
    ax.plot(graph_densities, lower_error, alpha=0.1, color=f'C{color_index}')
    ax.plot(graph_densities, upper_error, alpha=0.1, color=f'C{color_index}')
    ax.legend()
    ax.fill_between(graph_densities, lower_error, upper_error, alpha=0.2)


def connectivity_matrix(conn_matrix, fig_name, atlas_labels, output,
                        tri='lower', vmin=-0.8, vmax=0.8, reorder=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix_on_axis(conn_matrix, atlas_labels, ax, tri=tri, vmin=vmin, vmax=vmax, reorder=reorder)
    fig.savefig(output / f'{fig_name}.png')
    plt.close(fig)


def significance_bar(ax, categorized_pvalues, labels, spacing):
    for label in labels:
        significant_values = categorized_pvalues[categorized_pvalues == label]
        # Build a list of tuples with the start and end of each significant region
        if len(significant_values) > 0 and label not in 'ns':
            significant_regions = [(significant_values.index[0], significant_values.index[0])]
            for i, threshold in enumerate(significant_values.index):
                if i > 0:
                    if threshold - significant_values.index[i - 1] > spacing:
                        significant_regions.append((threshold, threshold))
                    else:
                        significant_regions[-1] = (significant_regions[-1][0], threshold)
            for region in significant_regions:
                ax.plot(region, [ax.get_ylim()[1] * 0.98, ax.get_ylim()[1] * 0.98], marker=f'{label}', linewidth=1,
                        color='k', alpha=0.8)
