import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
from nilearn import plotting
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import auc
from sklearn.svm import SVC

from modules.utils import score_to_bins
from modules.atlas_manager import is_network, get_network_name


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
                         init='pca',
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


def connectivity_matrix(conn_matrix, fig_name, atlas_labels, output,
                        tri='lower', vmin=-0.8, vmax=0.8, reorder=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    matrix_on_axis(conn_matrix, atlas_labels, ax, tri=tri, vmin=vmin, vmax=vmax, reorder=reorder)
    fig.savefig(output / f'{fig_name}.png')
    plt.close(fig)


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


def global_measures(subjects_df, output, global_measures, results_file, atlas):
    atlas_basename = atlas.name if not is_network(atlas.name) else atlas.name.split('_')[0]
    atlas_networks_dirs = [dir_ for dir_ in output.parent.iterdir() if dir_.is_dir() and atlas_basename in dir_.name]
    output = output.parent / atlas_basename
    if not output.exists():
        output.mkdir()
    for measure in global_measures:
        plot_measure(atlas_basename, atlas_networks_dirs, atlas.networks_names, measure, global_measures[measure],
                     output, results_file)
        plot_nce_to_measure(atlas_basename, atlas_networks_dirs, atlas.networks_names, subjects_df, measure,
                            global_measures[measure], atlas.networks_nce, output, results_file)


def plot_measure(atlas_basename, networks_dirs, networks_names, measure_label, measure_desc, output, filename):
    ncols, nrows = 2, -(-len(networks_dirs) // 2)
    fig, axes = plt.subplots(figsize=(15, 5 * nrows), nrows=nrows, ncols=ncols)
    aucs = {network.name: {} for network in networks_dirs}
    for i, network in enumerate(networks_dirs):
        measures_values = pd.read_csv(network / filename.name, index_col=0)
        ax = axes[i // 2, i % 2] if nrows > 1 else axes[i % 2]
        groups = sorted(measures_values['group'].unique())
        for color_index, group in enumerate(groups):
            aucs[network.name][group] = add_group_to_plot(measures_values, group, color_index, measure_label, ax)
        if f'{measure_label}_p' in measures_values.columns:
            p_at_thresholds = measures_values[['threshold', f'{measure_label}_p']].drop_duplicates().set_index(
                'threshold')
            add_statistical_significance(p_at_thresholds, ax, significance_levels=[0.001, 0.005, 0.01])
        network_basename = get_network_name(atlas_basename, network.name)
        ax.set_title(f'{networks_names[network_basename]}')
        ax.set_xlabel('Connection density (%)', fontsize=12)
        ax.set_ylabel(measure_desc, fontsize=12)
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_yticks(ax.get_yticks()[1:-1])
        ax.set_yticklabels([f'{tick:.2f}' for tick in ax.get_yticks()])
        ax.set_xticklabels([f'{tick * 100:.0f}' for tick in ax.get_xticks()])
    fig.suptitle(measure_desc)
    fig.savefig(output / f'{measure_label}.png')
    plt.show()

    return aucs


def add_group_to_plot(measures_values, group, color_index, measure_label, ax):
    group_values = measures_values[measures_values['group'] == group]
    densities, auc_value = group_values['threshold'].values, 0.0
    if measure_label in group_values.columns:
        measure_values = group_values[measure_label].values
        lower_error, upper_error = group_values[measure_label] - group_values[f'{measure_label}_ste'], \
                                   group_values[measure_label] + group_values[f'{measure_label}_ste']
        sorted_densities = np.argsort(densities)
        add_curve(densities, measure_values, lower_error, upper_error, group, color_index, ax)
        if len(densities) > 1:
            auc_value = auc(densities[sorted_densities], measure_values[sorted_densities])
    return auc_value


def add_statistical_significance(p_at_thresholds, ax, significance_levels, eps=1e-4):
    pvalues = p_at_thresholds[p_at_thresholds.columns[0]]
    labels = ['*' * i for i in range(len(significance_levels), 0, -1)]
    significance_levels.insert(0, 0.0)
    significance_levels.append(1.)
    labels.append('ns')
    categorized_pvalues = pd.cut(pvalues, significance_levels, right=False, labels=labels)
    spacing = 0.1
    if len(pvalues) > 1:
        spacing = pvalues.index[1] - pvalues.index[0] + eps

    significance_bar(ax, categorized_pvalues, labels, spacing)


def plot_nce_to_measure(atlas_basename, networks_dirs, networks_names, subjects_df, measure_label, measure_desc,
                        networks_nce, output, filename):
    ncols, nrows = 2, -(-len(networks_dirs) // 2)
    fig, axes = plt.subplots(figsize=(15, 5 * nrows), nrows=nrows, ncols=ncols)
    gains = {network.name: {} for network in networks_dirs}
    for i, network in enumerate(networks_dirs):
        ax = axes[i // 2, i % 2] if nrows > 1 else axes[i % 2]
        network_basename = get_network_name(atlas_basename, network.name)
        if network_basename not in networks_nce:
            continue
        network_nce = networks_nce[network_basename]
        groups = sorted(subjects_df['group'].unique())
        connection_density, measure_df, group_mapping = get_measure_at_threshold(subjects_df, groups, measure_label,
                                                                                 network, network_nce, filename)
        sns.scatterplot(data=measure_df, x='nce', y='measure', hue='group', ax=ax)
        if not measure_df.empty:
            gains[network.name] = fit_and_plot_svm(measure_df, group_mapping, ax)
        ax.legend()
        ax.set_title(f'{networks_names[network_basename]}')
        ax.set_xlabel(f'{network_nce} score')
        ax.set_ylabel(f'{measure_desc} at t={connection_density * 100:.0f}%')
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    fig.suptitle(measure_desc)
    fig.savefig(output / f'NCE_to_{measure_label}.png')
    plt.show()

    return gains


def get_measure_at_threshold(subjects_df, groups, measure_label, network, network_nce, filename):
    graph_density, nces, values, categories, group_mapping = 0.0, [], [], [], {}
    for j, group in enumerate(groups):
        group_df = subjects_df[subjects_df['group'] == group]
        group_mapping[group] = j
        group_network_measures = pd.read_pickle(network / f'{filename.stem}_{group}.pkl')
        measures_at_threshold = group_network_measures.sort_values(by='threshold').iloc[-1]
        if measure_label not in measures_at_threshold.index:
            continue
        graph_density = measures_at_threshold['threshold']
        nces.extend(group_df[network_nce].values)
        values.extend(measures_at_threshold[measure_label])
        categories.extend([group] * len(group_df))
    df = pd.DataFrame({'nce': nces, 'measure': values, 'group': categories}).dropna()
    df = normalize_values(df, ['nce', 'measure'])
    return graph_density, df, group_mapping


def normalize_values(df, columns):
    df[columns] = df[columns].astype(float)
    for column in columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df


def fit_and_plot_svm(df, group_mapping, ax):
    df = df.replace({'group': group_mapping})
    clf_nces, clf = SVC(), SVC()
    features = df[['nce', 'measure']].values
    nces = features[:, 0].reshape(-1, 1)
    categories = df['group'].values
    accuracies_nce, accuracies_topology = [], []
    for i in range(len(features)):
        clf_nces.fit(np.delete(nces, i, axis=0), np.delete(categories, i))
        accuracies_nce.append(clf_nces.score(nces[i].reshape(1, -1), categories[i].reshape(1, -1)))
        clf.fit(np.delete(features, i, axis=0), np.delete(categories, i))
        accuracies_topology.append(clf.score(features[i].reshape(1, -1), categories[i].reshape(1, -1)))
    nce_mean, nce_std = np.mean(accuracies_nce), np.std(accuracies_nce)
    topology_mean, topology_std = np.mean(accuracies_topology), np.std(accuracies_topology)
    clf.fit(features, categories)
    xx, yy = meshgrid(features[:, 0], features[:, 1])
    add_decision_boundaries(ax, clf, xx, yy, cmap='coolwarm', alpha=0.1)

    return topology_mean - nce_mean, np.sqrt(nce_std ** 2 + topology_std ** 2)


def add_curve(graph_densities, measure, lower_error, upper_error, group, color_index, ax):
    ax.plot(graph_densities, measure, label=group, color=f'C{color_index}')
    ax.plot(graph_densities, lower_error, alpha=0.1, color=f'C{color_index}')
    ax.plot(graph_densities, upper_error, alpha=0.1, color=f'C{color_index}')
    ax.legend()
    ax.fill_between(graph_densities, lower_error, upper_error, alpha=0.2)


def significance_bar(ax, categorized_pvalues, labels, spacing):
    line_y = ax.get_ylim()[1]
    max_threshold, min_threshold = categorized_pvalues.index[-1], categorized_pvalues.index[0]
    # Use light grey for *, dark grey for **, and black for ***
    colors = {label: col for label, col in zip(labels, colormaps.get_cmap('Greys')(np.linspace(0.8, 0.2, len(labels))))}
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

            significant_regions = [(start - spacing, end) for start, end in significant_regions]
            for start, end in significant_regions:
                if start < min_threshold:
                    start = min_threshold
                ax.plot((start, end), [line_y * 0.98, line_y * 0.98], linewidth=2, color=colors[label])


def meshgrid(x, y, h=.02, offset=0.07):
    x_min, x_max = x.min() - offset, x.max() + offset
    y_min, y_max = y.min() - offset, y.max() + offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def add_decision_boundaries(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return ax.contourf(xx, yy, Z, **params)
