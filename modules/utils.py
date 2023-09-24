import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import mannwhitneyu
from filelock import FileLock
from . import atlas_manager

# NiLearn methods and classes
from nilearn import image, plotting, connectome
from nilearn.interfaces import fmriprep
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker


def plot_rdm(rdm, subjects_df, title, output, method='TSNE', clinical_score='global', draw_labels=False):
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


def save_connectivity_matrix(conn_matrix, fig_name, atlas_labels, output,
                             tri='lower', vmin=-0.8, vmax=0.8, reorder=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_matrix_on_axis(conn_matrix, atlas_labels, ax, tri=tri, vmin=vmin, vmax=vmax, reorder=reorder)
    fig.savefig(output / f'{fig_name}.png')
    plt.close(fig)


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


def plot_matrix_on_axis(connectivity_matrix, atlas_labels, ax,
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


def time_series(func_data, brain_mask, conf_strategy, atlas_maps, low_pass, high_pass, smoothing_fwhm, t_r):
    kwargs = {'mask_img': brain_mask, 'smoothing_fwhm': smoothing_fwhm, 'low_pass': low_pass, 'high_pass': high_pass,
              't_r': t_r, 'standardize': False, 'detrend': True, 'memory': 'cache', 'memory_level': 2}
    if is_probabilistic_atlas(atlas_maps):
        nifti_masker = NiftiMapsMasker(maps_img=atlas_maps,
                                       **kwargs)
    else:
        nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                         **kwargs)
    confounds, sample_mask = fmriprep.load_confounds(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series


def is_network(atlas_name):
    return len(atlas_name.split('_')) > 1


def is_probabilistic_atlas(atlas_maps):
    atlas_maps_img = image.load_img(atlas_maps)
    return len(atlas_maps_img.shape) == 4


def pad_timeseries(timeseries, pad_value=np.nan):
    n_timepoints = timeseries.apply(lambda ts: ts.shape[0]).value_counts().index[0]
    timeseries = timeseries.apply(lambda ts: np.pad(ts, ((0, n_timepoints - ts.shape[0]), (0, 0)),
                                                    'constant', constant_values=pad_value))

    return timeseries


def load_subjects(subjects, data_path, clinical_file):
    if subjects == 'all':
        subjects = [sub for sub in data_path.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [data_path / f'sub-{subjects.zfill(3)}']

    subjects_df = load_clinical_data(clinical_file)
    subjects_df = load_datapaths(subjects, subjects_df)

    return subjects_df


def load_clinical_data(clinical_datafile):
    subjects_df = pd.read_csv(clinical_datafile)
    subjects_df = subjects_df.astype({'id': int})
    subjects_df = subjects_df.set_index('id')

    return subjects_df


def load_datapaths(subjects_paths, subjects_df):
    for subj_path in subjects_paths:
        subj_id = int(subj_path.name.split('-')[1])
        if subj_id in subjects_df.index:
            func_path = subj_path / 'func'
            func_file = [f for f in func_path.glob('*.nii.gz') if 'preproc' in f.name][0]
            mask_file = [f for f in func_path.glob('*.nii.gz') if 'brain_mask' in f.name][0]
            subjects_df.loc[subj_id, 'func_path'] = str(func_file)
            subjects_df.loc[subj_id, 'mask_path'] = str(mask_file)

    return subjects_df


def score_to_bins(df, score, n_bins=3):
    df[score] = pd.qcut(df[score], q=n_bins)
    df[score] = df[score].cat.codes
    sizes = np.linspace(30, 100, n_bins)
    if -1 in df[score].unique():
        sizes = np.insert(sizes, 0, 100)
    sizes = {size: sizes[i] for i, size in enumerate(sorted(df[score].unique()))}

    return df, sizes


def q_test(data, mean):
    df = mean.shape[0] * (mean.shape[0] - 1) / 2
    data, mean = np.triu(data, k=1), np.triu(mean, k=1)
    q = np.sum(np.sum(np.square(data - mean)) / (len(data) - 1))
    return q, df


# TODO: refactorizar
def add_to_csv(dict_values, filename):
    series = pd.Series(dict_values)
    lock = FileLock(filename.parent / f'{filename.name}.lock')
    with lock:
        if filename.exists():
            df = pd.read_csv(filename, index_col=0)
        else:
            df = pd.DataFrame(columns=list(dict_values.keys()))
        group_df = df[df['group'] == series['group']]
        if series['threshold'] in group_df['threshold'].values:
            idx = df[(df['threshold'] == series['threshold']) & (df['group'] == series['group'])].index[0]
            for metric in dict_values:
                if metric in df and isinstance(series[metric], list):
                    df[metric] = df[metric].astype(object)
                df.at[idx, metric] = series[metric]
        else:
            df = pd.concat([df, series.to_frame().T], ignore_index=True)
        df.to_csv(filename)


def add_to_df(group, threshold, group_metrics, group_filename):
    group_metrics = group_metrics.copy()
    group_metrics['group'] = group
    group_metrics['threshold'] = threshold
    series = pd.Series(group_metrics)
    lock = FileLock(group_filename.parent / f'{group_filename.name}.lock')
    with lock:
        if group_filename.exists():
            df = pd.read_pickle(group_filename)
        else:
            df = pd.DataFrame(columns=list(group_metrics.keys()))
        if series['threshold'] in df['threshold'].values:
            idx = df[(df['threshold'] == series['threshold']) & (df['group'] == series['group'])].index[0]
            for metric in group_metrics:
                if metric not in df:
                    df[metric] = None
                df.at[idx, metric] = series[metric]
        else:
            df = pd.concat([df, series.to_frame().T], ignore_index=True)
        df.to_pickle(group_filename)


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


def add_curve(graph_densities, measure, lower_error, upper_error, group, ax):
    ax.plot(graph_densities, measure, label=group)
    ax.plot(graph_densities, lower_error, alpha=0.1)
    ax.plot(graph_densities, upper_error, alpha=0.1)
    ax.legend()
    ax.fill_between(graph_densities, lower_error, upper_error, alpha=0.2)


def add_statistical_significance(p_at_thresholds, ax, significance_level=0.05, eps=1e-4):
    pvalues = p_at_thresholds[p_at_thresholds.columns[0]]
    significant_values = pvalues[pvalues < significance_level]
    spacing = pvalues.index[1] - pvalues.index[0] + eps
    # Build a list of tuples with the start and end of each significant region
    if len(significant_values) > 0:
        significant_regions = [(significant_values.index[0], significant_values.index[0])]
        for i, threshold in enumerate(significant_values.index):
            if i > 0:
                if threshold - significant_values.index[i - 1] > spacing:
                    significant_regions.append((threshold, threshold))
                else:
                    significant_regions[-1] = (significant_regions[-1][0], threshold)
        for region in significant_regions:
            ax.plot(region, [ax.get_ylim()[1] * 0.98, ax.get_ylim()[1] * 0.98], linewidth=1, color='k', alpha=0.5)


def plot_measure(atlas_basename, networks, measure_label, measure_desc, output, filename):
    fig, axes = plt.subplots(figsize=(15, 15), nrows=len(networks) // 2 + 1, ncols=2)
    aucs = {network: {} for network in networks}
    for i, network in enumerate(networks):
        metrics_values = pd.read_csv(output / network / filename.name, index_col=0)
        ax = axes[i // 2, i % 2]
        groups = metrics_values['group'].unique()
        for group in groups:
            group_values = metrics_values[metrics_values['group'] == group]
            densities = group_values['threshold'].values
            if measure_label not in group_values.columns:
                continue
            measure_values = group_values[measure_label].values
            lower_error, upper_error = group_values[measure_label] - group_values[f'{measure_label}_ste'], \
                                       group_values[measure_label] + group_values[f'{measure_label}_ste']
            sorted_densities = np.argsort(densities)
            aucs[network][group] = auc(densities[sorted_densities], measure_values[sorted_densities])
            add_curve(densities, measure_values, lower_error, upper_error, group, ax)
        if f'{measure_label}_p' in metrics_values.columns:
            p_at_thresholds = metrics_values[['threshold', f'{measure_label}_p']].drop_duplicates().set_index('threshold')
            add_statistical_significance(p_at_thresholds, ax)
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


def compute_mean(group, threshold, group_metrics, num_nodes, num_edges, filename):
    mean_metrics = {'group': group, 'threshold': threshold}
    for metric in group_metrics:
        values = group_metrics[metric]
        if len(values) > 0:
            if isinstance(values, dict):
                networks_values = list(values.values())
                means = np.mean(networks_values, axis=1)
                stes = np.std(networks_values, axis=1) / np.sqrt(np.shape(networks_values)[1])
                for idx, network in enumerate(values):
                    network_path = filename.parents[1] / f'{filename.parent.name}_{network}'
                    if network_path.exists():
                        network_file = network_path / filename.name
                        add_to_csv({'group': group, 'threshold': threshold, metric: means[idx],
                                    f'{metric}_ste': stes[idx]}, network_file)
            else:
                mean_metrics[metric] = np.mean(values)
                ste = np.std(values) / np.sqrt(len(values))
                if ste > 0:
                    mean_metrics[f'{metric}_ste'] = ste
    mean_metrics['num_nodes'] = num_nodes
    mean_metrics['num_edges'] = num_edges
    add_to_csv(mean_metrics, filename)

    return mean_metrics


def rank_sum(groups, global_metrics, metrics_file):
    mean_measurements = pd.read_csv(metrics_file, index_col=0)
    groups_measurements = [pd.read_pickle(metrics_file.parent / f'{metrics_file.stem}_{group}.pkl') for group in groups]
    measures = set(global_metrics.keys()).intersection(groups_measurements[0].columns)
    densities = groups_measurements[0]['threshold'].values
    for density in densities:
        fst_group = groups_measurements[0][groups_measurements[0]['threshold'] == density]
        snd_group = groups_measurements[1][groups_measurements[1]['threshold'] == density]
        if not len(fst_group) > 0 and len(snd_group) > 0:
            print(f'No group measurements found for density {density}')
            continue
        for measure in measures:
            x, y = fst_group[measure].tolist()[0], snd_group[measure].tolist()[0]
            if len(x) > 0 and len(y) > 0:
                _, pvalue = mannwhitneyu(x, y)
                mean_measurements.loc[mean_measurements['threshold'] == density, f'{measure}_p'] = pvalue

    mean_measurements.to_csv(metrics_file)


def save_gephi_data(group_name, connectivity_matrix, atlas, conn_output):
    # Connectivity matrix and atlas.labels follow the same order
    n_rois = connectivity_matrix.shape[0]
    ids, zeros = np.arange(n_rois), np.zeros(n_rois)
    if not is_network(atlas.name) and 'schaefer' in atlas.name:
        networks = atlas_manager.get_schaefer_networks_names(atlas.labels)
        network_mapping = {network: i for i, network in enumerate(networks)}
        networks_ids = [network_mapping[region.split('_')[1]] for region in atlas.labels.name]
    else:
        networks_ids = zeros
    nodes_colors = []
    for roi in range(n_rois):
        color = sns.color_palette()[networks_ids[roi]]
        nodes_colors.append('%.03f,%.03f,%.03f' % (color[0] * 255, color[1] * 255, color[2] * 255))
    items = np.transpose([ids, zeros, zeros, networks_ids, nodes_colors, zeros, zeros, zeros])
    nodes_df = pd.DataFrame(items, columns=['Id', 'Label', 'Interval', 'Network', 'Color', 'Hub1', 'Hub2', 'Hub3'])
    nodes_df.to_csv(conn_output / f'gephi_nodes_{group_name}.csv', index=False)

    # Upper triangulize connectivity matrix
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
    edges_df.to_csv(conn_output / f'gephi_edges_{group_name}.csv', index=False)
