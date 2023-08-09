import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.decomposition import PCA

# NiLearn methods and classes
from nilearn import image, plotting, connectome
from nilearn.interfaces import fmriprep
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker


def plot_rdm(rdm, subjects_df, title, output, method='TSNE', draw_labels=False):
    embedding = initialize_embedding(method)
    embeddings = embedding.fit_transform(rdm)
    subjects_df['emb_x'], subjects_df['emb_y'] = embeddings[:, 0], embeddings[:, 1]
    fig, ax = plt.subplots()
    sns.scatterplot(subjects_df, x='emb_x', y='emb_y', hue='group', style='cluster', size='composite_global',
                    hue_order=sorted(subjects_df['group'].unique()), ax=ax)
    if draw_labels:
        for i, txt in enumerate(subjects_df.index.to_list()):
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


def q_test(data, mean):
    df = mean.shape[0] * (mean.shape[0] - 1) / 2
    data, mean = np.triu(data, k=1), np.triu(mean, k=1)
    q = np.sum(np.sum(np.square(data - mean)) / (len(data) - 1))
    return q, df


def add_to_csv(dict_values, filename):
    series = pd.Series(dict_values)
    if filename.exists():
        df = pd.read_csv(filename, index_col=0)
    else:
        df = pd.DataFrame(columns=list(dict_values.keys()))
    df = pd.concat([df, series.to_frame().T], ignore_index=True)
    df.to_csv(filename)


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
