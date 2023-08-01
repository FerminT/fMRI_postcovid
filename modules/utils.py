import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA

# NiLearn methods and classes
from nilearn import image, plotting
from nilearn.interfaces import fmriprep
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker


def plot_rdm(rdm, subjects_df, title, output, method='MDS', by_group=True):
    if method == 'MDS':
        embedding = MDS(n_components=2,
                        dissimilarity='precomputed',
                        random_state=42)
    elif method == 'Isomap':
        embedding = Isomap(n_components=2,
                           n_neighbors=5,
                           n_jobs=-1)
    elif method == 'PCA':
        embedding = PCA(n_components=2)
    else:
        raise NotImplementedError(f'Method {method} not implemented')

    title = title.replace(' ', '_')
    title += f'_{method}'
    coords = embedding.fit_transform(rdm)
    fig, ax = plt.subplots()
    groups = sorted(subjects_df['group'].unique())
    if by_group:
        for group, color in zip(groups, ['cyan', 'orange', 'black']):
            group_coords = coords[subjects_df['group'] == group]
            ax.scatter(group_coords[:, 0], group_coords[:, 1], color=color, label=f'{group}')
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1])
    for i, txt in enumerate(subjects_df.index.to_list()):
        ax.annotate(txt, (coords[i, 0], coords[i, 1]), alpha=0.6)

    ax.set_title(title)
    plt.show()

    output.mkdir(exist_ok=True, parents=True)
    fig.savefig(output / f'{title}.png')

    return coords


def plot_matrix_on_axis(connectivity_matrix, atlas_labels, ax, threshold,
                        tri='lower', vmin=-0.8, vmax=0.8, reorder=False):
    matrix_to_plot = connectivity_matrix.copy()
    matrix_to_plot = apply_threshold(matrix_to_plot, threshold)
    # Get labels in the correct format until plot_matrix is fixed
    labels = list(atlas_labels.name.values)
    plotting.plot_matrix(matrix_to_plot,
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
        # Probabilistic atlas
        nifti_masker = NiftiMapsMasker(maps_img=atlas_maps,
                                       **kwargs)
    else:
        nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                         **kwargs)
    confounds, sample_mask = fmriprep.load_confounds(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series


def is_probabilistic_atlas(atlas_maps):
    atlas_maps_img = image.load_img(atlas_maps)
    return len(atlas_maps_img.shape) == 4


def pad_timeseries(timeseries, pad_value=np.nan):
    n_timepoints = timeseries.apply(lambda ts: ts.shape[0]).value_counts().index[0]
    timeseries = timeseries.apply(lambda ts: np.pad(ts, ((0, n_timepoints - ts.shape[0]), (0, 0)),
                                                    'constant', constant_values=pad_value))

    return timeseries


def load_subjects(subjects, data_path, clinical_file, group_analysis):
    if subjects == 'all':
        subjects = [sub for sub in data_path.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [data_path / f'sub-{subjects.zfill(3)}']

    subjects_df = load_clinical_data(clinical_file, group_analysis)
    subjects_df = load_datapaths(subjects, subjects_df)

    return subjects_df


def load_clinical_data(clinical_datafile, group_analysis):
    cg = pd.read_csv(clinical_datafile, na_filter=False)
    subjects_data = cg[~cg['id'].isna()]
    subjects_data = subjects_data.astype({'id': int})
    subjects_data = subjects_data.set_index('id')
    if ['whodas_total', 'fss_63', 'hads_ansiedad', 'hads_depresion'] in subjects_data.columns.to_list():
        subjects_data = subjects_data.drop(['whodas_total', 'fss_63', 'hads_ansiedad', 'hads_depresion'], axis=1)

    if group_analysis:
        subjects_data = subjects_data[~(subjects_data['group'] == 'NA')]

    return subjects_data


def load_datapaths(subjects_paths, subjects_df):
    for subj_path in subjects_paths:
        subj_id = int(subj_path.name.split('-')[1])
        if subj_id in subjects_df.index:
            # Get the path to the preprocessed functional data
            func_path = subj_path / 'func'
            func_file = [f for f in func_path.glob('*.nii.gz') if 'preproc' in f.name][0]
            mask_file = [f for f in func_path.glob('*.nii.gz') if 'brain_mask' in f.name][0]
            subjects_df.loc[subj_id, 'func_path'] = str(func_file)
            subjects_df.loc[subj_id, 'mask_path'] = str(mask_file)

    return subjects_df


def apply_threshold(connectivity_matrix, threshold):
    percentile = np.percentile(connectivity_matrix, threshold)
    connectivity_matrix[connectivity_matrix < percentile] = 0
    return connectivity_matrix


def q_test(data, mean):
    # Upper triangulate the data
    df = data.shape[0] * (data.shape[0] - 1) / 2
    data, mean = np.triu(data, k=1), np.triu(mean, k=1)
    q = np.sum(np.sum(np.square(data - mean)) / (len(data) - 1))
    return q, df


def print_connectivity_metrics(connectivity_matrix, threshold):
    connectivity_matrix[connectivity_matrix < (threshold / 100)] = 0
    np.fill_diagonal(connectivity_matrix, 0)
    graph = nx.from_numpy_array(connectivity_matrix)
    print(f'Average clustering coefficient: {nx.average_clustering(graph)}')
    print(f'Average node connectivity: {nx.average_node_connectivity(graph)}')
    print(f'Average neighbor degree: {np.mean(list(nx.average_neighbor_degree(graph).values()))}')
