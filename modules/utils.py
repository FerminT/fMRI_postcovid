import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from filelock import FileLock
from . import plot

# NiLearn methods and classes
from nilearn import image
from nilearn.interfaces import fmriprep
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker


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


def add_to_csv(dict_values, filename):
    series = pd.Series(dict_values)
    update_file(series, filename, dict_values)


def add_to_df(group, threshold, group_metrics, group_filename):
    group_metrics = group_metrics.copy()
    group_metrics['group'] = group
    group_metrics['threshold'] = threshold
    series = pd.Series(group_metrics)
    update_file(series, group_filename, group_metrics)


def update_file(series, filename, dict_values):
    lock = FileLock(filename.parent / f'{filename.name}.lock')
    with lock:
        if filename.exists():
            if filename.name.endswith('.csv'):
                df = pd.read_csv(filename, index_col=0)
            else:
                df = pd.read_pickle(filename)
        else:
            df = pd.DataFrame(columns=list(dict_values.keys()))
        if series['threshold'] in df['threshold'].values:
            idx = df[(df['threshold'] == series['threshold']) & (df['group'] == series['group'])].index[0]
            for metric in dict_values:
                if metric not in df:
                    df[metric] = None
                elif isinstance(series[metric], list):
                    df[metric] = df[metric].astype(object)
                df.at[idx, metric] = series[metric]
        else:
            df = pd.concat([df, series.to_frame().T], ignore_index=True)
        save_file(df, filename)


def save_file(df, filename):
    if filename.name.endswith('.csv'):
        df.to_csv(filename)
    else:
        df.to_pickle(filename)


def add_statistical_significance(p_at_thresholds, ax, significance_levels, eps=1e-4):
    pvalues = p_at_thresholds[p_at_thresholds.columns[0]]
    labels = ['*' * i for i in range(len(significance_levels), 0, -1)]
    significance_levels.insert(0, 0.0)
    significance_levels.append(1.)
    labels.append('ns')
    categorized_pvalues = pd.cut(pvalues, significance_levels, right=False, labels=labels)
    spacing = pvalues.index[1] - pvalues.index[0] + eps

    plot.significance_bar(ax, categorized_pvalues, labels, spacing)


def networks_means(group, threshold, values, measure, filename):
    networks_values = list(values.values())
    means = np.mean(networks_values, axis=1)
    stes = np.std(networks_values, axis=1) / np.sqrt(np.shape(networks_values)[1])
    for idx, network in enumerate(values):
        network_path = filename.parents[1] / f'{filename.parent.name}_{network}'
        if network_path.exists():
            network_file = network_path / filename.name
            add_to_csv({'group': group, 'threshold': threshold, measure: means[idx],
                        f'{measure}_ste': stes[idx]}, network_file)


def compute_mean(group, threshold, group_measures, num_nodes, num_edges, filename):
    measures_mean = {'group': group, 'threshold': threshold}
    for measure in group_measures:
        values = group_measures[measure]
        if not len(values):
            continue
        if isinstance(values, dict):
            networks_means(group, threshold, values, measure, filename)
        else:
            measures_mean[measure] = np.mean(values)
            ste = np.std(values) / np.sqrt(len(values))
            if ste > 0:
                measures_mean[f'{measure}_ste'] = ste
    measures_mean['num_nodes'] = num_nodes
    measures_mean['num_edges'] = num_edges
    add_to_csv(measures_mean, filename)

    return measures_mean


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


def check_for_computed_metrics(group, threshold, filename):
    all_computed = False
    computed_thresholds = pd.read_csv(filename, index_col=0)
    if group in computed_thresholds['group'].unique():
        group_thresholds = computed_thresholds[computed_thresholds['group'] == group]
        if threshold in group_thresholds.threshold.values:
            all_computed = not group_thresholds[group_thresholds['threshold'] == threshold].isnull().values.any()
    return all_computed


def save_networks_pc(group, threshold, group_measures, filename, group_filename):
    for network in group_measures['avg_pc']:
        network_path = filename.parents[1] / f'{filename.parent.name}_{network}'
        values = group_measures['avg_pc'][network]
        if network_path.exists():
            network_file = network_path / group_filename.name
            add_to_df(group, threshold, {'avg_pc': values}, network_file)
