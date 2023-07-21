import json

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
from extract_components import extract_components, extract_regions

# NiLearn methods and classes
from nilearn import datasets, image
from nilearn.interfaces import fmriprep
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.regions import RegionExtractor


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
    groups = subjects_df['group'].unique()
    if by_group:
        for group, color in zip(groups, ['cyan', 'orange', 'black']):
            group_coords = coords[subjects_df['group'] == group] \
                if not np.isnan(group) else coords[subjects_df['group'].isna()]
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


def time_series(func_data, brain_mask, conf_strategy, atlas_maps, low_pass, high_pass, smoothing_fwhm, t_r):
    kwargs = {'mask_img': brain_mask, 'smoothing_fwhm': smoothing_fwhm, 'low_pass': low_pass, 'high_pass': high_pass,
              't_r': t_r, 'standardize': False, 'detrend': True, 'memory': 'nilearn_cache', 'memory_level': 2}
    if type(atlas_maps) == RegionExtractor:
        nifti_masker = atlas_maps
    else:
        atlas_maps_img = image.load_img(atlas_maps)
        if len(atlas_maps_img.shape) == 4:
            # Probabilistic atlas
            nifti_masker = NiftiMapsMasker(maps_img=atlas_maps,
                                           **kwargs)
        else:
            nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                             **kwargs)
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series


def timeseries_from_regions(regions_extractor, func_data, conf_strategy):
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    return regions_extractor.transform(func_data, confounds=confounds, sample_mask=sample_mask)


def pad_timeseries(timeseries, pad_value=np.nan):
    n_timepoints = timeseries.apply(lambda ts: ts.shape[0]).value_counts().index[0]
    timeseries = timeseries.apply(lambda ts: np.pad(ts, ((0, n_timepoints - ts.shape[0]), (0, 0)),
                                                    'constant', constant_values=pad_value))

    return timeseries


def get_schaefer_networks_indices(atlas_labels):
    all_atlas_labels = atlas_labels['name'].values
    networks_location, network_index = {}, 0
    for region in all_atlas_labels:
        network = region.split('_')[1]
        if network not in networks_location:
            networks_location[network] = {'index': network_index}
            network_index += 1

    return networks_location


def get_network_img(atlas, network_indices):
    atlas_img = image.load_img(atlas.maps)
    atlas_affine, atlas_data = atlas_img.affine, atlas_img.get_fdata()
    atlas_data[~np.isin(atlas_data, network_indices)] = 0
    network_img = image.new_img_like(atlas_img, atlas_data, affine=atlas_affine, copy_header=True)

    return network_img


def extract_network(atlas, network_name):
    if atlas.name == 'msdl':
        network_img, network_labels = extract_network_from_msdl(atlas, network_name)
    elif atlas.name == 'aal':
        network_img, network_labels = extract_network_from_aal(atlas, network_name)
    elif atlas.name == 'schaefer':
        network_img, network_labels = extract_network_from_schaefer(atlas, network_name)
    else:
        raise ValueError(f'Can not extract networks from {atlas.name} atlas')

    atlas.name = f'{atlas.name}_{network_name}'
    atlas.maps, atlas.labels = network_img, pd.DataFrame({'name': network_labels})
    return atlas


def extract_network_from_schaefer(atlas, network_name):
    # Add background region, since it is indexed as 0
    atlas.labels = pd.concat([pd.DataFrame({'name': ['_Background_']}), atlas.labels], ignore_index=True)
    network_indices = []
    for i, region in enumerate(atlas.labels.name):
        network = region.split('_')[1]
        if network == network_name:
            network_indices.append(i)
    if len(network_indices) == 0:
        raise ValueError(f'Network {network_name} not in {atlas.name} atlas')

    network_labels = atlas.labels.iloc[network_indices].name.to_list()
    network_img = get_network_img(atlas, network_indices)

    return network_img, network_labels


def extract_network_from_aal(atlas, network_name):
    networks_mapping = load_networks_mapping()
    if network_name not in networks_mapping or atlas.name not in networks_mapping[network_name]:
        raise ValueError(f'Network {network_name} not in {atlas.name} atlas')

    network_labels = networks_mapping[network_name][atlas.name]
    network_indices = atlas.labels[atlas.labels.name.isin(network_labels)].index
    network_img_indices = [int(atlas.indices[idx]) for idx in network_indices]
    network_img = get_network_img(atlas, network_img_indices)

    return network_img, network_labels


def extract_network_from_msdl(atlas, network_name):
    if network_name not in atlas.networks:
        raise ValueError(f'Network {network_name} not in {atlas.name} atlas')

    network_indices = [idx for idx, network in enumerate(atlas.networks) if network == network_name]
    network_labels = [atlas.labels.name.iloc[idx] for idx in network_indices]
    network_img = image.index_img(atlas.maps, network_indices)

    return network_img, network_labels


def atlas_from_regions(bold_imgs, mask_imgs, n_components, low_pass, high_pass, smoothing_fwhm, t_r, conf_strategy):
    independent_components = extract_components(bold_imgs, mask_imgs, conf_strategy, n_components, low_pass, high_pass,
                                                smoothing_fwhm, t_r)
    regions = extract_regions(independent_components)
    n_regions = regions.regions_img_.shape[-1]
    atlas = Bunch(name='ica', maps=regions, labels=pd.DataFrame({'name': [f'region_{idx + 1}'
                                                                          for idx in range(n_regions)]}))

    return atlas


def load_atlas(atlas_name):
    if atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
        atlas.labels = pd.DataFrame({'name': atlas.labels})
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
        # Remove missing regions in atlas.maps from atlas.labels
        # (0 == 'background', 42 == 'L Medial_wall', 117 == 'R Medial_wall)
        atlas.labels = atlas.labels.drop([0, 42, 117]).reset_index(drop=True)
    elif atlas_name == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
        # Remove '7Networks_' prefix
        atlas.labels = pd.DataFrame({'name': [label[10:].decode() for label in atlas.labels]})
    elif atlas_name == 'msdl':
        atlas = datasets.fetch_atlas_msdl()
        atlas.labels = pd.DataFrame({'name': atlas.labels})
    else:
        raise NotImplementedError(f'Atlas {atlas_name} not implemented')
    atlas.name = atlas_name

    return atlas


def load_clinical_data(clinical_datafile, group_analysis):
    cg = pd.read_csv(clinical_datafile)
    subjects_data = cg[~cg['id'].isna()]
    subjects_data = subjects_data.astype({'id': int})
    subjects_data = subjects_data.set_index('id')
    if ['whodas_total', 'fss_63', 'hads_ansiedad', 'hads_depresion'] in subjects_data.columns:
        subjects_data = subjects_data.drop(['whodas_total', 'fss_63', 'hads_ansiedad', 'hads_depresion'], axis=1)

    if group_analysis:
        subjects_data = subjects_data[~subjects_data['group'].isna()]

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


def apply_threshold(connectivity_matrix, threshold):
    percentile = np.percentile(connectivity_matrix, threshold)
    connectivity_matrix[connectivity_matrix < percentile] = 0
    return connectivity_matrix


def q_test(data, mean):
    # Upper triangulate the data
    data, mean = np.triu(data, k=1), np.triu(mean, k=1)
    q = np.sum(np.sum(np.square(data - mean)) / (len(data) - 1))
    return q


def print_connectivity_metrics(connectivity_matrix, threshold):
    connectivity_matrix[connectivity_matrix < (threshold / 100)] = 0
    np.fill_diagonal(connectivity_matrix, 0)
    graph = nx.from_numpy_array(connectivity_matrix)
    print(f'Average clustering coefficient: {nx.average_clustering(graph)}')
    print(f'Average node connectivity: {nx.average_node_connectivity(graph)}')
    print(f'Average neighbor degree: {np.mean(list(nx.average_neighbor_degree(graph).values()))}')


def load_networks_mapping(networks_mapping_file='brain_networks.json'):
    with open(networks_mapping_file, 'r') as f:
        networks_mapping = json.load(f)
    return networks_mapping
