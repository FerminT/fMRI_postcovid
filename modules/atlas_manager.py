import numpy as np
import pandas as pd
import json
from sklearn.utils import Bunch
from nilearn import image, datasets
from .ic_manager import extract_components, extract_regions


def build_atlas(atlas_name, network_name, hemisphere, subjects_df, n_components, n_rois,
                low_pass, high_pass, smoothing_fwhm, t_r, conf_strategy):
    bold_imgs, mask_imgs = subjects_df['func_path'].values, subjects_df['mask_path'].values
    if atlas_name:
        atlas = load_atlas(atlas_name, n_rois)
        if network_name:
            atlas = extract_network(atlas, network_name, hemisphere)
    else:
        atlas = atlas_from_components(bold_imgs, mask_imgs, n_components, low_pass, high_pass, smoothing_fwhm,
                                      t_r, conf_strategy)

    return atlas


def get_schaefer_networks_indices(atlas_labels):
    all_atlas_labels = atlas_labels['name'].values
    networks_location, network_index = {}, 0
    for region in all_atlas_labels:
        network = region.split('_')[1]
        if network not in networks_location:
            networks_location[network] = {'index': network_index}
            network_index += 1

    return networks_location


def get_schaefer_networks_names(atlas_labels):
    networks_names = atlas_labels.name.str.split('_', expand=True)[1].unique()
    return networks_names


def get_network_name(atlas_basename, network):
    return network.strip(f'{atlas_basename}_') if is_network(network) else 'Global'


def get_network_img(atlas, network_indices):
    atlas_img = image.load_img(atlas.maps)
    atlas_affine, atlas_data = atlas_img.affine, atlas_img.get_fdata()
    atlas_data[~np.isin(atlas_data, network_indices)] = 0
    network_img = image.new_img_like(atlas_img, atlas_data, affine=atlas_affine, copy_header=True)

    return network_img


def load_networks_mapping(networks_mapping_file='brain_networks.json'):
    with open(networks_mapping_file, 'r') as f:
        networks_mapping = json.load(f)
    return networks_mapping


def extract_network(atlas, network_name, hemisphere):
    networks_mapping = load_networks_mapping()
    if 'msdl' in atlas.name:
        network_img, network_labels = extract_network_from_msdl(atlas, network_name)
    elif 'aal' in atlas.name:
        network_img, network_labels = extract_network_from_aal(atlas, network_name, networks_mapping)
    elif 'schaefer' in atlas.name:
        network_img, network_labels = extract_network_from_schaefer(atlas, network_name, hemisphere)
    else:
        raise ValueError(f'Can not extract networks from {atlas.name} atlas')

    atlas.name = f'{atlas.name}_{network_name}'
    if hemisphere != 'both':
        atlas.name += f'{hemisphere}'
    atlas.maps, atlas.labels = network_img, pd.DataFrame({'name': network_labels})
    return atlas


def extract_network_from_schaefer(atlas, network_name, hemisphere):
    # Add background region, since it is indexed as 0
    atlas.labels = pd.concat([pd.DataFrame({'name': ['_Background_']}), atlas.labels], ignore_index=True)
    network_indices = []
    for i, region in enumerate(atlas.labels.name):
        region_hemisphere, region_network = region.split('_')[0], region.split('_')[1]
        if region_network == network_name and (hemisphere == 'both' or hemisphere == region_hemisphere):
            network_indices.append(i)
    if len(network_indices) == 0:
        raise ValueError(f'Network {network_name} not in {atlas.name} atlas')

    network_labels = atlas.labels.iloc[network_indices].name.to_list()
    network_img = get_network_img(atlas, network_indices)

    return network_img, network_labels


def extract_network_from_aal(atlas, network_name, networks_mapping):
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


def atlas_from_components(bold_imgs, mask_imgs, n_components, low_pass, high_pass, smoothing_fwhm, t_r, conf_strategy):
    independent_components = extract_components(bold_imgs, mask_imgs, conf_strategy, n_components, low_pass, high_pass,
                                                smoothing_fwhm, t_r)
    regions = extract_regions(independent_components)
    n_regions = regions.regions_img_.shape[-1]
    atlas = Bunch(name='ica', maps=regions.maps_img, labels=pd.DataFrame({'name': [f'region_{idx + 1}'
                                                                          for idx in range(n_regions)]}))

    return atlas


def load_atlas(atlas_name, n_rois):
    if atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
        atlas.labels = pd.DataFrame({'name': atlas.labels})
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009(legacy_format=False)
        # Remove missing regions in atlas.maps from atlas.labels
        # (0 == 'background', 42 == 'L Medial_wall', 117 == 'R Medial_wall)
        atlas.labels = atlas.labels.drop([0, 42, 117]).reset_index(drop=True)
    elif atlas_name == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
        # Remove '7Networks_' prefix
        atlas.labels = pd.DataFrame({'name': [label[10:].decode() for label in atlas.labels]})
        atlas_name += f'{n_rois}'
    elif atlas_name == 'msdl':
        atlas = datasets.fetch_atlas_msdl()
        atlas.labels = pd.DataFrame({'name': atlas.labels})
    else:
        raise NotImplementedError(f'Atlas {atlas_name} not implemented')
    atlas.name = atlas_name

    return atlas


def is_probabilistic_atlas(atlas_maps):
    atlas_maps_img = image.load_img(atlas_maps)
    return len(atlas_maps_img.shape) == 4


def is_network(atlas_name):
    return len(atlas_name.split('_')) > 1
