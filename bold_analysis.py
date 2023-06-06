import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# NiLearn methods and classes
from nilearn import datasets, plotting, image
from nilearn.interfaces import fmriprep
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, MultiNiftiMasker

TEMPLATE_SHAPE = [55, 65, 55]


def compute_timeseries(subjects_df, conf_strategy, atlas_name):
    atlas = load_atlas(atlas_name)
    # Use apply_mask to all subjects in dataframe
    subjects_df['time_series'] = subjects_df.apply(lambda subj: apply_mask(subj['func_path'], subj['mask_path'],
                                                                           conf_strategy, atlas.maps), axis=1)


def apply_mask(func_data, brain_mask, conf_strategy, atlas_maps):
    nifti_masker = NiftiLabelsMasker(labels_img=atlas_maps,
                                     mask_img=brain_mask,
                                     smoothing_fwhm=6,
                                     low_pass=0.08,
                                     high_pass=0.01,
                                     t_r=2.,
                                     standardize=False,
                                     detrend=True,
                                     memory='nilearn_cache', memory_level=2)
    confounds, sample_mask = fmriprep.load_confounds_strategy(func_data, conf_strategy)
    time_series = nifti_masker.fit_transform(func_data, confounds=confounds, sample_mask=sample_mask)

    return time_series


def load_atlas(atlas_name):
    # Use nilearn datasets to fetch atlas
    if atlas_name == 'aal':
        atlas = datasets.fetch_atlas_aal()
    elif atlas_name == 'destrieux':
        atlas = datasets.fetch_atlas_destrieux_2009()
    elif atlas_name == 'schaefer':
        atlas = datasets.fetch_atlas_schaefer_2018()
    else:
        raise ValueError(f'Atlas {atlas_name} not recognized')

    return atlas


def load_clinical_data(clinical_datafile):
    cg = pd.read_csv(clinical_datafile)
    subjects_data = cg[~cg['AnonID'].isna()]

    # Remove invalid subjects (subj 29 has different data shapes)
    subjects_to_remove = [2, 17, 29]
    subjects_data = subjects_data.drop(subjects_data[subjects_data['AnonID'].isin(subjects_to_remove)].index)
    # Remove subjects with no cluster
    subjects_data = subjects_data[~subjects_data['cluster'].isna()]
    subjs_by_cluster = divide_by_cluster(subjects_data)

    return subjects_data


def load_datapaths(subjects_paths, subjects_df):
    for subj_path in subjects_paths:
        subj_id = int(subj_path.name.split('-')[1])
        if subj_id in subjects_df['AnonID'].values:
            # Get the path to the preprocessed functional data
            func_path = subj_path / 'func'
            func_file = [f for f in func_path.glob('*.nii.gz') if 'preproc' in f.name][0]
            mask_file = [f for f in func_path.glob('*.nii.gz') if 'brain_mask' in f.name][0]
            subjects_df.loc[subjects_df['AnonID'] == subj_id, 'func_path'] = str(func_file)
            subjects_df.loc[subjects_df['AnonID'] == subj_id, 'mask_path'] = str(mask_file)


def divide_by_cluster(subjects_df):
    # Get subjects per cluster
    clusters = subjects_df['cluster'].unique()
    subjs_by_cluster = {int(cluster): subjects_df[subjects_df['cluster'] == cluster]['AnonID'].values
                        for cluster in clusters}

    return subjs_by_cluster


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--subject', type=str, default='all')
    arg_parser.add_argument('-c', '--confounds', type=str, default='simple',
                            help='Strategy for loading fMRIPrep denoising strategies. \
                           Options: simple, compcor, srubbing, ica_aroma')
    arg_parser.add_argument('-m', '--mask', type=bool, default=True, help='Whether to use fMRIPrep brain mask or not')
    arg_parser.add_argument('-a', '--atlas', type=str, default='schaefer', help='Atlas to use for brain parcellation')
    arg_parser.add_argument('-d', '--derivatives', type=str, default='neurocovid_derivatives',
                            help='Path to BIDS derivatives folder')
    arg_parser.add_argument('-o', '--output', type=str, default='analysis/functional_connectivity')
    arg_parser.add_argument('-n', 'n_components', type=int, default=5,
                            help='Number of components to use for DictLearning')
    arg_parser.add_argument('-clinical', '--clinical', type=str, default='clinical_data.csv')

    args = arg_parser.parse_args()

    # Set up paths
    derivatives, output = Path(args.derivatives), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.subject == 'all':
        subjects = [sub for sub in derivatives.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [derivatives / f'sub-{args.subject.zfill(3)}']