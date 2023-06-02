import numpy as np
import pandas as pd
import argparse
import constants
from pathlib import Path

# NiLearn methods and classes
from nilearn import datasets, plotting, image
from nilearn.interfaces import fmriprep
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, MultiNiftiMasker

def load_clinical_data(clinical_datafile):
    cg = pd.read_csv(clinical_datafile)
    subjects_data = cg[~cg['AnonID'].isna()]

    # Remove invalid subjects (subj 29 has different data shapes)
    subjects_to_remove = [2, 17, 29]
    subjects_data = subjects_data.drop(subjects_data[subjects_data['AnonID'].isin(subjects_to_remove)].index)
    # Remove subjects with no cluster
    subjects_data = subjects_data[~subjects_data['cluster'].isna()]

    return subjects_data


def divide_by_cluster(subjects_df):
    # Get subjects per cluster
    clusters = subjects_df['cluster'].unique()
    subjs_per_cluster = {int(cluster): subjects_df[subjects_df['cluster'] == cluster]['AnonID'].values
                         for cluster in clusters}

    return subjs_per_cluster

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

