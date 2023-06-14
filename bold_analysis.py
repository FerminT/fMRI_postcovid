from pathlib import Path
import argparse
import utils
from extract_components import extract_components_by_cluster
from build_connectome import build_connectome

TEMPLATE_SHAPE = [55, 65, 55]


def main(subject, conf_strategy, atlas_name, derivatives, n_components, threshold, clinical_file, output):
    derivatives, output = Path(derivatives), Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if subject == 'all':
        subjects = [sub for sub in derivatives.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [derivatives / f'sub-{subject.zfill(3)}']

    subjects_df = utils.load_clinical_data(clinical_file)
    utils.load_datapaths(subjects, subjects_df)

    if atlas_name:
        build_connectome(subjects_df, conf_strategy, atlas_name, output / atlas_name, threshold)
    if n_components:
        extract_components_by_cluster(subjects_df, conf_strategy, n_components, output / 'components')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--subject', type=str, default='all')
    arg_parser.add_argument('-c', '--confounds', type=str, default='simple',
                            help='Strategy for loading fMRIPrep denoising strategies. \
                           Options: simple, compcor, srubbing, ica_aroma')
    arg_parser.add_argument('-a', '--atlas', type=str, default='aal', help='Atlas to use for brain parcellation')
    arg_parser.add_argument('-d', '--derivatives', type=str, default='neurocovid_derivatives',
                            help='Path to BIDS derivatives folder')
    arg_parser.add_argument('-o', '--output', type=str, default='analysis/functional_connectivity')
    arg_parser.add_argument('-n', '--n_components', type=int, default=20,
                            help='Number of components to use for DictLearning')
    arg_parser.add_argument('-t', '--threshold', type=int, default=95,
                            help='Activity threshold for connectome (percentile)')
    arg_parser.add_argument('-clinical', '--clinical', type=str, default='clinical_data.csv',
                            help='Path to file with subjects clinical data')

    args = arg_parser.parse_args()
    main(args.subject, args.confounds, args.atlas, args.derivatives, args.n_components, args.threshold, args.clinical,
         args.output)
