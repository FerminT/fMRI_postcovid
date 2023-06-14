from pathlib import Path
import argparse
import utils
from extract_components import extract_components_by_cluster
from build_connectome import build_connectome

TEMPLATE_SHAPE = [55, 65, 55]


def main(subject, conf_strategy, atlas_name, n_components,
         threshold, low_pass, high_pass, smoothing_fwhm, t_r,
         data_path, clinical_file, output):
    data_path, output = Path(data_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if subject == 'all':
        subjects = [sub for sub in data_path.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [data_path / f'sub-{subject.zfill(3)}']

    subjects_df = utils.load_clinical_data(clinical_file)
    utils.load_datapaths(subjects, subjects_df)

    if atlas_name:
        build_connectome(subjects_df, conf_strategy, atlas_name,
                         threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                         output / atlas_name)
    if n_components:
        extract_components_by_cluster(subjects_df, conf_strategy, n_components,
                                      low_pass, high_pass, smoothing_fwhm, t_r,
                                      output / 'components')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--subjects', type=str, default='all')
    arg_parser.add_argument('-c', '--confounds_strategy', type=str, default='simple',
                            help='Strategy for loading fMRIPrep denoising strategies. \
                           Options: simple, compcor, srubbing, ica_aroma')
    arg_parser.add_argument('-a', '--atlas', type=str, default='aal',
                            help='(Discrete) atlas to use for brain parcellation')
    arg_parser.add_argument('-n', '--n_components', type=int, default=20,
                            help='Number of components to use for dictionary learning')
    arg_parser.add_argument('-t', '--threshold', type=int, default=95,
                            help='Activity threshold for connectome (percentile)')
    arg_parser.add_argument('-lp', '--low_pass', type=float, default=0.08,
                            help='Low pass filtering value for signal extraction')
    arg_parser.add_argument('-hp', '--high_pass', type=float, default=0.01,
                            help='High pass filtering value for signal extraction')
    arg_parser.add_argument('-fwhm', '--smoothing_fwhm', type=float, default=6.,
                            help='Kernel size for smoothing functional images')
    arg_parser.add_argument('-tr', '--repetition_time', type=float, default=2.,
                            help='Sampling rate of functional images (in seconds)')
    arg_parser.add_argument('-d', '--data_path', type=str, default='neurocovid_derivatives',
                            help='Path to BIDS derivatives folder')
    arg_parser.add_argument('-clinical', '--clinical_file', type=str, default='clinical_data.csv',
                            help='Path to file with subjects clinical data')
    arg_parser.add_argument('-o', '--output_path', type=str, default='analysis/functional_connectivity')

    args = arg_parser.parse_args()
    main(args.subject, args.confounds_strategy, args.atlas, args.n_components,
         args.threshold, args.low_pass, args.high_pass, args.smoothing_fwhm, args.tr,
         args.data_path, args.clinical_file, args.output_path)
