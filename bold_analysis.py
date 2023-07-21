from pathlib import Path
import argparse
import utils
from atlas_manager import build_atlas
from extract_components import extract_group_components
from build_connectome import build_connectome
from rsa import rsa


def main(subjects, conf_strategy, atlas_name, network_name, n_components,
         threshold, low_pass, high_pass, smoothing_fwhm, t_r,
         data_path, clinical_file, group_analysis, output):
    subjects_df = utils.load_subjects(subjects, data_path, clinical_file, group_analysis)
    atlas = build_atlas(atlas_name, network_name, subjects_df, n_components, low_pass, high_pass, smoothing_fwhm,
                              t_r, conf_strategy)

    do_analysis(subjects_df, conf_strategy, atlas, n_components, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                output, group_analysis)


def do_analysis(subjects_df, conf_strategy, atlas, n_components, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                output, group_analysis):
    if group_analysis:
        build_connectome(subjects_df, conf_strategy, atlas, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                         output / atlas.name)
        if n_components:
            extract_group_components(subjects_df, conf_strategy, n_components,
                                     low_pass, high_pass, smoothing_fwhm, t_r,
                                     output / 'components')
    else:
        # Data-driven approach
        rsa(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r, output / 'rsa')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--subjects', type=str, default='all')
    arg_parser.add_argument('-c', '--confounds_strategy', type=str, default='simple',
                            help='Strategy for loading fMRIPrep denoising strategies. \
                           Options: simple, compcor, srubbing, ica_aroma')
    arg_parser.add_argument('-a', '--atlas', type=str, default='aal',
                            help='Atlas to use for brain parcellation')
    arg_parser.add_argument('-n', '--network', type=str, default=None,
                            help='Network to extract from atlas. If None, the whole atlas is used')
    arg_parser.add_argument('-nc', '--n_components', type=int, default=0,
                            help='Number of components to use for dictionary learning. \
                            If specified, the atlas is ignored')
    arg_parser.add_argument('-t', '--threshold', type=int, default=30,
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
    arg_parser.add_argument('-g', '--group_analysis', action='store_true',
                            help='Whether to perform group-based analysis')
    arg_parser.add_argument('-o', '--output_path', type=str, default='analysis/functional_connectivity')

    args = arg_parser.parse_args()
    if args.network and not args.atlas:
        raise ValueError('Network cannot be extracted without an atlas')
    if not (args.n_components or args.atlas):
        raise ValueError('Either atlas or n_components must be specified')
    args.atlas = args.atlas if not args.n_components else None
    data_path, output = Path(args.data_path), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    main(args.subjects, args.confounds_strategy, args.atlas, args.network, args.n_components,
         args.threshold, args.low_pass, args.high_pass, args.smoothing_fwhm, args.repetition_time,
         data_path, args.clinical_file, args.group_analysis, output)
