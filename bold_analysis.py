from pathlib import Path
import argparse
from modules import utils
from modules.atlas_manager import build_atlas
from modules.ic_manager import extract_group_components
from modules.connectome_manager import build_connectome
from modules.rsa import rsa


def main(subjects, conf_strategy, atlas_name, network_name, n_components, n_rois,
         threshold, low_pass, high_pass, smoothing_fwhm, t_r, rdm_decomposition, rdm_similarity,
         data_path, clinical_file, clinical_score, group_analysis, output):
    subjects_df = utils.load_subjects(subjects, data_path, clinical_file)
    atlas = build_atlas(atlas_name, network_name, subjects_df, n_components, n_rois, low_pass, high_pass,
                        smoothing_fwhm, t_r, conf_strategy)

    do_analysis(subjects_df, conf_strategy, atlas, n_components, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                rdm_decomposition, rdm_similarity, clinical_score, output, group_analysis)


def do_analysis(subjects_df, conf_strategy, atlas, n_components, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                rdm_decomposition, rdm_similarity, clinical_score, output, group_analysis):
    if group_analysis:
        build_connectome(subjects_df, conf_strategy, atlas, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                         output / atlas.name)
        if n_components:
            extract_group_components(subjects_df, conf_strategy, n_components,
                                     low_pass, high_pass, smoothing_fwhm, t_r,
                                     output / atlas.name / 'components')
    else:
        # Data-driven approach
        rsa(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r,
            rdm_decomposition, rdm_similarity, clinical_score, output / 'rsa')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-s', '--subjects', type=str, default='all')
    arg_parser.add_argument('-c', '--confounds_strategy', nargs='*', type=str,
                            default=['motion', 'high_pass', 'wm_csf'],
                            help='Strategy for loading fMRIPrep denoising strategies.')
    arg_parser.add_argument('-a', '--atlas', type=str, default='schaefer',
                            help='Atlas to use for brain parcellation')
    arg_parser.add_argument('-n', '--network', type=str, default=None,
                            help='Network to extract from atlas. If None, the whole atlas is used')
    arg_parser.add_argument('-nc', '--n_components', type=int, default=0,
                            help='Number of components to use for dictionary learning. \
                            If specified, the atlas is ignored')
    arg_parser.add_argument('-nr', '--n_rois', type=int, default=100,
                            help='Number of ROIs for Schaefer atlas. Otherwise ignored.')
    arg_parser.add_argument('-t', '--threshold', type=float, nargs='+', default=[0.1],
                            help='Connection density for connectome construction. If a list is provided, \
                            a connectome is built for each value')
    arg_parser.add_argument('-lp', '--low_pass', type=float, default=0.08,
                            help='Low pass filtering value for signal extraction')
    arg_parser.add_argument('-hp', '--high_pass', type=float, default=0.01,
                            help='High pass filtering value for signal extraction. \
                            Only considered if high_pass is not in confounds_strategy')
    arg_parser.add_argument('-fwhm', '--smoothing_fwhm', type=float, default=6.,
                            help='Kernel size for smoothing functional images')
    arg_parser.add_argument('-tr', '--repetition_time', type=float, default=2.,
                            help='Sampling rate of functional images (in seconds)')
    arg_parser.add_argument('-d', '--data_path', type=str, default='dataset',
                            help='Path to BIDS derivatives folder')
    arg_parser.add_argument('-clinical', '--clinical_file', type=str, default='clinical_data.csv',
                            help='Path to file with subjects clinical data')
    arg_parser.add_argument('-g', '--group_analysis', action='store_true',
                            help='Whether to perform group-based analysis')
    arg_parser.add_argument('-dc', '--decomposition', type=str, default='TSNE',
                            help='Decomposition to use for plotting relational distance matrix. \
                            Options are TSNE, MDS, ISOMAP, PCA')
    arg_parser.add_argument('-sm', '--similarity', type=str, default='distance',
                            help='Similarity measure for computing relational distance matrix. \
                            Options are distance, correlation')
    arg_parser.add_argument('-cs', '--clinical_score', type=str, default='global',
                            help='Clinical score to use when plotting relational distance matrix.')
    arg_parser.add_argument('-o', '--output_path', type=str, default='results')

    args = arg_parser.parse_args()
    if args.network and not args.atlas:
        raise ValueError('Network cannot be extracted without an atlas')
    if not (args.n_components or args.atlas):
        raise ValueError('Either atlas or n_components must be specified')
    args.atlas = args.atlas if not args.n_components else None
    data_path, output = Path(args.data_path), Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)

    high_pass = args.high_pass if 'high_pass' not in args.confounds_strategy else None
    main(args.subjects, args.confounds_strategy, args.atlas, args.network, args.n_components, args.n_rois,
         args.threshold, args.low_pass, high_pass, args.smoothing_fwhm, args.repetition_time, args.decomposition,
         args.similarity, data_path, args.clinical_file, args.clinical_score, args.group_analysis, output)
