from pathlib import Path
import argparse
import utils
from extract_components import extract_components_by_cluster
from build_connectome import build_connectome
from rsa import rsa

TEMPLATE_SHAPE = [55, 65, 55]


def main(subjects, conf_strategy, atlas_name, network_name, n_components,
         threshold, low_pass, high_pass, smoothing_fwhm, t_r,
         data_path, clinical_file, use_clinical_cluster, output):
    data_path, output = Path(data_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if subjects == 'all':
        subjects = [sub for sub in data_path.glob('sub-*') if sub.is_dir()]
    else:
        subjects = [data_path / f'sub-{subjects.zfill(3)}']

    subjects_df = utils.load_clinical_data(clinical_file, use_clinical_cluster)
    utils.load_datapaths(subjects, subjects_df)

    bold_imgs, mask_imgs = subjects_df['func_path'].values, subjects_df['mask_path'].values
    if atlas_name:
        atlas = utils.load_atlas(atlas_name)
        if network_name:
            atlas = utils.extract_network(atlas, network_name)
    else:
        atlas = utils.atlas_from_regions(bold_imgs, mask_imgs, n_components, low_pass, high_pass, smoothing_fwhm,
                                         t_r, conf_strategy)

    if use_clinical_cluster:
        build_connectome(subjects_df, conf_strategy, atlas, threshold, low_pass, high_pass, smoothing_fwhm, t_r,
                         output / atlas.name)

        if n_components:
            extract_components_by_cluster(subjects_df, conf_strategy, n_components,
                                          low_pass, high_pass, smoothing_fwhm, t_r,
                                          output / 'components')
    else:
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
    arg_parser.add_argument('-ccluster', '--clinical_cluster', action='store_true',
                            help='Whether to use precomputed clinical cluster for analysis')
    arg_parser.add_argument('-o', '--output_path', type=str, default='analysis/functional_connectivity')

    args = arg_parser.parse_args()
    if args.network and not args.atlas:
        raise ValueError('Network cannot be extracted without an atlas')
    if not (args.n_components or args.atlas):
        raise ValueError('Either atlas or n_components must be specified')
    args.atlas = args.atlas if not args.n_components else None

    main(args.subjects, args.confounds_strategy, args.atlas, args.network, args.n_components,
         args.threshold, args.low_pass, args.high_pass, args.smoothing_fwhm, args.repetition_time,
         args.data_path, args.clinical_file, args.clinical_cluster, args.output_path)
