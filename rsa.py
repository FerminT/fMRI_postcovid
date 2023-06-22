import numpy as np
import rsatoolbox
from utils import timeseries_from_regions, time_series, load_atlas
from extract_components import extract_components, extract_regions
from build_connectome import connectivity_matrix


def rsa(subjects_df, conf_strategy, n_components, atlas_name,
        low_pass, high_pass, smoothing_fwhm, t_r, output):
    n_subjects = len(subjects_df)
    behavioral_distance_matrix = behavioral_distance(subjects_df)

    if n_components:
        timeseries = ica_timeseries(subjects_df, conf_strategy, n_components, low_pass, high_pass, smoothing_fwhm, t_r)
    else:
        atlas = load_atlas(atlas_name)
        timeseries = subjects_df.apply(lambda subj: time_series(subj['func_path'], subj['mask_path'],
                                                                conf_strategy, atlas.maps,
                                                                low_pass, high_pass, smoothing_fwhm,
                                                                t_r), axis=1)
    connectivity_matrices = np.stack(timeseries.apply(lambda ts: connectivity_matrix([ts])[0][0]))
    connectivity_std = np.std(connectivity_matrices, axis=0)
    np.fill_diagonal(connectivity_std, 1)
    # Compute the distance between correlation matrices
    connectivity_distance_matrix = np.empty((n_subjects, n_subjects))
    for i in range(n_subjects):
        for j in range(n_subjects):
            connectivity_distance_matrix[i, j] = np.linalg.norm((connectivity_matrices[i, :] -
                                                                 connectivity_matrices[j, :]) / connectivity_std)

    distance_matrices = np.concatenate((connectivity_distance_matrix[None, :], behavioral_distance_matrix[None, :]))
    rdms = rsatoolbox.rdm.RDMs(distance_matrices,
                               dissimilarity_measure='Euclidean',
                               rdm_descriptors={'name': ['Connectivity', 'Behavioral']},
                               pattern_descriptors={'subjects': subjects_df.index.to_list()})
    rdms = rsatoolbox.rdm.transform(rdms, lambda x: x / x.max())
    return


def behavioral_distance(subjects_df, normalize=False):
    fields = ['edad', 'composite_attention', 'composite_visuoespatial', 'composite_language', 'composite_memory',
              'composite_executive']
    behavioral_data = subjects_df[fields]
    behavioral_data /= behavioral_data.std()
    behavioral_distance = np.linalg.norm(behavioral_data.values[:, None] - behavioral_data.values[None, :], axis=2)

    return behavioral_distance


def ica_timeseries(subjects_df, conf_strategy, n_components, low_pass, high_pass, smoothing_fwhm, t_r):
    independent_components = extract_components(subjects_df['func_path'].values, subjects_df['mask_path'].values,
                                                conf_strategy, n_components, low_pass, high_pass, smoothing_fwhm, t_r)
    regions = extract_regions(independent_components)
    timeseries = subjects_df.apply(lambda subj: timeseries_from_regions(regions, subj['func_path'], conf_strategy),
                                   axis=1)

    return timeseries
