import numpy as np
import rsatoolbox
from utils import timeseries_from_regions
from extract_components import extract_components, extract_regions
from build_connectome import connectivity_matrix


def rsa(subjects_df, conf_strategy, n_components, atlas_name,
        low_pass, high_pass, smoothing_fwhm, t_r, output):
    behavioral_data = subjects_df[['edad', 'composite_attention', 'composite_visuoespatial', 'composite_language',
                                   'composite_memory', 'composite_executive']]
    behavioral_data /= behavioral_data.std()
    behavioral_distance = np.linalg.norm(behavioral_data.values[:, None] - behavioral_data.values[None, :], axis=2)

    subjects_components = extract_components(subjects_df['func_path'].values, subjects_df['mask_path'].values,
                                             conf_strategy, n_components, low_pass, high_pass, smoothing_fwhm, t_r)
    subjects_regions = extract_regions(subjects_components)
    subjects_timeseries = subjects_df.apply(lambda subj: timeseries_from_regions(subjects_regions,
                                                                                 subj['func_path'], conf_strategy),
                                            axis=1)
    subjects_corrmatrices = np.stack(subjects_timeseries.apply(lambda ts: connectivity_matrix([ts])[0][0]))
    subjects_connstd = np.std(subjects_corrmatrices, axis=0)
    np.fill_diagonal(subjects_connstd, 1)
    # Compute the distance between correlation matrices
    connectivity_distance = np.empty((subjects_corrmatrices.shape[0], subjects_corrmatrices.shape[0]))
    for i in range(subjects_corrmatrices.shape[0]):
        for j in range(subjects_corrmatrices.shape[0]):
            connectivity_distance[i, j] = np.linalg.norm((subjects_corrmatrices[i, :] -
                                                          subjects_corrmatrices[j, :]) / subjects_connstd)

    rdms = rsatoolbox.rdm.RDMs(np.concatenate((connectivity_distance[None, :], behavioral_distance[None, :])),
                               dissimilarity_measure='Euclidean',
                               rdm_descriptors={'name': ['Connectivity', 'Behavioral']},
                               pattern_descriptors={'subjects': subjects_df.index.to_list()})
    rdms = rsatoolbox.rdm.transform(rdms, lambda x: x / x.max())
    # # Check what the most common number of timepoints is
    # n_timepoints = subjects_timeseries.apply(lambda ts: ts.shape[0]).value_counts().index[0]
    # # Pad with nans for those that do not have that many timepoints
    # subjects_timeseries = subjects_timeseries.apply(lambda ts: np.pad(ts, ((0, n_timepoints - ts.shape[0]), (0, 0)),
    #                                                                   'constant', constant_values=np.nan))

    return
