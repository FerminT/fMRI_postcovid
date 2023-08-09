import numpy as np
from rsatoolbox import rdm
from sklearn import mixture
from .utils import time_series, plot_rdm
from .connectome_manager import connectivity_matrix


def rsa(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r, rdm_decomposition, output):
    connectivity_embeddings = connectome_rsa(subjects_df, conf_strategy, atlas,
                                             low_pass, high_pass, smoothing_fwhm, t_r, rdm_decomposition, output)
    behavioral_embeddings = behavioral_rsa(subjects_df, rdm_decomposition, output)

    return connectivity_embeddings, behavioral_embeddings


def connectome_rsa(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r,
                   rdm_decomposition, output):
    timeseries = subjects_df.apply(lambda subj: time_series(subj['func_path'], subj['mask_path'],
                                                            conf_strategy, atlas.maps,
                                                            low_pass, high_pass, smoothing_fwhm,
                                                            t_r), axis=1)
    connectivity_matrices = np.stack(timeseries.apply(lambda ts: connectivity_matrix([ts])[0][0]))
    connectivity_distance_matrix = connectivity_distance(connectivity_matrices)
    subjects_df['cluster'] = clusters_rdm(connectivity_distance_matrix)

    connectivity_embeddings = plot_rdm(connectivity_distance_matrix, subjects_df, f'Connectivity {atlas.name}',
                                       output, rdm_decomposition)

    return connectivity_embeddings


def behavioral_rsa(subjects_df, rdm_decomposition, output):
    fields = ['sexo', 'edad', 'composite_attention', 'composite_visuoespatial', 'composite_language',
              'composite_memory', 'composite_executive']
    if fields not in subjects_df.columns.to_list():
        behavioral_embeddings = np.zeros((subjects_df.shape[0], 2))
    else:
        behavioral_data = subjects_df[fields].copy()
        behavioral_data.dropna(inplace=True)
        behavioral_data['sexo'] = behavioral_data['sexo'].map({'Masculino': 0, 'Femenino': 1})
        behavioral_data /= behavioral_data.std()
        behavioral_distance = np.linalg.norm(behavioral_data.values[:, None] - behavioral_data.values[None, :], axis=2)
        behavioral_embeddings = plot_rdm(behavioral_distance, subjects_df.dropna(), 'Behavioral',
                                         output, rdm_decomposition)

    return behavioral_embeddings


def connectivity_distance(connectivity_matrices):
    connectivity_std = np.std(connectivity_matrices, axis=0)
    np.fill_diagonal(connectivity_std, 1)
    # Compute the distance between correlation matrices
    n_subjects = connectivity_matrices.shape[0]
    connectivity_distance_matrix = np.empty((n_subjects, n_subjects))
    for i in range(n_subjects):
        for j in range(n_subjects):
            connectivity_distance_matrix[i, j] = np.linalg.norm((connectivity_matrices[i, :] -
                                                                 connectivity_matrices[j, :]) / connectivity_std)

    return connectivity_distance_matrix


def clusters_rdm(connectivity_distance_matrix):
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    return gmm.fit_predict(connectivity_distance_matrix)


def get_rdm(distance_matrix, descriptor, pattern_descriptors):
    rdm_obj = rdm.RDMs(distance_matrix,
                       dissimilarity_measure='Euclidean',
                       rdm_descriptors={'name': [descriptor]},
                       pattern_descriptors={'subjects': pattern_descriptors})
    rdm_obj = rdm.transform(rdm_obj, lambda x: x / x.max())

    return rdm_obj
