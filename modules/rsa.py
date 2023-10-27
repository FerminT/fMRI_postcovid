import numpy as np
from rsatoolbox import rdm
from sklearn import mixture
from .utils import time_series
from .plot import plot_rdm
from .connectome_manager import connectivity_matrix


def rsa(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r,
        rdm_decomposition, similarity_measure, clinical_score, output):
    connectivity_embeddings = connectome_rsa(subjects_df, conf_strategy, atlas,
                                             low_pass, high_pass, smoothing_fwhm, t_r,
                                             rdm_decomposition, similarity_measure, clinical_score, output)
    behavioral_embeddings = behavioral_rsa(subjects_df, rdm_decomposition, output)

    return connectivity_embeddings, behavioral_embeddings


def connectome_rsa(subjects_df, conf_strategy, atlas, low_pass, high_pass, smoothing_fwhm, t_r,
                   rdm_decomposition, similarity_measure, clinical_score, output):
    timeseries = subjects_df.apply(lambda subj: time_series(subj['func_path'], subj['mask_path'],
                                                            conf_strategy, atlas.maps,
                                                            low_pass, high_pass, smoothing_fwhm,
                                                            t_r), axis=1)
    connectivity_matrices = np.stack(timeseries.apply(lambda ts: connectivity_matrix([ts])[0][0]))
    if similarity_measure == 'distance':
        connectivity_similarity_matrix = connectivity_correlation(connectivity_matrices, atlas.labels)
    else:
        connectivity_similarity_matrix = connectivity_distance(connectivity_matrices)
    subjects_df['cluster'] = clusters_rdm(connectivity_similarity_matrix)

    connectivity_embeddings = plot_rdm(connectivity_similarity_matrix, subjects_df, f'Connectivity {atlas.name}',
                                       output, rdm_decomposition, clinical_score)

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


def connectivity_correlation(connectivity_matrices, atlas_labels):
    # Follows Gratton et al. (2018), Neuron
    n_rois, n_subjects = len(atlas_labels.name), connectivity_matrices.shape[0]
    triu_inds = np.triu_indices(n_rois, 1)
    linearized_triu = np.empty((n_subjects, n_rois * (n_rois - 1) // 2))
    for i in range(n_subjects):
        subj_connectivity_matrix = connectivity_matrices[i]
        linearized_triu[i] = subj_connectivity_matrix[triu_inds]
    connectivity_similarity_matrix = np.corrcoef(linearized_triu)
    # Apply Fisher's z transformation
    all_but_diagonal = np.ones(connectivity_similarity_matrix.shape, dtype=bool)
    np.fill_diagonal(all_but_diagonal, False)
    connectivity_similarity_matrix = np.arctanh(connectivity_similarity_matrix, where=all_but_diagonal)

    return connectivity_similarity_matrix


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
