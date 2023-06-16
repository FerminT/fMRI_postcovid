import numpy as np


def rsa(subjects_df, conf_strategy, n_components, atlas_name,
        low_pass, high_pass, smoothing_fwhm, t_r, output):
    behavioral_data = subjects_df[['edad', 'composite_attention', 'composite_visuoespatial', 'composite_language',
                                   'composite_memory', 'composite_executive']]
    behavioral_data /= behavioral_data.std()
    behavioral_matrix = np.linalg.norm(behavioral_data.values[:, None] - behavioral_data.values[None, :], axis=2)

    return

