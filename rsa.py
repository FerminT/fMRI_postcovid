import numpy as np
from utils import timeseries_from_regions
from extract_components import extract_components, extract_regions


def rsa(subjects_df, conf_strategy, n_components, atlas_name,
        low_pass, high_pass, smoothing_fwhm, t_r, output):
    behavioral_data = subjects_df[['edad', 'composite_attention', 'composite_visuoespatial', 'composite_language',
                                   'composite_memory', 'composite_executive']]
    behavioral_data /= behavioral_data.std()
    behavioral_matrix = np.linalg.norm(behavioral_data.values[:, None] - behavioral_data.values[None, :], axis=2)

    subjects_components = extract_components(subjects_df['func_path'].values, subjects_df['mask_path'].values,
                                             conf_strategy, n_components, low_pass, high_pass, smoothing_fwhm, t_r)
    subjects_regions = extract_regions(subjects_components)
    subjects_timeseries = subjects_df.apply(lambda subj: timeseries_from_regions(subjects_regions,
                                                                                 subj['func_path'], conf_strategy),
                                            axis=1)
    # Check what the most common number of timepoints is
    n_timepoints = subjects_timeseries.apply(lambda ts: ts.shape[0]).value_counts().index[0]
    # Pad with nans for those that do not have that many timepoints
    subjects_timeseries = subjects_timeseries.apply(lambda ts: np.pad(ts, ((0, n_timepoints - ts.shape[0]), (0, 0)),
                                                                      'constant', constant_values=np.nan))
    # Get the standard deviation for each of the 67 regions (should be 1)
    subjects_timeseries_std = subjects_timeseries.apply(lambda ts: np.nanstd(ts, axis=0))
    subjects_timeseries /= subjects_timeseries_std
    # Hay 205 valores para cada región/componente, pensar cómo reducirlo a un número
    # timeseries_matrix = np.linalg.norm(subjects_timeseries.values[:, None] - subjects_timeseries.values[None, :], axis=1)

    return
