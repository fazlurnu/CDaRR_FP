'''CNS (Communication, Navigation, Surveillance) model — functional style.

The package is built up file by file. Exports are added as each module lands;
for now only the noise distributions are available.
'''
from .distributions import (
    CI95_TO_STD_2D,
    ci95_to_std,
    gaussian,
    make_biased_gaussian,
    tstudent,
)
from .sensor import SensorState, measure
from .reception_model import ReceptionModel, make_reception, ensure_size, set_pair, sample_mask, p_from_range
from .adsl_observation import ADSLObservation, empty_observation, update, field
from .adsl_observation import ensure_size as obs_ensure_size
from .cns import CNSState, make_cns, step, ownship_field, adsl_field

__all__ = [
    'CI95_TO_STD_2D',
    'ci95_to_std',
    'gaussian',
    'make_biased_gaussian',
    'tstudent',
    'SensorState',
    'measure',
    'ReceptionModel',
    'make_reception',
    'ensure_size',
    'set_pair',
    'sample_mask',
    'p_from_range',
    'ADSLObservation',
    'empty_observation',
    'update',
    'field',
    'obs_ensure_size',
    'CNSState',
    'make_cns',
    'step',
    'ownship_field',
    'adsl_field',
]
