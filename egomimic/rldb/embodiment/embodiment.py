from abc import ABC
from enum import Enum

from egomimic.rldb.zarr.action_chunk_transforms import Transform


class EMBODIMENT(Enum):
    EVE_RIGHT_ARM = 0
    EVE_LEFT_ARM = 1
    EVE_BIMANUAL = 2
    ARIA_RIGHT_ARM = 3
    ARIA_LEFT_ARM = 4
    ARIA_BIMANUAL = 5
    EVA_RIGHT_ARM = 6
    EVA_LEFT_ARM = 7
    EVA_BIMANUAL = 8
    MECKA_BIMANUAL = 9
    MECKA_RIGHT_ARM = 10
    MECKA_LEFT_ARM = 11
    SCALE_BIMANUAL = 12
    SCALE_RIGHT_ARM = 13
    SCALE_LEFT_ARM = 14


EMBODIMENT_ID_TO_KEY = {
    member.value: key for key, member in EMBODIMENT.__members__.items()
}


def get_embodiment(index):
    return EMBODIMENT_ID_TO_KEY.get(index, None)


def get_embodiment_id(embodiment_name):
    embodiment_name = embodiment_name.upper()
    return EMBODIMENT[embodiment_name].value


class Embodiment(ABC):
    """Base embodiment class. An embodiment is responsible for defining the transform pipeline that converts between the raw data in the dataset and the canonical representation used by the model."""

    @staticmethod
    def get_transform_list() -> list[Transform]:
        """Returns the list of transforms that convert between the raw data in the dataset and the canonical representation used by the model."""
        raise NotImplementedError

    @staticmethod
    def viz_transformed_batch(batch):
        """Visualizes a batch of transformed data."""
        raise NotImplementedError

    @staticmethod
    def get_keymap():
        """Returns a dictionary mapping from the raw keys in the dataset to the canonical keys used by the model."""
        raise NotImplementedError
