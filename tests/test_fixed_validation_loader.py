import torch
from torch.utils.data import Dataset

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper


class _TinyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}


def test_max_size_validation_visits_each_domain_once_without_cycling():
    module = MultiDataModuleWrapper(
        train_datasets={},
        valid_datasets={"short": _TinyDataset(3), "long": _TinyDataset(5)},
        train_dataloader_params={},
        valid_dataloader_params={
            "short": {"batch_size": 2, "num_workers": 0},
            "long": {"batch_size": 2, "num_workers": 0},
        },
        valid_combined_mode="max_size",
    )

    rows = list(module.val_dataloader())

    assert len(rows) == 3
    assert rows[-1][0]["short"] is None
    assert rows[-1][0]["long"]["value"].tolist() == [4]
