from torch.utils.data import DataLoader, random_split
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning import LightningDataModule
from egomimic.utils.egomimicUtils import nds
import json
import os
from egomimic.rldb.utils import RLDBDataset
from termcolor import cprint

class RLDBModule(LightningDataModule):
    """
    Deprecated and is not supported by trainHydra.py
    """
    def __init__(self, train_dataset, valid_dataset, train_dataloader_kwargs, valid_dataloader_kwargs):
        cprint("RLDBModule is deprecated and is not supported by trainHydra.py. Use MultiDataModuleWrapper instead", "red")

        super().__init__()
        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.valid_dataloader_kwargs = valid_dataloader_kwargs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.train_dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.valid_dataloader_kwargs)

class MultiDataModuleWrapper(LightningDataModule):
    """
    New functionality for dictionary based multi embodiment loading using CombinedLoader.

    Uses hydra to instantiate DataLoader objects and then wraps them in a combined loader
    """
    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params : dict,
        valid_dataloader_params : dict,
    ):

        super().__init__()
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name, {})
            iterables[dataset.embodiment] = DataLoader(dataset, shuffle=True, **dataset_params)
        
        return CombinedLoader(iterables, 'max_size_cycle')
    
    def val_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.valid_datasets.items():
            dataset_params = self.valid_dataloader_params.get(dataset_name, {})
            iterables[dataset.embodiment] = DataLoader(dataset, shuffle=False, **dataset_params)
        
        return CombinedLoader(iterables, 'max_size_cycle')

class DualDataModuleWrapper(LightningDataModule):
    """
    Same as DataModuleWrapper but there are two train datasets and two valid datasets
    """

    """
    Deprecated and is not supported by trainHydra.py
    """

    def __init__(
        self,
        train_dataset1,
        valid_dataset1,
        train_dataset2,
        valid_dataset2,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        cprint("DualDataModuleWrapper is deprecated and is not supported by trainHydra.py. Use MultiDataModuleWrapper instead", "red")

        super().__init__()
        self.train_dataset1 = train_dataset1
        self.valid_dataset1 = valid_dataset1
        self.train_dataset2 = train_dataset2
        self.valid_dataset2 = valid_dataset2
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.train_dataset1, **self.train_dataloader_params
        )
        new_dataloader2 = DataLoader(
            dataset=self.train_dataset2, **self.train_dataloader_params
        )
        return [new_dataloader1, new_dataloader2]
    
    ## to change embodiment sampling freq, just change the batch_size
    def val_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.valid_dataset1, shuffle=False, **self.valid_dataloader_params
        )
        new_dataloader2 = DataLoader(
            dataset=self.valid_dataset2, shuffle=False, **self.train_dataloader_params
        )
        return [new_dataloader1, new_dataloader2]

    # def val_dataloader(self):
    #     new_dataloader1 = DataLoader(dataset=self.valid_dataset1, **self.valid_dataloader_params)
    #     new_dataloader2 = DataLoader(dataset=self.valid_dataset2, **self.valid_dataloader_params)
    #     return [new_dataloader1, new_dataloader2]


class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader = DataLoader(
            dataset=self.train_dataset, **self.train_dataloader_params
        )
        return new_dataloader

    def val_dataloader_1(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset, **self.valid_dataloader_params
        )
        return new_dataloader