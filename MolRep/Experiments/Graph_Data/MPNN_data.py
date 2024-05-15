# -*- coding: utf-8 -*-
"""
Created on 2020.05.19

@author: Jiahua Rao, Weiming Li, Hui Yang, Jiancong Xie

Code based on:
Yang et al "Analyzing Learned Molecular Representations for Property Prediction" & "A Deep Learning Approach to Antibiotic Discovery" -> https://github.com/chemprop/chemprop
Song et al "Communicative Representation Learning on Attributed Molecular Graphs" -> https://github.com/SY575/CMPNN
"""

import os
import logging
import pickle
import random
import threading

import numpy as np
import pandas as pd

from random import Random
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, Callable, List, Union, Optional, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from rdkit import Chem

from MolRep.Featurization.MPNN_embeddings import BatchMolGraph, MolGraph
from MolRep.Featurization.MPNN_embeddings import get_features_generator

from MolRep.Models.scalers import StandardScaler
from MolRep.Utils.utils import worker_init


# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: str,
                 targets: List[Optional[float]] = None,
                 features: np.ndarray = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None):
        """
        :param smiles: The SMILES string for the molecule.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param row: The raw CSV row containing the information for this molecule.
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param atom_features: A list of atom features to use.
        :param atom_descriptors: A list of atom descriptorss to use.
        """

        self.smiles = smiles
        self.targets = targets
        self.features = features
        self.atom_descriptors = atom_descriptors
        self.atom_features = atom_features

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom_descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)

        # Fix nans in atom_features
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets = self.features, self.targets

    @property
    def mol(self) -> Chem.Mol:
        """Gets the corresponding RDKit molecule for this molecule's SMILES."""
        mol = SMILES_TO_MOL.get(self.smiles, Chem.MolFromSmiles(self.smiles))

        if cache_mol():
            SMILES_TO_MOL[self.smiles] = mol

        return mol

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.
        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.
        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()

    def smiles(self) -> List[str]:
        """
        Returns a list containing the SMILES associated with each molecule.
        :return: A list of SMILES strings.
        """
        return [d.smiles for d in self._data]

    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with each molecule.
        :return: A list of RDKit molecules.
        """
        return [d.mol for d in self._data]

    def batch_graph(self) -> BatchMolGraph:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.
        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.
        :return: A :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the molecules.
        """
        if self._batch_graph is None:
            mol_graphs = []
            for d in self._data:
                if d.smiles in SMILES_TO_GRAPH:
                    mol_graph = SMILES_TO_GRAPH[d.smiles]
                else:
                    mol_graph = MolGraph(d.mol, d.atom_features)
                    if cache_graph():
                        SMILES_TO_GRAPH[d.smiles] = mol_graph
                mol_graphs.append(mol_graph)

            self._batch_graph = BatchMolGraph(mol_graphs)

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).
        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).
        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.
        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.
        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.
        :return: The size of the additional atom descriptor vector.
        """
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.
        :return: The size of the additional atom feature vector.
        """
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.
        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.
        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        if scaler is not None:
            self._scaler = scaler

        elif self._scaler is None:
            features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)

        for d in self._data:
            d.set_features(self._scaler.transform(d.raw_features.reshape(1, -1))[0])

        return self._scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.
        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).
        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.
        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.
    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.
    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    smiles_batch, features_batch, target_batch, atom_descriptors_batch = data.batch_graph(), data.features(), data.targets(), data.atom_descriptors()

    return [smiles_batch, features_batch, target_batch, atom_descriptors_batch]


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 2,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )
        
        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout,
            worker_init_fn=worker_init,
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.
        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def smiles(self):
        return [self._dataset[index].smiles for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()



def _construct_dataset(smiles_all, x_all, y_all):
    """Construct a MolDataset object from the provided data.
    Args:
        x_all (list): A list of molecule features.
        y_all (list): A list of the corresponding labels.
    Returns:
        A MolDataset object filled with the provided data.
    """
    dataset = MoleculeDataset([
                MoleculeDatapoint(
                    smiles=smiles,
                    targets=targets,
                    features=x_all[i][0] if x_all[i][0] is not None else None,
                    atom_features=x_all[i][1] if x_all[i][1] is not None else None,
                    atom_descriptors=x_all[i][2] if x_all[i][2] is not None else None,
                ) for i, (smiles, targets) in enumerate(zip(smiles_all, y_all))
    ])

    return dataset

def _construct_dataloader(data_set, batch_size, shuffle=True, num_workers=0, seed=0, class_balance=False):
    """Construct a data loader for the provided data.
    Args:
        data_set (): 
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.
    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    if data_set is not None:
        loader = MoleculeDataLoader(
                    dataset=data_set,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    class_balance=class_balance,
                    shuffle=shuffle,
                    seed=seed,
                )
    else:
        loader = None
    return loader

def MPNN_construct_dataset(features_path, train_idxs=None, valid_idxs=None, test_idxs=None):
    # smiles_all, x_all, y_all = pickle.load(open(features_path, "rb"))
    dataset = torch.load(features_path)
    smiles_all, x_all, y_all = dataset["smiles_all"], dataset["x_all"], dataset["y_all"]

    trainset = _construct_dataset(np.array(smiles_all)[train_idxs], np.array(x_all)[train_idxs], np.array(y_all)[train_idxs]) if train_idxs is not None else None
    validset = _construct_dataset(np.array(smiles_all)[valid_idxs], np.array(x_all)[valid_idxs], np.array(y_all)[valid_idxs]) if valid_idxs is not None else None
    testset = _construct_dataset(np.array(smiles_all)[test_idxs], np.array(x_all)[test_idxs], np.array(y_all)[test_idxs]) if test_idxs is not None else None
    return trainset, validset, testset

def MPNN_construct_dataloader(trainset=None, validset=None, testset=None, batch_size=1, shuffle=True, task_type='Classification', seed=0, features_scaling=True):

    if features_scaling and trainset is not None:
        features_scaler = trainset.normalize_features(replace_nan_token=0)
        if validset is not None:
            validset.normalize_features(features_scaler)
        if testset is not None:
            testset.normalize_features(features_scaler)
    else:
        features_scaler = None

    if task_type == 'Regression' and trainset is not None:
        train_targets = trainset.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        trainset.set_targets(scaled_targets)
    else:
        scaler = None

    return _construct_dataloader(trainset, batch_size, shuffle=shuffle, seed=seed), \
           _construct_dataloader(validset, batch_size, shuffle=False, seed=seed), \
           _construct_dataloader(testset, batch_size, shuffle=False, seed=seed), \
           features_scaler, scaler

