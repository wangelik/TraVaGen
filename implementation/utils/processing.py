# import table
import random as ra

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset

from .tools import import_variants, unpack_variants


class EventDataset(Dataset):
    """Class to transform XES event log into pytorch dataset

    Methods:
        __len__: standard torch dataset handler
        __getitem__: standard torch dataset handler
        raw: return list of variant names
        encode: return encoded list of variants
        decode: decode given list of encoded variants

    """

    def __init__(self, path: str, filename: str, shuffle: bool = True):
        """
        Args:
            path (str): path of event log file (excl. last "/")
            filename (str): name of event log file
            shuffle (bool): randomize list of variants or not

        """

        # import XES event log and extract variants
        self.raw_variants = import_variants(f"{path}/{filename}")

        # reduce raw variants to simple categorical list
        self.variants = unpack_variants(self.raw_variants)

        # shuffle variants
        if shuffle:
            ra.shuffle(self.variants)

        # fit One-Hot-Encoding
        self.encoder = LabelBinarizer()
        self.enc_variants = self.encoder.fit_transform(self.variants)
        self.dim = self.enc_variants.shape[1]

    def __len__(self) -> int:
        # return data size
        return len(self.enc_variants)

    def __getitem__(self, idx: int) -> np.ndarray:
        # return item at position idx
        return self.enc_variants[idx]

    def raw(self) -> list[str]:
        """Utility method to return list of variant names

        Returns:
            self.variants (list): list of variant names

        """

        return self.variants

    def encode(self) -> np.ndarray:
        """Utility method to return encoded variants

        Returns:
            self.enc_variants (np.ndarray): encoded variants

        """

        return self.enc_variants

    def decode(self, encoded_data: np.ndarray) -> list[str]:
        """Utility method to reverse variant encoding from new encoded data

        Args:
            encoded_data (np.ndarray): encoded variant matrix

        Returns:
            decoded_data (list): decoded list of variants

        """

        # inverse_transform (fitted) of encoded data
        decoded_data = self.encoder.inverse_transform(encoded_data).tolist()

        return decoded_data


def data_loader(dataset: EventDataset, **kwargs) -> DataLoader:
    """Dataloader wrapper to embed EventDataset into torch dataloader

    Args:
        dataset (EventDataset): variant dataset of type EventDataset
        **kwargs: optional torch dataloader arguments

    Returns:
        loader (DataLoader): final created torch dataloader

    """

    # create torch loader
    loader = DataLoader(dataset, **kwargs)

    return loader
