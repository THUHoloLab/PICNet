import os
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class QPM_Dataset(Dataset):
    """
    A unified dataset for loading related .mat files.

    Expected directory structure:
        root/
            <image_set>/
                holo/   # when data_type contains "holo"
                    xxx.mat
                pha/    # when data_type contains "pha"
                    xxx.mat
                ...

    Each .mat file is expected to contain:
        - if data_type=["holo"]: keys "holo" and "label_z"
        - if data_type=["pha"]:  key "pha"

    Parameters
    ----------
    root : str
        Root path of the dataset.
    data_type : Sequence[str]
        What to load, e.g. ["holo"] or ["pha"].
    image_set : str
        Which split to use, e.g. "traindata" or "testdata".
    transform : callable, optional
        Transform applied to loaded numpy arrays (typically ToTensor()).
    ratio : float, optional
        If given, will randomly select this ratio of samples from the split,
    """
    def __init__(
        self,
        root: str,
        data_type: Sequence[str],
        image_set: str,
        transform: Optional[callable] = None,
        ratio: Optional[float] = None,
    ) -> None:
        self.root = root
        self.data_type = list(data_type)
        self.image_set = image_set
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        self.dir_paths: List[str] = []
        file_lists: List[List[str]] = []

        for t in self.data_type:
            dir_path = os.path.join(root, image_set, t)
            self.dir_paths.append(dir_path)

            # filter out non-mat files if needed
            files = [f for f in os.listdir(dir_path) if f.endswith(".mat")]
            files.sort()  # keep deterministic order
            file_lists.append(files)

        self.file_lists = np.array(file_lists)

        # optional: subsample a ratio of the data
        if ratio is not None:
            num_total = len(self.file_lists[0])
            indices = np.arange(num_total)
            np.random.shuffle(indices)
            keep = indices[: int(num_total * ratio)]
            self.file_lists = self.file_lists[:, keep]

        self.num_samples = len(self.file_lists[0])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        # loading intensity
        if "holo" in self.data_type:
            path = os.path.join(self.dir_paths[0], self.file_lists[0][index])
            mat_data = self._load_mat(path)
            holo = mat_data["holo"]
            label_z = mat_data["label_z"]

            if self.transform is not None:
                holo = self.transform(holo)
                label_z = self.transform(label_z)

            return holo, label_z

        # loading phase (ground truth)
        else:
            path = os.path.join(self.dir_paths[0], self.file_lists[0][index])
            mat_data = self._load_mat(path)
            pha = mat_data["pha"]

            if self.transform is not None:
                pha = self.transform(pha)

            return pha

    @staticmethod
    def _load_mat(path: str):
        """Load a .mat file and return the dict."""
        import scipy.io as sio
        return sio.loadmat(path)

