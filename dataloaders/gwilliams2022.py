"""Gwilliams dataset dataloader."""

import gc

from torch.utils.data import Dataset

import dataloaders.data_utils as data_utils


class Gwilliams2022(Dataset):
    """
    Loads fixed windows from the MEG data in Gwilliams et al. (2022).
    Source: https://arxiv.org/abs/2208.11488
    """

    def __init__(
        self,
        subject_id: str,
        task: str,
        session: str,
        slice_len: float,
        preproc_config: dict,
        bids_root: str = data_utils.DATA_PATH / "gwilliams2022",
    ):
        """
        Args:
            bids_root: Directory of dataset.
            subject_id: Subject.
            session: Recording session.
            task: Recording task.
            slice_len: Length (in seconds) of items returned by getter.
            preproc_config: Dictionary with preprocessing settings.
        """

        self.session = session
        self.subject_id = subject_id

        raw, preprocessed, cache_path = data_utils.load_dataset(
            bids_root=bids_root,
            subject_id=subject_id,
            task=task,
            session=session,
            preproc_config=preproc_config,
        )

        self.valid_indices, self.samples_per_slice = data_utils.get_valid_indices(raw, slice_len)
        self.num_slices = len(self.valid_indices)

        self.raw = raw

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        data_slice, times = data_utils.get_slice(self.raw, idx, self.samples_per_slice, self.valid_indices)

        identifiers = {
            "dataset": self.__class__.__name__,
            "subject": self.subject_id,
            "session": self.session,
        }

        return data_slice, times, identifiers


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_dataset = Gwilliams2022(
        subject_id="01",
        task="1",
        session="0",
        slice_len=5,
        preproc_config={
            "filtering": True,
            "resample": 300,
            "notch_freqs": [50, 100, 150],
            "bandpass_lo": 0.1,
            "bandpass_hi": 150,
        },
    )

    data, times, dataset = test_dataset[0]
    for channel in data:
        plt.plot(times, channel)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.ylim(-6, 6)
    plt.savefig("gwilliams.png")
