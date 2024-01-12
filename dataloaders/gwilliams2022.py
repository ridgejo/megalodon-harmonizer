"""Gwilliams dataset dataloader."""

import dataloaders.data_utils as data_utils
from typing import List, Optional, Union
from torch.utils.data import Dataset

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
        bids_root: str = data_utils.DATA_PATH / 'gwilliams2022',
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

        raw, preprocessed, cache_path = data_utils.load_dataset(
            bids_root=bids_root,
            subject_id=subject_id,
            task=task,
            session=session,
            preproc_config=preproc_config,
        )

        if not preprocessed:

            # Gwilliams MEG channels are named starting with "MEG"
            meg_channels = [ch_name for ch_name in raw.ch_names if ch_name.startswith("MEG")]

            # All channels are gradiometer channels (I think?)
            raw.set_channel_types(dict(zip(meg_channels, ['grad' for _ in meg_channels])))

            raw = data_utils.preprocess(
                raw=raw,
                preproc_config=preproc_config,
                channels=meg_channels,
                cache_path=cache_path,
            )

        self.raw = raw
        
        self.mean, self.p5, self.p95 = data_utils.get_norm_stats(raw)
        self.num_slices, self.samples_per_slice = data_utils.get_slice_stats(raw, slice_len)

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):

        data_slice, times = self.raw[
            : , self.samples_per_slice * idx : self.samples_per_slice * (idx + 1)
        ]

        data_slice = 2 * (((data_slice) - self.p5) / (self.p95 - self.p5)) - 1
        data_slice -= self.mean

        return data_slice, times


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
        }
    )

    data, times = test_dataset[0] # 35-40s
    for channel in data:
        plt.plot(times, channel)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
    
    plt.savefig("gwilliams.png")