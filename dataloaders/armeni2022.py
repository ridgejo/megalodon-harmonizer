"""10-hour dataset dataloader"""

import gc
import typing as tp

from torch.utils.data import Dataset

import dataloaders.data_utils as data_utils


class Armeni2022(Dataset):
    """
    Loads fixed windows from the MEG data in Armeni et al. (2022).
    Source: https://www.nature.com/articles/s41597-022-01382-7
    """

    def __init__(
        self,
        subject_id: str,
        task: str,
        session: str,
        slice_len: float,
        preproc_config: dict,
        bids_root: str = data_utils.DATA_PATH / "armeni2022",
        truncate: tp.Optional[int] = None,
    ):
        """
        Args:
            bids_root: Directory of dataset.
            subject_id: Subject (1-3).
            session: Recording session (1-10).
            task: Recording task ('compr'/'emptyroom').
            slice_len: Length (in seconds) of items returned by getter.
            preproc_config: Dictionary with preprocessing settings.
        """

        self.subject_id = subject_id
        self.session = session

        raw, preprocessed, cache_path = data_utils.load_dataset(
            bids_root=bids_root,
            subject_id=subject_id,
            task=task,
            session=session,
            preproc_config=preproc_config,
        )

        if not preprocessed:
            # Subject 3 suffers from unstable channels... remove during preprocessing
            bad_channels = ["MRC23", "MLO22", "MRP5", "MLC23"]

            # Armeni MEG channels are named starting with an "M"
            meg_channels = [
                ch_name
                for ch_name in raw.ch_names
                if ch_name.startswith("M") and ch_name not in bad_channels
            ]

            # All channels are gradiometer channels (I think?)
            raw.set_channel_types(
                dict(zip(meg_channels, ["grad" for _ in meg_channels]))
            )

            raw = data_utils.preprocess(
                raw=raw,
                preproc_config=preproc_config,
                channels=meg_channels,
                cache_path=cache_path,
            )

            del raw
            gc.collect()
            # Lazy read after preprocessing
            raw, _, _ = data_utils.load_dataset(
                bids_root=None,
                subject_id=None,
                task=None,
                session=None,
                preproc_config=None,
                cache_path=cache_path,
            )

        self.raw = raw

        self.num_slices, self.samples_per_slice = data_utils.get_slice_stats(
            raw, slice_len
        )

        if truncate:
            self.num_slices = min(truncate, self.num_slices)

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        data_slice, times = data_utils.get_slice(self.raw, idx, self.samples_per_slice)

        identifiers = {
            "dataset": self.__class__.__name__,
            "subject": self.subject_id,
            "session": self.session,
        }

        return data_slice, times, identifiers


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_dataset = Armeni2022(
        subject_id="003",
        session="004",
        task="compr",
        slice_len=5,
        preproc_config={
            "filtering": True,
            "resample": 300,
            "notch_freqs": [50, 100, 150],
            "bandpass_lo": 0.1,
            "bandpass_hi": 150,
        },
    )

    data, times, dataset, subject = test_dataset[7]  # 35-40s
    for channel in data:
        plt.plot(times, channel)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.ylim(-6, 6)
    plt.savefig("armeni.png")
