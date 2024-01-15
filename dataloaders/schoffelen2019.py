"""MOUS dataset dataloader."""

import gc

from torch.utils.data import Dataset

import dataloaders.data_utils as data_utils


class Schoffelen2019(Dataset):
    """
    Loads fixed windows from the MEG data in Schoeffelen et al. (2019).
    Source: https://www.nature.com/articles/s41597-019-0020-y
    """

    def __init__(
        self,
        subject_id: str,
        task: str,
        slice_len: float,
        preproc_config: dict,
        bids_root: str = data_utils.DATA_PATH / "schoffelen2019",
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

        self.subject_id = subject_id

        raw, preprocessed, cache_path = data_utils.load_dataset(
            bids_root=bids_root,
            subject_id=subject_id,
            task=task,
            session=None,
            preproc_config=preproc_config,
        )

        if not preprocessed:
            # Schoeffelen MEG channels are named starting with an "M"
            meg_channels = sorted(
                [ch_name for ch_name in raw.ch_names if ch_name.startswith("M")]
            )[:273]  # issue: hopefully this doesn't change channel order in data

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

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        data_slice, times = data_utils.get_slice(self.raw, idx, self.samples_per_slice)

        return data_slice, times, self.__class__.__name__, self.subject_id


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_dataset = Schoffelen2019(
        subject_id="A2002",
        task="auditory",
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
    plt.savefig("schoffelen.png")
