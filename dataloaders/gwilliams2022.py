"""Gwilliams dataset dataloader."""

import pandas as pd
from torch.utils.data import Dataset

import dataloaders.data_utils as data_utils
import dataloaders.label_utils as label_utils


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
        label_type: str,
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
        self.label_type = label_type

        raw, preprocessed, cache_path = data_utils.load_dataset(
            bids_root=bids_root,
            subject_id=subject_id,
            task=task,
            session=session,
        )

        phoneme_codes = pd.read_csv(bids_root / "phoneme_info.csv")

        events = label_utils.read_events_file(
            bids_root=bids_root,
            subject_id=subject_id,
            session=session,
            task=task,
        )

        self.valid_indices, self.samples_per_slice = data_utils.get_valid_indices(
            raw, slice_len
        )
        self.num_slices = len(self.valid_indices)

        if label_type == "voiced":
            self.phoneme_onsets, self.labels = label_utils.get_voiced_labels_gwilliams(
                events, phoneme_codes, raw
            )
            self.num_slices = len(self.phoneme_onsets)
        elif label_type == "vad":
            self.labels = label_utils.get_vad_labels_gwilliams(events, raw)

        self.raw = raw

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        if self.label_type is None:
            data_slice, times = data_utils.get_slice(
                self.raw, idx, self.samples_per_slice, self.valid_indices
            )
            return {
                "data": data_slice,
                "times": times,
            }

        if self.label_type == "voiced":
            start = self.phoneme_onsets[idx]
            data_slice, times = self.raw[:, start : start + self.samples_per_slice]
            label = self.labels[idx]
            return {
                "data": data_slice,
                "voiced_label": label,
                "times": times,
            }
        elif self.label_type == "vad":
            data_slice, label_slice, times = label_utils.get_slice(
                self.raw,
                self.labels,
                idx,
                self.samples_per_slice,
            )
            return {
                "data": data_slice,
                "vad_labels": label_slice,
                "times": times,
            }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_dataset = Gwilliams2022(
        subject_id="01",
        task="1",
        session="0",
        slice_len=100.0,
        preproc_config={
            "filtering": True,
            "resample": 250,
            "notch_freqs": [50, 100],
            "bandpass_lo": 0.5,
            "bandpass_hi": 125,
        },
        label_type="vad",
    )

    # # Plot label distribution for voiced labels
    # labels = []
    # for i in range(100):
    #     data, label, times, identifiers = test_dataset[i]
    #     labels.append(label)
    # plt.plot(labels)
    # plt.savefig("gwi_voiced.png")

    # Plot label distribution for VAD labels
    data, labels, times, identifiers = test_dataset[0]  # 35-40s (7)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    for channel in data:
        axes[0].plot(times, channel)
    axes[1].plot(times, labels)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("gwilliams_vad.png")
