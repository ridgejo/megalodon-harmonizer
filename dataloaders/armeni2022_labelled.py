"""10-hour dataset dataloader"""

import gc
import typing as tp

from torch.utils.data import Dataset

import dataloaders.data_utils as data_utils
import dataloaders.label_utils as label_utils


class Armeni2022Labelled(Dataset):
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
        label_type: str,
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
            label_type: Required label type (vad / etc.)
        """

        self.subject_id = subject_id
        self.session = session
        self.label_type = label_type

        raw, preprocessed, cache_path = data_utils.load_dataset(
            bids_root=bids_root,
            subject_id=subject_id,
            task=task,
            session=session,
            preproc_config=preproc_config,
        )

        events = label_utils.read_events_file(
            bids_root=bids_root,
            subject_id=subject_id,
            session=session,
            task=task,
        )
        
        textgrid = label_utils.read_textgrid(
            bids_root=bids_root,
            subject_id=subject_id,
            session=session,
            events=events,
        )

        self.valid_indices, self.samples_per_slice = data_utils.get_valid_indices(raw, slice_len)
        self.num_slices = len(self.valid_indices)

        if label_type == "vad":
            self.labels = label_utils.get_vad_labels_from_textgrid(textgrid, raw)
            # self.labels = label_utils.get_vad_labels(events, raw)
        elif label_type == "voiced":
            self.phoneme_onsets, self.labels = label_utils.get_voiced_labels(
                events, raw
            )
            self.num_slices = len(self.phoneme_onsets)
        else:
            raise ValueError(f"Unrecognised label type {label_type}")

        self.raw = raw

        if truncate:
            self.num_slices = min(truncate, self.num_slices)

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        if self.label_type == "vad":
            data_slice, label_slice, times = label_utils.get_slice(
                self.raw, self.labels, idx, self.samples_per_slice
            )
        elif self.label_type == "voiced":
            start = self.phoneme_onsets[idx]
            data_slice, times = self.raw[:, start : start + self.samples_per_slice]
            label_slice = self.labels[
                idx
            ]  # Warning: not actually a slice, just a single label.

        identifiers = {
            "dataset": "Armeni2022",
            "subject": self.subject_id,
            "session": self.session,
        }

        return data_slice, label_slice, times, identifiers


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_dataset = Armeni2022Labelled(
        subject_id="001",
        session="001",
        task="compr",
        slice_len=1000,#0.3,  # 5
        preproc_config={
            "filtering": True,
            "resample": 250,
            "notch_freqs": [50, 100],
            "bandpass_lo": 0.5,
            "bandpass_hi": 125,
        },
        label_type="vad",
    )

    # # Plot distribution of voiced labels
    # all_labels = []
    # for i in range(1000):
    #     _, label, _, _, _ = test_dataset[i]
    #     all_labels.append(label)
    # plt.plot(all_labels)
    # plt.savefig("voiced_labels.png")
    # print(np.histogram(all_labels))

    data, labels, times, identifiers = test_dataset[0]  # 35-40s (7)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    for channel in data:
        axes[0].plot(times, channel)
    axes[1].plot(times, labels)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.savefig("armeni_vad.png")
