"""MOUS dataset dataloader."""

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

import dataloaders.data_utils as data_utils

MEG_CHANNELS = [
    "MLC11",
    "MLC12",
    "MLC13",
    "MLC14",
    "MLC15",
    "MLC16",
    "MLC17",
    "MLC21",
    "MLC22",
    "MLC23",
    "MLC24",
    "MLC25",
    "MLC31",
    "MLC32",
    "MLC41",
    "MLC42",
    "MLC51",
    "MLC52",
    "MLC53",
    "MLC54",
    "MLC55",
    "MLC61",
    "MLC62",
    "MLC63",
    "MLF11",
    "MLF12",
    "MLF13",
    "MLF14",
    "MLF21",
    "MLF22",
    "MLF23",
    "MLF24",
    "MLF25",
    "MLF31",
    "MLF32",
    "MLF33",
    "MLF34",
    "MLF35",
    "MLF41",
    "MLF42",
    "MLF43",
    "MLF44",
    "MLF45",
    "MLF46",
    "MLF51",
    "MLF52",
    "MLF53",
    "MLF54",
    "MLF55",
    "MLF56",
    "MLF61",
    "MLF62",
    "MLF63",
    "MLF64",
    "MLF65",
    "MLF66",
    "MLF67",
    "MLO11",
    "MLO12",
    "MLO13",
    "MLO14",
    "MLO21",
    "MLO22",
    "MLO23",
    "MLO24",
    "MLO31",
    "MLO32",
    "MLO33",
    "MLO34",
    "MLO41",
    "MLO42",
    "MLO43",
    "MLO44",
    "MLO51",
    "MLO52",
    "MLO53",
    "MLP11",
    "MLP12",
    "MLP21",
    "MLP22",
    "MLP23",
    "MLP31",
    "MLP32",
    "MLP33",
    "MLP34",
    "MLP35",
    "MLP41",
    "MLP42",
    "MLP43",
    "MLP44",
    "MLP45",
    "MLP51",
    "MLP52",
    "MLP53",
    "MLP54",
    "MLP55",
    "MLP56",
    "MLP57",
    "MLT11",
    "MLT12",
    "MLT13",
    "MLT14",
    "MLT15",
    "MLT16",
    "MLT21",
    "MLT22",
    "MLT23",
    "MLT24",
    "MLT25",
    "MLT26",
    "MLT27",
    "MLT31",
    "MLT32",
    "MLT33",
    "MLT34",
    "MLT35",
    "MLT36",
    "MLT37",
    "MLT41",
    "MLT42",
    "MLT43",
    "MLT44",
    "MLT45",
    "MLT46",
    "MLT47",
    "MLT51",
    "MLT52",
    "MLT53",
    "MLT54",
    "MLT55",
    "MLT56",
    "MLT57",
    "MRC11",
    "MRC12",
    "MRC13",
    "MRC14",
    "MRC15",
    "MRC16",
    "MRC17",
    "MRC21",
    "MRC22",
    "MRC23",
    "MRC24",
    "MRC25",
    "MRC31",
    "MRC32",
    "MRC41",
    "MRC42",
    "MRC51",
    "MRC52",
    "MRC53",
    "MRC54",
    "MRC55",
    "MRC61",
    "MRC62",
    "MRC63",
    "MRF11",
    "MRF12",
    "MRF13",
    "MRF14",
    "MRF21",
    "MRF22",
    "MRF23",
    "MRF24",
    "MRF25",
    "MRF31",
    "MRF32",
    "MRF33",
    "MRF34",
    "MRF35",
    "MRF41",
    "MRF42",
    "MRF43",
    "MRF44",
    "MRF45",
    "MRF46",
    "MRF51",
    "MRF52",
    "MRF53",
    "MRF54",
    "MRF55",
    "MRF56",
    "MRF61",
    "MRF62",
    "MRF63",
    "MRF64",
    "MRF65",
    "MRF67",
    "MRO11",
    "MRO12",
    "MRO13",
    "MRO14",
    "MRO21",
    "MRO22",
    "MRO23",
    "MRO24",
    "MRO31",
    "MRO32",
    "MRO33",
    "MRO34",
    "MRO41",
    "MRO42",
    "MRO43",
    "MRO44",
    "MRO51",
    "MRO53",
    "MRP11",
    "MRP12",
    "MRP21",
    "MRP22",
    "MRP23",
    "MRP31",
    "MRP32",
    "MRP33",
    "MRP34",
    "MRP35",
    "MRP41",
    "MRP42",
    "MRP43",
    "MRP44",
    "MRP45",
    "MRP51",
    "MRP52",
    "MRP53",
    "MRP54",
    "MRP55",
    "MRP56",
    "MRP57",
    "MRT11",
    "MRT12",
    "MRT13",
    "MRT14",
    "MRT15",
    "MRT16",
    "MRT21",
    "MRT22",
    "MRT23",
    "MRT24",
    "MRT25",
    "MRT26",
    "MRT27",
    "MRT31",
    "MRT32",
    "MRT33",
    "MRT34",
    "MRT35",
    "MRT36",
    "MRT37",
    "MRT41",
    "MRT42",
    "MRT43",
    "MRT44",
    "MRT45",
    "MRT46",
    "MRT47",
    "MRT51",
    "MRT52",
    "MRT53",
    "MRT54",
    "MRT55",
    "MRT56",
    "MRT57",
    "MZC01",
    "MZC02",
    "MZC03",
    "MZC04",
    "MZF01",
    "MZF02",
    "MZF03",
    "MZO01",
    "MZO02",
    "MZO03",
    "MZP01",
]


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
        )

        picks = mne.pick_types(
            raw.info, meg=True, eeg=False, stim=False, eog=False, ecg=False
        )[:273]  # [28: (28 + 273)]
        raw = raw.pick(picks)

        sfreq = float(raw.info["sfreq"])
        self.samples_per_slice = int(sfreq * slice_len)
        self.num_slices = int(len(raw) / self.samples_per_slice)

        self.raw = raw

        # Extract 3D sensor positions from raw object
        sensor_positions = []
        for ch in raw.info["chs"]:
            pos = ch["loc"][:3]  # Extracts the first three elements: X, Y, Z
            sensor_positions.append(pos)
        self.sensor_positions = torch.tensor(np.array(sensor_positions))

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):

        data_slice, times = self.raw[:, idx * self.samples_per_slice : (idx + 1) * self.samples_per_slice]

        return {
            "data": data_slice,
            "times": times,
            "identifier": {"subject": self.subject_id, "dataset": "schoffelen2019"},
            # "sensor_pos": self.sensor_positions,  # Sensor (x, y, z) provided in [m]
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_dataset = Schoffelen2019(
        subject_id="A2002",
        task="auditory",
        slice_len=3,
    )

    data, times, dataset = test_dataset[0]
    for channel in data:
        plt.plot(times, channel)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.ylim(-6, 6)
    plt.savefig("schoffelen.png")
