"""One dataloader to rule them all. This 'dataloader' loads all self-supervised pretraining data."""

import glob
import os
from random import shuffle

from torch.utils.data import ConcatDataset, DataLoader, random_split

import dataloaders.data_utils as data_utils
from dataloaders.armeni2022 import Armeni2022
from dataloaders.armeni2022_labelled import Armeni2022Labelled
from dataloaders.gwilliams2022 import Gwilliams2022
from dataloaders.schoffelen2019 import Schoffelen2019


class BatchInvariantSampler:
    """Takes a list of dataloaders and iterates by randomly selecting batches from the dataloaders."""

    def __init__(self, dataloaders, shuffle=True):
        self.dataloaders = dataloaders
        self.data_sizes = [len(dataloader) for dataloader in dataloaders]
        self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
        self.batch_order = []

        if not shuffle:
            self._reset()
            self.fixed_batch_order = self.batch_order.copy()

    def __iter__(self):
        if shuffle:
            self._reset()
        else:
            self.batch_order = self.fixed_batch_order.copy()
        return self

    def __next__(self):
        if not self.batch_order:
            raise StopIteration
        dl_idx = self.batch_order.pop(0)
        return next(self.dataloader_iters[dl_idx])

    def __len__(self):
        return sum(self.data_sizes)

    def _generate_batch_order(self):
        batch_order = []
        for i, data_size in enumerate(self.data_sizes):
            batch_order.extend([i for _ in range(data_size)])
        shuffle(batch_order)
        return batch_order

    def _reset(self):
        self.batch_order = self._generate_batch_order()
        self.dataloader_iters = [iter(dataloader) for dataloader in self.dataloaders]


def load_pretraining_data(
    slice_len,
    preproc_config,
    train_ratio,
    batch_size,
    norm_config,
    debug=False,
    labels=None,
    exclude_subjects=None,
):
    """Loads all pretraining data.

    Since we load data from different MEG datasets, we require a dataset-conditional transform for each dataset so that we can transform all data into the same space. Moreover, we will utilise subject-conditional layers to account for differences in each patient. For maximum GPU efficiency, each batch should contain data from only one dataset and subject so a batch can be processed in parallel. To enable this, the BatchInvariantSampler is used to randomly return batches from dataloaders for each dataset and subject.
    """

    loaders = {
        "armeni2022": _load_armeni_2022,
        "gwilliams2022": _load_gwilliams_2022,
        "schoffelen2019": _load_schoffelen_2019,
    }

    if debug:
        ds_name = next(iter(preproc_config.keys()))
        datasets = loaders[ds_name](
            slice_len,
            preproc_config[ds_name],
            debug=debug,
            labels=labels,
            exclude_subjects=exclude_subjects,
        )
    else:
        datasets = []
        for k in preproc_config.keys():
            datasets.extend(
                loaders[k](
                    slice_len,
                    preproc_config[k],
                    labels=labels,
                    exclude_subjects=exclude_subjects,
                )
            )

    train_datasets, test_datasets = [], []

    for dataset in datasets:
        if debug:
            # Use only a single batch when debugging
            train_size = batch_size
            test_size = batch_size
            rest = len(dataset) - train_size - test_size
            train_dataset, test_dataset, _ = random_split(
                dataset, [train_size, test_size, rest]
            )
        else:
            train_size = int(len(dataset) * train_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    train_dataloaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for dataset in train_datasets
    ]
    test_dataloaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for dataset in test_datasets
    ]

    print("Fitting scalers")
    scalers = {}
    for train_dataloader in train_dataloaders:
        scaler = data_utils.BatchScaler(
            n_sample_batches=norm_config["n_sample_batches"] if not debug else 1,
            per_channel=norm_config["per_channel"],
            scaler_conf=norm_config["scaler_conf"],
        ).cuda()
        scaler.fit(train_dataloader)

        first_batch = next(iter(train_dataloader))
        identifiers = [first_batch[-1][k][0] for k in first_batch[-1].keys()]
        scalers[str(identifiers)] = scaler

    train_sampler = BatchInvariantSampler(train_dataloaders, shuffle=True)
    test_sampler = BatchInvariantSampler(test_dataloaders, shuffle=False)

    return train_sampler, test_sampler, scalers


def _load_gwilliams_2022(
    slice_len, preproc_config, debug=False, labels=None, exclude_subjects=None
):
    seconds = 0
    datasets = []

    # Determined via std. rejection
    bad_subjects = [
        # "05", "15", "21", "14", "4"
        "05",
        "19",
        "17",
    ]
    # Generally: watch out for bad channels

    if not debug:
        n_subjects = 27
        n_sessions = 1
        n_tasks = 3
    else:
        n_subjects = 1
        n_sessions = 0
        n_tasks = 0

    # Loop over subjects
    for subj_no in range(1, n_subjects + 1):
        subject = "{:02d}".format(subj_no)  # 01, 02, etc.

        if subject in bad_subjects:
            continue

        # Loop over sessions
        for sess_no in range(0, n_sessions + 1):
            session = str(sess_no)

            for task in range(0, n_tasks + 1):
                task = str(task)

                try:
                    data = Gwilliams2022(
                        subject_id=subject,
                        session=session,
                        task=task,
                        slice_len=slice_len,
                        preproc_config=preproc_config,
                    )
                except Exception as e:
                    print("Skipping: ", e)
                    continue  # Subject may not have completed task or session

                seconds += len(data) * slice_len

                datasets.append(data)

    print(
        f"Loaded approximately {seconds // 3600} hours of data from Gwilliams et al. 2022"
    )

    return datasets


def _load_schoffelen_2019(
    slice_len, preproc_config, debug=False, labels=None, exclude_subjects=None
):
    seconds = 0
    datasets = []

    # Reject the same subjects as Defossez et al. 2023
    bad_nums = [2011, 2036, 2062, 2063, 2076, 2084, 1006, 1014, 1090, 1115]
    no_subject = [1014, 1018, 1021, 1023, 1041, 1043, 1047, 1051, 1056]
    no_subject += [1060, 1067, 1082, 1091, 1096, 1112]
    no_subject += [2012, 2018, 2022, 2023, 2026, 2043, 2044, 2045, 2048]
    no_subject += [2054, 2060, 2074, 2081, 2082, 2087, 2093, 2100, 2107]
    no_subject += [2112, 2115, 2118, 2123]

    # Determined via std. dev. rejection
    reject = ['A2101', 'V1081', 'A2077', 'A2020', 'A2071', 'V1084', 'A2104', 'A2090', 'A2005', 'A2035', 'V1113', 'A2069', 'V1080', 'A2061', 'A2016', 'V1086', 'A2078', 'A2108', 'A2097', 'A2095', 'A2038', 'A2009', 'A2013', 'A2099', 'A2070', 'A2052', 'V1097', 'V1104', 'A2057', 'A2098', 'A2068', 'A2042', 'A2096', 'A2040', 'V1074']

    # Note: "V" subjects read the stimuli, while "A" subjects heard it
    subjects = sorted(
        [
            os.path.basename(path).replace("sub-", "")
            for path in glob.glob(str(data_utils.DATA_PATH) + "/schoffelen2019/sub-*")
        ]
    )

    if debug:
        subjects = [subjects[0]]
        tasks = ["rest"]
    else:
        tasks = ["auditory", "rest", "visual"]

    # Loop over subjects
    for subject in subjects:

        if int(subject[1:]) in (bad_nums + no_subject):
            continue  # ignore incomplete subject data
        elif subject in reject:
            continue

        # Loop over sessions
        for task in tasks:
            if subject.startswith("V") and task == "auditory":
                continue
            elif subject.startswith("A") and task == "visual":
                continue

            try:
                data = Schoffelen2019(
                    subject_id=subject,
                    task=task,
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                )
            except Exception as e:
                print("Skipping: ", e)
                continue  # Not all subjects have recordings for both tasks

            seconds += len(data) * slice_len

            datasets.append(data)

    print(
        f"Loaded approximately {seconds // 3600} hours of data from Schoffelen et al. 2019"
    )

    return datasets


def _load_armeni_2022(
    slice_len, preproc_config, debug=False, labels=None, exclude_subjects=None
):
    seconds = 0
    datasets = []

    if exclude_subjects:
        bad_subjects = exclude_subjects
    else:
        bad_subjects = []

    bad_sessions = {
        "001": [],
        "002": [], #["009"],
        "003": [], #["003", "004", "005", "006", "008"],
    }

    if debug:
        n_subjects = 1
        n_sessions = 1
    else:
        n_subjects = 3
        n_sessions = 10

    # Loop over subjects
    for subj_no in range(1, n_subjects + 1):
        subject = "{:03d}".format(subj_no)  # 001, 002, and 003

        if subject in bad_subjects:
            continue

        # Loop over sessions
        for sess_no in range(1, n_sessions + 1):
            session = "{:03d}".format(sess_no)

            if session in bad_sessions[subject]:
                continue

            if labels:
                data = Armeni2022Labelled(
                    subject_id=subject,
                    session=session,
                    task="compr",
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                    label_type=labels,
                )
            else:
                data = Armeni2022(
                    subject_id=subject,
                    session=session,
                    task="compr",  # There is only one relevant task (compr), the emptyroom task is irrelevant.
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                )

            seconds += len(data) * slice_len

            datasets.append(data)

    print(
        f"Loaded approximately {seconds // 3600} hours of data from Armeni et al. 2022"
    )

    return datasets


if __name__ == "__main__":
    import pprint

    import matplotlib.pyplot as plt
    import numpy as np

    preproc_config = {
        "filtering": True,
        "resample": 300,
        "notch_freqs": [50, 100, 150],
        "bandpass_lo": 0.5,  # Minimum 0.5 to suppress slow-drift artifacts in Armeni et al. 2022
        "bandpass_hi": 150,
    }

    # Even worth running on its own as preprocessing work is cached for next time! 😉
    train_sampler, test_sampler, scalers = load_pretraining_data(
        preproc_config={
            "armeni2022": preproc_config,
            # "gwilliams2022": preproc_config,
            # "schoffelen2019": preproc_config,
        },
        slice_len=3.0,
        train_ratio=0.95,
        batch_size=32,
        norm_config={
            "n_sample_batches": 8,
            "per_channel": True,
            "scaler_conf": {
                # "robust_scaler": {
                #     "lo_q": 0.25,
                #     "hi_q": 0.75,
                # }
                "standard_scaler": None,
            },
        },
        debug=False,  # TODO: change as required
    )

    # Analyse dataset statistics by subject
    print("Analysing dataset statistics")
    subject_data = {}
    sample_batches = 8
    sample_subjects = 3  # TODO: change for different datasets
    for i, batch in enumerate(train_sampler):
        data, times, subject, dataset = batch[0], batch[1], batch[-1]["subject"][0], batch[-1]["dataset"][0]

        scaled_batch = scalers[data_utils.get_scaler_hash(batch)](data)

        if subject not in subject_data:
            subject_data[subject] = [scaled_batch]
        elif len(subject_data[subject]) < sample_batches:
            subject_data[subject].append(scaled_batch)

        # Issue: only collects 3 subjects
        if (
            all([len(v) >= sample_batches for v in subject_data.values()])
            and len(subject_data) >= 3
        ):
            break

    subject_data = {k: np.concatenate(v) for k, v in subject_data.items()}
    stats = {
        k: {
            "mean": v.mean(),
            "std": v.std(),
            "max": v.max(),
            "min": v.min(),
            "0.25": np.quantile(v, 0.25),
            "0.50": np.quantile(v, 0.5),
            "0.75": np.quantile(v, 0.75),
        }
        for k, v in subject_data.items()
    }
    pprint.pprint(stats)
    bad_subjects = [
        k for k in stats.keys() if stats[k]["std"] < 0.9 or stats[k]["std"] > 1.1
    ]
    print("Bad subjects", bad_subjects)
    # ---

    subject_ids = {}
    for batch in train_sampler:
        data, times, subject, dataset = batch[0], batch[1], batch[-1]["subject"][0], batch[-1]["dataset"][0]

        if subject not in subject_ids:
            scaled_batch = scalers[data_utils.get_scaler_hash(batch)](data)
            subject_ids[subject] = scaled_batch

            for channel in scaled_batch[0]:
                plt.plot(times[0], channel)
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")

            plt.ylim(-6, 6)
            plt.savefig(f"{subject}.png")

            plt.cla()
            plt.clf()

        if len(subject_ids) >= sample_subjects:
            print(subject_ids.keys())
            exit(0)
            breakpoint()
            subject_ids = {}
