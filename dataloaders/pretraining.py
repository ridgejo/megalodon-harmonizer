"""One dataloader to rule them all. This 'dataloader' loads all self-supervised pretraining data."""

import glob
import os
from random import shuffle

from torch.utils.data import ConcatDataset, DataLoader, random_split

import dataloaders.data_utils as data_utils
from dataloaders.armeni2022 import Armeni2022
from dataloaders.gwilliams2022 import Gwilliams2022
from dataloaders.schoffelen2019 import Schoffelen2019

class BatchInvariantSampler:
    """Takes a list of dataloaders and iterates by randomly selecting batches from the dataloaders."""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.data_sizes = [len(dataloader) for dataloader in dataloaders]
        self.dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
        self.batch_order = []

    def __iter__(self):
        self._reset()
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
    baseline_correction_samples,
    n_sample_batches,
    debug=False,
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
            slice_len, preproc_config[ds_name], debug=debug
        )
    else:
        datasets = []
        for k in preproc_config.keys():
            datasets.extend(loaders[k](slice_len, preproc_config[k]))

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
            correction_samples=baseline_correction_samples,
            n_sample_batches=n_sample_batches if not debug else 1,
        ).cuda()
        scaler.fit(train_dataloader)

        first_batch = next(iter(train_dataloader))
        subject_name, dataset_name = first_batch[-1][0], first_batch[-2][0]
        if dataset_name in scalers:
            scalers[dataset_name][subject_name] = scaler
        else:
            scalers[dataset_name] = {subject_name: scaler}

    train_sampler = BatchInvariantSampler(train_dataloaders)
    test_sampler = BatchInvariantSampler(test_dataloaders)

    return train_sampler, test_sampler, scalers


def _load_gwilliams_2022(slice_len, preproc_config, debug=False):
    seconds = 0
    datasets = []

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

        subject_datasets = []

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
                except Exception:
                    continue  # Subject may not have completed task or session

                seconds += len(data) * slice_len

                subject_datasets.append(data)

        if len(subject_datasets) > 0:
            datasets.append(ConcatDataset(subject_datasets))

    print(
        f"Loaded approximately {seconds // 3600} hours of data from Gwilliams et al. 2022"
    )

    return datasets


def _load_schoffelen_2019(slice_len, preproc_config, debug=False):
    seconds = 0
    datasets = []

    # Note: "V" subjects read the stimuli, while "A" subjects heard it
    subjects = sorted([
        os.path.basename(path).replace("sub-", "")
        for path in glob.glob(str(data_utils.DATA_PATH) + "/schoffelen2019/sub-*")
    ])

    if debug:
        subjects = [subjects[0]]
        tasks = ["rest"]
    else:
        tasks = ["auditory", "rest"]

    # Loop over subjects
    for subject in subjects:
        subject_datasets = []

        # Loop over sessions
        for task in tasks:
            try:
                data = Schoffelen2019(
                    subject_id=subject,
                    task=task,
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                )
            except Exception:
                continue  # Not all subjects have recordings for both tasks

            seconds += len(data) * slice_len

            subject_datasets.append(data)

        if len(subject_datasets) > 0:
            datasets.append(ConcatDataset(subject_datasets))

    print(
        f"Loaded approximately {seconds // 3600} hours of data from Schoffelen et al. 2019"
    )

    return datasets


def _load_armeni_2022(slice_len, preproc_config, debug=False):
    seconds = 0
    datasets = []

    if debug:
        n_subjects = 1
        n_sessions = 1
    else:
        n_subjects = 3
        n_sessions = 10

    # Loop over subjects
    for subj_no in range(1, n_subjects + 1):
        subject = "{:03d}".format(subj_no)  # 001, 002, and 003

        subject_datasets = []

        # Loop over sessions
        for sess_no in range(1, n_sessions + 1):
            session = "{:03d}".format(sess_no)

            data = Armeni2022(
                subject_id=subject,
                session=session,
                task="compr",  # There is only one relevant task (compr), the emptyroom task is irrelevant.
                slice_len=slice_len,
                preproc_config=preproc_config,
            )

            seconds += len(data) * slice_len

            subject_datasets.append(data)

        if len(subject_datasets) > 0:
            datasets.append(ConcatDataset(subject_datasets))

    print(
        f"Loaded approximately {seconds // 3600} hours of data from Armeni et al. 2022"
    )

    return datasets


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    preproc_config = {
        "filtering": True,
        "resample": 300,
        "notch_freqs": [50, 100, 150],
        "bandpass_lo": 0.5, # Minimum 0.5 to suppress slow-drift artifacts in Armeni et al. 2022
        "bandpass_hi": 150,
    }

    # Even worth running on its own as preprocessing work is cached for next time! 😉
    train_sampler, test_sampler, scalers = load_pretraining_data(
        preproc_config={
            "armeni2022": preproc_config,
            "gwilliams2022": preproc_config,
            "schoffelen2019": preproc_config,
        },
        slice_len=3.0,
        train_ratio=0.95,
        batch_size=32,
        baseline_correction_samples=1000,
        n_sample_batches=1,  # torch.quantile() errors if given more than 16M samples in 1D tensor
        debug=False,  # TODO: change as required
    )

    subject_ids = {}
    for batch in train_sampler:
        data, times, subject, dataset = batch[0], batch[1], batch[-1][0], batch[-2][0]
        scaled_batch = scalers[dataset][subject](data)

        if subject not in subject_ids:
            subject_ids[subject] = scaled_batch

            for channel in scaled_batch[0]:
                plt.plot(times[0], channel)
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")

            plt.ylim(-6, 6)
            plt.savefig(f"{subject}.png")

            plt.cla()
            plt.clf()

        if len(subject_ids) >= 3:
            print(subject_ids.keys())
            breakpoint()
            subject_ids = {}