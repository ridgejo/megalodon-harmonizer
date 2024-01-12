"""One dataloader to rule them all. This 'dataloader' loads all self-supervised pretraining data."""

import glob
import dataloaders.data_utils as data_utils
from dataloaders.armeni2022 import Armeni2022
from dataloaders.schoffelen2019 import Schoffelen2019
from dataloaders.gwilliams2022 import Gwilliams2022
from torch.utils.data import ConcatDataset

def load_pretraining_data(slice_len, preproc_config):
    
    datasets = _load_armeni_2022(slice_len, preproc_config)
    datasets.extend(_load_gwilliams_2022(slice_len, preproc_config))
    datasets.extend(_load_schoffelen_2019(slice_len, preproc_config))

    return ConcatDataset(datasets)

def _load_gwilliams_2022(slice_len, preproc_config):

    seconds = 0
    datasets = []

    # Loop over subjects
    for subj_no in range(1, 27 + 1):
        subject = "{:02d}".format(subj_no) # 01, 02, etc.

        # Loop over sessions
        for sess_no in range(0, 1 + 1):
            session = str(sess_no)

            for task in range(0, 3 + 1):
                task = str(task)

                data = Gwilliams2022(
                    subject_id=subject,
                    session=session,
                    task="compr",
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                )

                seconds += len(data) * slice_len
                
                datasets.append(data)

    print(f"Loaded approximately {seconds // 3600} hours of data from Gwilliams et al. 2022")
    
    return datasets

def _load_schoffelen_2019(slice_len, preproc_config):

    seconds = 0
    datasets = []

    # Note: "V" subjects read the stimuli, while "A" subjects heard it
    subjects = [
        os.path.basename(path).replace('sub-', '') for path in glob.glob(
            str(data_utils.DATA_PATH) + "/schoffelen2019/sub-*"
        )
    ]

    # Loop over subjects
    for subject in subjects:

        # Loop over sessions
        for task in ['auditory', 'rest']:

            try:
                data = Schoffelen2019(
                    subject_id=subject,
                    task=task,
                    slice_len=slice_len,
                    preproc_config=preproc_config,
                )
            except:
                continue # Not all subjects have recordings for both tasks

            seconds += len(data) * slice_len
            
            datasets.append(data)

    print(f"Loaded approximately {seconds // 3600} hours of data from Schoffelen et al. 2019")
    
    return datasets

def _load_armeni_2022(slice_len, preproc_config):

    seconds = 0
    datasets = []

    # Loop over subjects
    for subj_no in range(1, 3 + 1):
        subject = "{:03d}".format(subj_no) # 001, 002, and 003

        # Loop over sessions
        for sess_no in range(1, 10 + 1):
            session = "{:03d}".format(sess_no)

            data = Armeni2022(
                subject_id=subject,
                session=session,
                task="compr", # There is only one relevant task (compr), the emptyroom task is irrelevant.
                slice_len=slice_len,
                preproc_config=preproc_config,
            )

            seconds += len(data) * slice_len
            
            datasets.append(data)

    print(f"Loaded approximately {seconds // 3600} hours of data from Armeni et al. 2022")
    
    return datasets

if __name__ == "__main__":

    load_pretraining_data(
        preproc_config={
            "filtering": True,
            "resample": 300,
            "notch_freqs": [50, 100, 150],
            "bandpass_lo": 0.1,
            "bandpass_hi": 150,
        },
        slice_len=0.5,
    )