import glob
import os

from osl import preprocessing

from dataloaders.data_utils import DATA_PATH


def preproc_schoffelen2019():
    config = """
        preproc:
        - pick_types: {meg: true, ref_meg: false}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff}
        - bad_channels: {picks: mag}
        - interpolate_bads: {}
    """

    root = DATA_PATH / "schoffelen2019"
    preproc_root = root / "preproc"
    preproc_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted(
        [
            os.path.basename(path).replace("sub-", "")
            for path in glob.glob(str(root) + "/sub-*")
        ]
    )
    for subject in subjects:
        # Find tasks
        inputs = glob.glob(str(root) + f"/sub-{subject}/meg/*_task-*_meg.ds")

        preproc_dir = preproc_root / f"sub-{subject}"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=str(preproc_dir),
            overwrite=True,
            dask_client=False,
        )


def preproc_gwilliams2022():
    config = """
        preproc:
        - pick_types: {meg: true, ref_meg: false}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff}
        - bad_channels: {picks: mag}
        - interpolate_bads: {}
    """

    root = DATA_PATH / "gwilliams2022"
    preproc_root = root / "preproc"
    preproc_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted(
        [
            os.path.basename(path).replace("sub-", "")
            for path in glob.glob(str(root) + "/sub-*")
        ]
    )
    for subject in subjects:
        # Find sessions
        sessions = sorted(
            [
                os.path.basename(path).replace("ses-", "")
                for path in glob.glob(str(root) + f"/sub-{subject}/ses-*")
            ]
        )

        inputs = []
        for session in sessions:
            # Find tasks
            inputs.extend(
                sorted(
                    glob.glob(
                        str(root) + f"/sub-{subject}/ses-{session}/meg/*_task-*_meg.con"
                    )
                )
            )

        preproc_dir = preproc_root / f"sub-{subject}"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=str(preproc_dir),
            overwrite=True,
            dask_client=False,
        )


def preproc_armeni2022():
    """
    Batch preprocess all runs for all subjects using OSL.
    """

    config = """
        preproc:
        - pick_types: {meg: true, ref_meg: false}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff}
        - bad_channels: {picks: mag}
        - interpolate_bads: {}
    """

    # Armeni 2022
    armeni_root = DATA_PATH / "armeni2022"
    preproc_root = armeni_root / "preproc"
    preproc_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted(
        [
            os.path.basename(path).replace("sub-", "")
            for path in glob.glob(str(armeni_root) + "/sub-*")
        ]
    )
    for subject in subjects:
        inputs = []
        for session in range(1, 10 + 1):
            sess = "{:03d}".format(session)
            inputs.append(
                str(
                    armeni_root
                    / f"sub-{subject}/ses-{sess}/meg/sub-{subject}_ses-{sess}_task-compr_meg.ds"
                )
            )

        preproc_dir = preproc_root / f"sub-{subject}"

        preprocessing.run_proc_batch(
            config,
            inputs,
            outdir=str(preproc_dir),
            overwrite=True,
            dask_client=False,
        )


if __name__ == "__main__":
    # preproc_armeni2022()
    preproc_gwilliams2022()
    # preproc_schoffelen2019()
