import numpy as np
import pandas as pd

ARPABET = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
ARPABET_NEGVOICE = ["P", "T", "CH", "F", "TH", "S", "SH", "HH"]


def _is_phoneme(description):
    if description in ARPABET:
        return description, True
    elif (
        len(description) == 3
        and description[2].isnumeric()
        and description[:2] in ARPABET
    ):
        return description[:2], True
    else:
        return "", False


def read_events_file(bids_root, subject_id, session, task):
    return pd.read_csv(
        bids_root
        / f"sub-{subject_id}/ses-{session}/meg/sub-{subject_id}_ses-{session}_task-{task}_events.tsv",
        sep="\t",
    )


def get_vad_labels(events, raw, offset=0.0):
    """Get labels corresponding to occurrence of human speech."""

    # TODO: Armeni's labels are "gapless" and therefore VAD is not a good task for this. Use Birtan's labels instead.

    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)

    phoneme_events = events[["phoneme_onset" in c for c in list(events["type"])]]
    labels = np.zeros(len(raw))
    for i, phoneme_event in phoneme_events.iterrows():
        onset = float(phoneme_event["onset"])
        duration = float(phoneme_event["duration"])
        t_start = (
            int(onset * sample_freq) + offset_samples
        )  # Delay labels so they occur at same time as brain response
        t_end = int((onset + duration) * sample_freq) + offset_samples
        labels[t_start : t_end + 1] = 1.0

    # Warning: labels may need to be downsampled later (e.g. if encoded). Deal with label downsampling online (scipy?)

    return labels


def get_voiced_labels(events, raw, offset=0.0):
    """Get aligned index for start of every phoneme and align to longest."""
    # Longest phoneme in Armeni: 0.8s
    # 99% are less than 0.27s however

    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)

    phoneme_events = events[["phoneme_onset" in c for c in list(events["type"])]]

    phoneme_onsets = []
    labels = []
    for i, phoneme_event in phoneme_events.iterrows():
        value = phoneme_event["value"]
        value, is_phoneme = _is_phoneme(value)
        if is_phoneme:
            onset = float(phoneme_event["onset"])

            if onset > 0:
                t_start = int(onset * sample_freq) + offset_samples

                if value in ARPABET_NEGVOICE:
                    labels.append(0.0)
                else:
                    labels.append(1.0)

                phoneme_onsets.append(t_start)

    return phoneme_onsets, labels


def get_slice(raw, labels, idx, samples_per_slice):
    data_slice, times = raw[:, samples_per_slice * idx : samples_per_slice * (idx + 1)]
    label_slice = labels[samples_per_slice * idx : samples_per_slice * (idx + 1)]
    return data_slice, label_slice, times
