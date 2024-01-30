import numpy as np
import pandas as pd
import textgrid

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

def read_textgrid(bids_root, subject_id, session, events):
    grid = textgrid.TextGrid.fromFile(
        bids_root / f"adventuresofsherlockholmes_{session[1:]}_doyle_64kb.TextGrid"
    )

    # warning: Sessions 003-009 have not been checked.
    session_timing = {
        "001": {
            "start": 13,
            "end": -4,
        },
        "002": {
            "start": 7,
            "end": -4,
        },
        "003": {
            "start": 7,
            "end": -4,
        },
        "004": {
            "start": 7,
            "end": -4,
        },
        "005": {
            "start": 7,
            "end": -4,
        },
        "006": {
            "start": 7,
            "end": -4,
        },
        "007": {
            "start": 7,
            "end": -4,
        },
        "008": {
            "start": 7,
            "end": -4,
        },
        "009": {
            "start": 7,
            "end": -4,
        },
        "010": {
            "start": 7,
            "end": -4,
        },
    }

    true_start = events[["word_onset" in c for c in list(events["type"])]]
    true_start = true_start[[v != "sp" for v in list(true_start["value"])]]
    true_start = float(true_start.iloc[0]["onset"])
    start, end = session_timing[session]["start"], session_timing[session]["end"]

    return grid[0][start : end + 1], true_start # Return intervals in tier zero

def get_vad_labels_from_textgrid(tier, brain_start, raw, offset=0.0):

    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)
    grid_start = tier[0].minTime

    labels = np.zeros(len(raw))
    for event in tier:
        onset = float(event.minTime) - grid_start + brain_start
        offset = float(event.maxTime) - grid_start + brain_start

        t_start = int(onset * sample_freq + offset_samples)
        t_end = int(offset * sample_freq + offset_samples)

        if event.mark != '':
            labels[t_start : t_end + 1] = 1.0
    
    return labels


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
