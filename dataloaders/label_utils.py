import ast

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

    # warning: one of the sessions has a script crash error. ignore that session.

    # Align the labels with the recording, including the long periods of silence.
    word_events = events[["word_onset" in c for c in list(events["type"])]]
    gaps = (
        word_events["onset"] - (word_events["onset"] + word_events["duration"]).shift()
    )
    gaps = gaps.sort_values(ascending=False)
    gaps = gaps[gaps > 10]
    gaps = gaps.sort_index()

    # Get onset of gap and insert a marker that's the corresponding length of silence. Then labels will be aligned.

    # Step 1. Align start points of brain recording and force-aligned labels
    true_start = word_events[[v != "sp" for v in list(word_events["value"])]]
    true_start = float(true_start.iloc[0]["onset"])

    # This is measured in markers
    start, end = session_timing[session]["start"], session_timing[session]["end"]
    session_markers = grid[0][start : end + 1]

    # Step 2. Align existing markers to position in brain recording
    marker_start = session_markers[0].minTime
    for i in range(len(session_markers)):
        # Compute appropriate delay to add
        delay = 0.0
        for idx, gap in zip(gaps.index, gaps):
            t_start = events.loc[idx - 4]["onset"]
            if session_markers[i].minTime - marker_start + true_start + delay > t_start:
                delay += gap

        session_markers[i].minTime = (
            session_markers[i].minTime - marker_start + true_start + delay
        )
        session_markers[i].maxTime = (
            session_markers[i].maxTime - marker_start + true_start + delay
        )

    return session_markers  # Return intervals in tier zero


def get_vad_labels_from_csv(raw, bids_root, subject, session, task, offset=0.0):
    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)

    # Read CSV
    vad_events = pd.read_csv(
        f"{bids_root}/sub-{subject}/sub-{subject}_ses-{session}_task-{task}_tgs.csv"
    )

    labels = np.zeros(len(raw))
    for i, event in vad_events.iterrows():
        onset = float(event["start_events"])
        end = float(event["end_events"])

        t_start = int(onset * sample_freq + offset_samples)
        t_end = int(end * sample_freq + offset_samples)

        labels[t_start : t_end + 1] = 1.0

    return labels


def get_vad_labels_from_textgrid(tier, raw, offset=0.0):
    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)

    labels = np.zeros(len(raw))
    for event in tier:
        onset = float(event.minTime)
        offset = float(event.maxTime)

        t_start = int(onset * sample_freq + offset_samples)
        t_end = int(offset * sample_freq + offset_samples)

        if event.mark != "":
            labels[t_start : t_end + 1] = 1.0

    return labels


def get_vad_labels(events, raw, offset=0.0):
    """Get labels corresponding to occurrence of human speech."""

    # TODO: Armeni's labels are "gapless" and therefore VAD is not a good task for this. Use Birtan's labels instead.

    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)

    phoneme_events = events[["word_onset" in c for c in list(events["type"])]]
    labels = np.zeros(len(raw))
    for i, phoneme_event in phoneme_events.iterrows():
        onset = float(phoneme_event["onset"])
        duration = float(phoneme_event["duration"])
        t_start = (
            int(onset * sample_freq) + offset_samples
        )  # Delay labels so they occur at same time as brain response
        t_end = int((onset + duration) * sample_freq) + offset_samples
        labels[t_start : t_end + 1] = 1.0

    return labels


def get_vad_labels_gwilliams(events, raw, offset=0.0):
    sample_freq = raw.info["sfreq"]
    offset_samples = int(sample_freq * offset)

    word_events = events[
        ["'kind': 'word'" in trial_type for trial_type in list(events["trial_type"])]
    ]
    labels = np.zeros(len(raw))
    for i, word_event in word_events.iterrows():
        onset = float(word_event["onset"])
        duration = float(word_event["duration"])
        t_start = (
            int(onset * sample_freq) + offset_samples
        )  # Delay labels so they occur at same time as brain response
        t_end = int((onset + duration) * sample_freq) + offset_samples
        labels[t_start : t_end + 1] = 1.0

    return labels


def get_voiced_labels_gwilliams(events, phoneme_codes, raw, offset=0.0):
    sample_freq = raw.info["sfreq"]

    # Filter events with phoneme labels
    phoneme_events = events[
        ["'kind': 'phoneme'" in trial_type for trial_type in list(events["trial_type"])]
    ]

    phoneme_onsets = []
    labels = []

    bad_segments = 0
    for i, phoneme_event in phoneme_events.iterrows():
        trial_type = ast.literal_eval(phoneme_event["trial_type"])

        phoneme = trial_type["phoneme"].split("_")[0]  # Remove BIE indicators
        onset_samples = int(float(phoneme_event["onset"]) * sample_freq)
        duration_samples = int(float(phoneme_event["duration"]) * sample_freq)
        phonation = phoneme_codes[phoneme_codes["phoneme"] == phoneme][
            "phonation"
        ].item()

        # Check that we're not in a bad segment
        bad_phoneme = False
        for annot in raw.annotations:
            if "bad_segment" in annot["description"]:
                bad_onset = int(sample_freq * annot["onset"])
                bad_samples = int(sample_freq * annot["duration"])
                phone_end = onset_samples + duration_samples
                # Check phoneme onset is not in a bad segment
                if (onset_samples >= bad_onset) and (
                    onset_samples <= (bad_onset + bad_samples)
                ):
                    bad_phoneme = True
                elif (phone_end >= bad_onset) and (
                    phone_end <= (bad_onset + bad_samples)
                ):
                    bad_phoneme = True
        if bad_phoneme:
            bad_segments += 1
            continue

        # Label as voiced or unvoiced
        if phonation == "v":
            labels.append(1.0)
            phoneme_onsets.append(onset_samples)
        elif phonation == "uv":
            labels.append(0.0)
            phoneme_onsets.append(onset_samples)

    print(
        f"Found {bad_segments} (out of {len(phoneme_onsets) + bad_segments}) phonemes in bad segments while computing phoneme label onsets"
    )

    return phoneme_onsets, labels


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
