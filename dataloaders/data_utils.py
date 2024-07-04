from pathlib import Path

DATA_PATH = Path("/data/engs-pnpl/lina4368")


def get_key_from_batch_identifier(batch_identifier: dict) -> str:
    identifier = {k: batch_identifier[k][0] for k in batch_identifier.keys()}
    return get_key_from_identifier(identifier)


def get_key_from_identifier(identifier: dict) -> str:
    key = f"dat={identifier['dataset']}"
    if "subject" in identifier:
        key += f"_sub={identifier['subject']}"
    return key

def get_dset_encoding(dataset):
    if dataset == "armeni2022":
        return 0
    elif dataset == "gwilliams2022":
        return 1
    elif dataset == "schoffelen2019":
        return 2
    elif dataset == "shafto2014":
        return 3
    else:
        raise ValueError("Dataset not supported")