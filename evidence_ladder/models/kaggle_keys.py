"""Kaggle Secret loader for the separate critic kernel (D7). Only for that kernel (internet on, never runs candidate code).
Never log or print the key values. The labels key_1..key_n are the only names the KeyScheduler sees."""

SECRET_LABEL = "NVIDIA_API_KEYS"


def split_keys(text: str) -> dict:
    keys, seen = [], set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for part in line.split(","):
            k = part.strip()
            if not k:
                continue
            if any(c.isspace() for c in k):
                raise ValueError("line %d is malformed" % lineno)
            if k not in seen:
                seen.add(k)
                keys.append(k)
    if not keys:
        raise ValueError("no keys in secret")
    return {"key_%d" % i: k for i, k in enumerate(keys, 1)}


def load_kaggle_keys(label: str = SECRET_LABEL) -> dict:
    """Only for the separate critic kernel (D7, internet on, never runs candidate code). Never log or print the values."""
    from kaggle_secrets import UserSecretsClient
    return split_keys(UserSecretsClient().get_secret(label))
