"""Kaggle Secret loader for the separate critic kernel (D7). Only for that kernel (internet on, never runs candidate code).
Never log or print the key values. The labels key_1..key_n are the only names the KeyScheduler sees."""

SECRET_LABEL = "NVIDIA_API_KEYS"
MAX_NUMBERED = 100


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


def load_kaggle_keys(label: str = SECRET_LABEL, client=None) -> dict:
    """Only for the separate critic kernel (D7, internet on, never runs candidate code). Never log or print the values.

    Two forms are read. Numbered secrets `<label>_1`, `<label>_2`, ... (one key each) are read in order,
    stopping at the first one that is missing, raises, or is blank; if at least one is found, their
    stripped values are joined with newlines and parsed by split_keys. Otherwise the single combined
    secret `<label>` is parsed by split_keys.
    """
    if client is None:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    values = []
    for i in range(1, MAX_NUMBERED + 1):
        try:
            v = client.get_secret("%s_%d" % (label, i))
        except Exception:
            break
        if not v or not v.strip():
            break
        values.append(v.strip())
    if values:
        return split_keys("\n".join(values))
    return split_keys(client.get_secret(label))
