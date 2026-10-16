"""Evidence-Ladder critic kernel v1 (D7): internet on, NVIDIA keys as Kaggle Secrets NVIDIA_API_KEYS_1..n, never runs candidate code,
holds no grading material. Part 1 counts the visible keys (prints NO key values). Part 2 is the FC-2 probe: two tiny GLM calls on key_1
(pilot GO 2026-10-16). The loader is a copy of evidence_ladder/models/kaggle_keys.py (tested on Kaggle, run 8)."""
import json
from pathlib import Path

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


res = {"kernel": "evidence_ladder_critic_v1", "check": "key_count_only", "nvidia_calls": 0, "expected_keys": 8}
import os, socket
try:                                       # diagnostic: is internet on? (secrets service needs it)
    socket.create_connection(("www.kaggle.com", 443), timeout=10).close(); res["internet"] = True
except Exception as e:
    res["internet"] = False; res["internet_error"] = type(e).__name__
res["secret_env_present"] = bool(os.environ.get("KAGGLE_USER_SECRETS_TOKEN"))   # set only when secrets are attached; value never read
try:                                       # diagnostic: why does label _1 fail? (message names the label; anything key-like is cut)
    from kaggle_secrets import UserSecretsClient
    UserSecretsClient().get_secret(SECRET_LABEL + "_1"); res["label_1"] = "readable"
except Exception as e:
    m = str(e).split("nvapi")[0][:300]; res["label_1"] = "%s: %s" % (type(e).__name__, m)
    c = e.__cause__                        # kaggle_secrets wraps HTTPError/URLError; the code tells "not attached" (4xx) from network
    res["label_1_cause"] = "%s %s %s" % (type(c).__name__, getattr(c, "code", ""), str(getattr(c, "reason", "")).split("nvapi")[0][:200])
try:
    keys = load_kaggle_keys()
    res.update(ok=True, n_keys=len(keys), labels=list(keys), all_nonempty=all(bool(v) for v in keys.values()),
               distinct=len(set(keys.values())) == len(keys))
except Exception as e:
    res.update(ok=False, error_type=type(e).__name__)   # message dropped: it could carry secret text
Path("/kaggle/working/KEY_CHECK.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))

# ---- FC-2 (pilot GO 2026-10-16): does z-ai/glm-5.3-flash accept reasoning_effort? Two tiny streamed calls on key_1 ("max", then "low" as control).
# Same setting the project used for z-ai/glm-5.3 (tools/judge_qual_v1.py glm_max64, accepted 2026-10-11). Records status, finish reason, usage and
# the LENGTH of reasoning/answer text only; no key value or response text is written.
import time, urllib.request, urllib.error

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
PROMPT = "Reply with the single word OK."


def probe(key, effort):
    body = dict(model="z-ai/glm-5.3-flash", messages=[{"role": "user", "content": PROMPT}], temperature=0.5, top_p=1.0, max_tokens=4096,
                stream=True, reasoning_effort=effort)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                                                               "Accept": "text/event-stream"})
    out = {"reasoning_effort": effort, "reasoning_chars": 0, "answer_chars": 0, "finish_reason": None, "usage": None}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=290) as r:
            out["status"] = r.status
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    out["reasoning_chars"] += len(d.get("reasoning_content") or d.get("reasoning") or "")
                    out["answer_chars"] += len(d.get("content") or "")
                    out["finish_reason"] = ch.get("finish_reason") or out["finish_reason"]
                if ev.get("usage"):
                    out["usage"] = ev["usage"]
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error_body_chars"] = len(e.read() or b"")
    except Exception as e:
        out["status"] = None
        out["error_type"] = type(e).__name__
    out["seconds"] = round(time.time() - t0, 1)
    return out


fc2 = {"check": "FC-2", "model": "z-ai/glm-5.3-flash", "calls": []}
if res.get("ok"):
    for effort in ("max", "low"):
        fc2["calls"].append(probe(keys["key_1"], effort))
fc2["nvidia_calls"] = len(fc2["calls"])
fc2["accepted"] = {c["reasoning_effort"]: c.get("status") == 200 for c in fc2["calls"]}
Path("/kaggle/working/FC2.json").write_text(json.dumps(fc2, indent=1))
print(json.dumps(fc2))
