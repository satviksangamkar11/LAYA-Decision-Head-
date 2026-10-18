"""[Qwen2.5-Coder-1.5B local run] Tests of the new backbone profile (configs/backbones_v2.toml), the additive v1-then-v2 loader, the non-thinking path of HFQwenBackend and build_local_backend.
No GPU, no network: the model/tokenizer/torch are labelled stubs (LOGIC-ONLY) except the manifest test, which hashes the REAL downloaded files (skipped and listed when the model dir is absent).
Run from the repo root: python evidence_ladder/models/tests/qwen25_tests.py   (prints {"tests","passed","failed","skipped"}; exit 1 on failure)."""
import hashlib
import json
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evidence_ladder.common.interfaces import GenerationRequest, NotAuthorized   # noqa: E402
from evidence_ladder.controller import generation as GEN, l2 as L2               # noqa: E402
from evidence_ladder.models import backbone as BB, qwen as Q                     # noqa: E402

PID = "qwen2.5-coder-1.5b-instruct"
TESTS, SKIPPED = {}, {}


class Skip(Exception):
    pass


def test(f):
    TESTS[f.__name__] = f
    return f


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected %s" % (exc,))


# ------------------------------------------------------------------ profile
@test
def profile_is_registered_with_the_requested_shape():
    P = BB.get(PID)
    assert P.repo == "Qwen/Qwen2.5-Coder-1.5B-Instruct" and P.revision == "2e1fd397ee46e1388853d2af2c993145b0f1098a" and P.hidden_size == 1536 and P.num_layers == 28
    assert P.depths == (14, 19, 23) and P.dtype == "float16" and P.local_device_policy == "resident" and P.enable_thinking is False and P.head_compatible == ()
    assert dict(P.generation) == {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1}
    gc = json.loads((ROOT / "models" / "qwen2.5-coder-1.5b-instruct" / "generation_config.json").read_text("utf-8")) if (ROOT / "models" / "qwen2.5-coder-1.5b-instruct").is_dir() else None
    if gc:      # the sampling values are the model's own generation_config.json (repetition_penalty 1.1, not 1.05)
        assert all(gc[k] == P.generation[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty"))
    assert P.sha256 == BB.get(PID).sha256 and len(P.sha256) == 64


@test
def head_guard_refuses_this_models_states():
    raises(BB.ProfileError, BB.require_head_compatible, BB.get(PID))
    BB.require_head_compatible(BB.get())      # the 4B stays consumable by the frozen adapters


@test
def v1_profiles_are_unchanged_and_v2_is_additive():
    v1_only = BB.load_all(ROOT / BB.CONFIG_REL)
    both = BB.load_all()
    for k, v in v1_only.items():
        assert both[k] == v and both[k].sha256 == v.sha256, k            # 4B (builtin cross-check) and 0.6B identical with or without v2
    assert set(both) == set(v1_only) | {PID}


@test
def v2_may_not_redefine_a_v1_profile_nor_the_4b():
    tmp = Path(tempfile.mkdtemp(prefix="bbv2-"))
    v1 = (ROOT / BB.CONFIG_REL).read_text("utf-8")
    (tmp / "configs").mkdir()
    (tmp / "configs" / "backbones_v1.toml").write_text(v1, "utf-8")
    v2_text = (ROOT / BB.CONFIG_V2_REL).read_text("utf-8")
    (tmp / "configs" / "backbones_v2.toml").write_text(v2_text + '\n[backbone."qwen3-0.6b"]\n' + v1.split('[backbone."qwen3-0.6b"]')[1], "utf-8")
    old = BB.REPO_ROOT
    BB.REPO_ROOT = tmp
    try:
        e = raises(BB.ProfileError, BB.load_all)
        assert "redefines" in str(e)
        (tmp / "configs" / "backbones_v2.toml").write_text(v2_text.replace('revision = "2e1fd', 'revision = "2e1fd') + '\n[backbone."qwen3-4b-thinking-2507"]\n' + v1.split('[backbone."qwen3-4b-thinking-2507"]')[1].split("[backbone.")[0].replace("0.6", "0.7"), "utf-8")
        raises(BB.ProfileError, BB.load_all)                              # a changed 4B entry is refused by the embedded-copy cross-check
    finally:
        BB.REPO_ROOT = old


# ------------------------------------------------------------------ manifest of the real download
@test
def manifest_matches_every_downloaded_file():
    d = ROOT / "models" / "qwen2.5-coder-1.5b-instruct"
    if not (d / "MANIFEST_pinned.json").is_file():
        raise Skip("NOT_TESTED: model directory absent")
    m = json.loads((d / "MANIFEST_pinned.json").read_text("utf-8"))
    P = BB.get(PID)
    assert m["repo"] == P.repo and m["revision"] == P.revision and m["license"] == "apache-2.0"
    assert set(m["files"]) >= {"model.safetensors", "config.json", "tokenizer.json", "vocab.json", "merges.txt", "generation_config.json", "tokenizer_config.json", "LICENSE"}
    for rel, rec in m["files"].items():
        h = hashlib.sha256()
        with open(d / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == rec["sha256"] and (d / rel).stat().st_size == rec["bytes"], rel
    assert m["files"]["model.safetensors"]["lfs_sha256_matches"] is True
    cfg = json.loads((d / "config.json").read_text("utf-8"))
    assert cfg["hidden_size"] == P.hidden_size and cfg["num_hidden_layers"] == P.num_layers


# ------------------------------------------------------------------ backend (stub torch / tokenizer / model: LOGIC-ONLY)
class _Arr:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, ix):
        if isinstance(ix, tuple):
            r, sl = ix
            row = self.rows[r][sl]
            a = _Arr([row])
            a.shape = (len(row),)
            return a
        return self.rows[ix]


class _Enc(dict):
    def to(self, dev):
        return self


def _stack(log, answer, template="plain"):
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"
    torch.manual_seed = lambda s: log.append(("seed", s))
    torch.cuda = types.SimpleNamespace(manual_seed_all=lambda s: log.append(("cuda_seed", s)), get_device_name=lambda i: "STUB GPU")

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False
    torch.no_grad = NoGrad
    tf = types.ModuleType("transformers")

    class Tok:
        eos_token_id = 2
        chat_template = template

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            log.append(("template_kwargs", tuple(sorted(kw))))
            return "<chat>" + msgs[0]["content"]

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            return _Enc(input_ids=_Arr([list(range(5))]))

        def convert_tokens_to_ids(self, t):
            return 3

        def decode(self, row, skip_special_tokens=True):
            return answer

        @staticmethod
        def from_pretrained(src, revision=None):
            return Tok()

    class Model:
        device = "cuda"

        def eval(self):
            return self

        def generate(self, **kw):
            log.append(("generate", tuple(sorted(k for k in kw if k not in ("input_ids",))), kw.get("repetition_penalty"), kw.get("do_sample")))
            return _Arr([list(range(5)) + [9, 9, 9]])

        @staticmethod
        def from_pretrained(src, revision=None, torch_dtype=None, device_map=None, low_cpu_mem_usage=None):
            log.append(("model_load", device_map, low_cpu_mem_usage))
            return Model()
    tf.AutoTokenizer, tf.AutoModelForCausalLM = Tok, Model
    return torch, tf


class _Stub:
    def __init__(self, answer, template="plain"):
        self.log = []
        self.torch, self.tf = _stack(self.log, answer, template)

    def __enter__(self):
        self.saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
        sys.modules["torch"], sys.modules["transformers"] = self.torch, self.tf
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _backend(prof_id=PID, **kw):
    tmp = Path(tempfile.mkdtemp(prefix="q25-"))
    (tmp / "tokenizer.json").write_text("{}", "utf-8")
    be = Q.HFQwenBackend(authorized=True, model_dir=str(tmp), env={Q.HFQwenBackend.INFERENCE_ENV: "1"}, profile=BB.get(prof_id), local=True,
                         verify_files={"tokenizer.json": hashlib.sha256(b"{}").hexdigest()}, **kw)
    return be, tmp


CODE_ANSWER = "Here is the solution.\n```python\ndef solve(data):\n    return data['x'] * 2\n```"


@test
def non_thinking_profile_generates_and_ladder_parsers_see_a_closed_think_block():
    with _Stub(CODE_ANSWER) as st:
        be, _ = _backend()
        gens = be.generate(GenerationRequest("T", "PROMPT", 2, (5, 6), 50, "sample", True, "qwen_sample_v2"))       # the 'thinking-only' refusal does not fire for this profile
        g = gens[0]
        assert g.text.startswith(Q.NONTHINK_PREFIX) and g.text.endswith(CODE_ANSWER) and g.extract_status == "ok" and g.code.strip().startswith("def solve")      # candidate extraction works
        assert [x.seed for x in gens] == [5, 6] and g.model == "Qwen/Qwen2.5-Coder-1.5B-Instruct" and g.model_revision == BB.get(PID).revision
        # L2 test text: the strict parser needs a closed think block
        tests = '[{"input": {"x": 2}, "expected": 4, "quote": "twice the value"}]'
        st.tf.AutoTokenizer.decode = lambda self, row, skip_special_tokens=True: tests
        gt = be.generate(GenerationRequest("T", "P2", 1, (1,), 50, "tests", True, "qwen_tests_v1"))[0]
        assert L2.parse_checks(gt.text)["status"] == "ok"
        # action proposals
        acts = '[{"name":"Bash","arguments":{"command":"ls"}}]'
        st.tf.AutoTokenizer.decode = lambda self, row, skip_special_tokens=True: acts
        ga = be.generate(GenerationRequest("T", "P3", 1, (1,), 50, "sample", True, GEN.ACTION_TEMPLATE_ID))[0]
        assert GEN.parse_actions(ga.text)["status"] == "ok"
        # nothing was passed to the chat template (the Qwen2.5 template has no enable_thinking switch)
        assert all(k == ("template_kwargs", ()) for k in st.log if k[0] == "template_kwargs")


@test
def thinking_profiles_are_not_weakened():
    # the strict parsers still reject a thinking answer that never closed, and a text WITHOUT the prefix from a non-thinking model would be rejected: only the backend adds the prefix
    assert GEN.parse_actions('[{"name":"Bash","arguments":{"command":"ls"}}]')["status"] == "no_think_close"
    assert L2.parse_checks("[]")["status"] == "no_think_close"
    with _Stub("reasoning</think>\n```python\ndef solve(data):\n    return 1\n```", template="... enable_thinking ...") as st:
        be, _ = _backend("qwen3-0.6b")
        g = be.generate(GenerationRequest("T", "PROMPT", 1, (1,), 50, "sample", True, "qwen_sample_v2"))[0]
        assert not g.text.startswith(Q.NONTHINK_PREFIX) and g.text.startswith("reasoning")        # no prefix for a thinking profile
        assert ("template_kwargs", ("enable_thinking",)) in st.log                                # a template that has the switch still receives it
    with _Stub("x") as st:
        be4, _ = _backend("qwen3-4b-thinking-2507")
        raises(ValueError, be4.generate, GenerationRequest("T", "p", 1, (1,), 10, thinking=False))


@test
def sampling_profile_is_never_cached_and_seeds_are_recorded():
    with _Stub(CODE_ANSWER) as st:
        be, _ = _backend()
        req = GenerationRequest("T", "SAME PROMPT", 3, (7, 8, 9), 50, "sample", True, "qwen_sample_v2")
        gens = be.generate(req)
        be.generate(req)
        assert [g.seed for g in gens] == [7, 8, 9] and sum(1 for x in st.log if x[0] == "generate") == 6      # six real generations, no greedy cache reuse
        seeds = [x[1] for x in st.log if x[0] == "seed"]
        assert seeds == [7, 8, 9, 7, 8, 9]
        assert all(x[2] == 1.1 and x[3] is True for x in st.log if x[0] == "generate")                          # repetition_penalty and do_sample reach generate()
        assert all(g.tokens_generated == 3 for g in gens) and be._greedy == {}
    with _Stub(CODE_ANSWER) as st:                                                                              # contrast: a greedy profile does cache
        be, _ = _backend("qwen3-0.6b")
        be.generate(GenerationRequest("T", "SAME PROMPT", 2, (1, 2), 50, "sample", True, "qwen_sample_v2"))
        assert sum(1 for x in st.log if x[0] == "generate") == 1 and len(be._greedy) == 1


@test
def model_info_carries_identity_snapshot_dtype_device_and_sampling():
    with _Stub(CODE_ANSWER):
        be, _ = _backend()
        info0 = be.model_info()
        assert info0["profile_id"] == PID and info0["profile_sha256"] == BB.get(PID).sha256 and info0["local"] is True and info0["dtype"] == "fp16"
        assert info0["thinking"] is False and info0["nonthinking_empty_think_prefix"] is True and info0["device"] is None and info0["snapshot_verified"] is False
        be.generate(GenerationRequest("T", "p", 1, (1,), 10, "sample", True, "qwen_sample_v2"))
        info = be.model_info()
        assert info["snapshot_verified"] is True and info["device"] == "STUB GPU" and info["pinned_files"] == ["tokenizer.json"]
        assert info["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1} and info["do_sample"] is True and info["seed_recorded"] is True
        assert info["revision"] == BB.get(PID).revision and info["repo"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        be4, _ = _backend("qwen3-4b-thinking-2507")
        i4 = be4.model_info()
        assert i4["thinking"] is True and i4["nonthinking_empty_think_prefix"] is False and i4["sampling"] == Q.SAMPLING       # 4B values unchanged


@test
def local_load_goes_straight_to_the_gpu_and_a_corrupt_snapshot_is_refused():
    with _Stub(CODE_ANSWER) as st:
        be, tmp = _backend()
        be.generate(GenerationRequest("T", "p", 1, (1,), 10, "sample", True, "qwen_sample_v2"))
        assert ("model_load", "cuda", True) in st.log
        bad = Q.HFQwenBackend(authorized=True, model_dir=str(tmp), env={Q.HFQwenBackend.INFERENCE_ENV: "1"}, profile=BB.get(PID), local=True, verify_files={"tokenizer.json": "0" * 64})
        raises(Q.SnapshotMismatch, bad.generate, GenerationRequest("T", "p", 1, (1,), 10))
    raises(NotAuthorized, Q.HFQwenBackend, profile=BB.get(PID), local=True)                                    # still gated: authorized=True required


@test
def build_local_backend_signature_and_think_policy_guard():
    import inspect
    sig = inspect.signature(Q.build_local_backend)
    assert list(sig.parameters) == ["profile_id", "log_path", "think_policy"] and sig.parameters["log_path"].default is None
    raises(BB.ProfileError, Q.build_local_backend, "no-such-profile")
    if not (ROOT / "models" / "qwen2.5-coder-1.5b-instruct").is_dir():
        raise Skip("NOT_TESTED: model directory absent")
    be = Q.build_local_backend(PID)
    assert isinstance(be, Q.HFQwenBackend) and be.local and be.profile.id == PID and be.snapshot_verified is False
    assert "model.safetensors" in be.verify_files and "config.json" in be.verify_files and "MANIFEST_pinned.json" not in be.verify_files
    assert be.model_info()["pinned_files"] == sorted(be.verify_files) and os.environ.get(Q.HFQwenBackend.INFERENCE_ENV) == "1"
    raises(ValueError, Q.build_local_backend, PID, None, {"think_budget": 512})                                # a think budget makes no sense for a non-thinking model


def main():
    failed = {}
    for name, f in TESTS.items():
        try:
            f()
        except Skip as s:
            SKIPPED[name] = str(s)
        except (Exception, SystemExit):      # noqa: BLE001
            failed[name] = traceback.format_exc()[-1100:]
    print(json.dumps({"tests": len(TESTS), "passed": len(TESTS) - len(failed) - len(SKIPPED), "failed": failed, "skipped": SKIPPED}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
