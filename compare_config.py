"""Field-by-field diff of a new run's resolved training arguments against the paper run's.

Usage: python compare_config.py <NEW> <REF>
  NEW : a run directory (TTRL) or a group directory (Co-RL+: <run>/group_A, <run>/group_B). The newest
        checkpoint-*/training_args.bin inside it is used (a smoke run saves checkpoint-2); if there is none,
        <dir>/training_args.bin; if there is none either, the offline wandb config (**/files/config.yaml).
        Or give the file directly: targs:<path/to/training_args.bin>
  REF : targs:<path/to/training_args.bin>                  (paper run's training_args.bin, e.g. refs in this kit)
        json:<path/to/ref_config_*.json>                    (paper run's wandb config exported into this kit)
        wandb:<run_id>  or  wandb:<entity/project/run_id>   (needs a wandb key; default project
                                                             logan-yang2002-johns-hopkins-university/Co-learning)
Output: (1) fields present on both sides whose values differ, (2) fields only in the paper record,
        (3) fields only in the new run. Enum spellings (IntervalStrategy.STEPS / 'steps' / STEPS) are normalised.
Needs PyYAML only for the wandb-config fallback; run it with the kit's env python, e.g. envs/collab/venv/bin/python.
"""
import glob, io, json, os, pickle, re, sys, zipfile

IGNORE = {"output_dir", "logging_dir", "run_name", "run_config", "_wandb", "hub_model_id", "push_to_hub_token", "hub_token",
          "_n_gpu", "local_rank", "world_size", "process_index", "local_process_index", "distributed_state",
          "deepspeed_plugin", "hf_deepspeed_config", "accelerator_config", "ddp_backend", "rendezvous_dir", "_setup_devices"}


class _Any:
    def __init__(self, *a, **k): self._args = a
    def __repr__(self): return f"{type(self).__name__}{getattr(self, '_args', ())}"
    def __setstate__(self, st):
        if isinstance(st, dict): self.__dict__.update(st)


class _EnumRef:  # result of pickled getattr(EnumClass, "MEMBER")
    def __init__(self, cls, member): self.cls, self.member = cls, member


def _getattr(obj, name, *default):
    if isinstance(obj, type) and issubclass(obj, _Any):
        return _EnumRef(obj.__name__, name)
    return getattr(obj, name, *default)


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if name == "getattr":  # enums pickled as getattr(EnumClass, "member"), whatever module name the pickle uses
            return _getattr
        if mod.split(".")[0] in ("builtins", "collections", "datetime", "enum"):
            try: return super().find_class(mod, name)
            except Exception: pass
        return type(name, (_Any,), {})


def load_targs(path):
    raw = open(path, "rb").read()
    try:
        z = zipfile.ZipFile(io.BytesIO(raw)); data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
    except zipfile.BadZipFile:
        data = raw
    return dict(_U(io.BytesIO(data)).load().__dict__)


def load_new(spec):
    if spec.startswith("targs:"):
        return load_targs(spec[6:]), [], spec[6:]
    if os.path.isfile(spec):
        return load_targs(spec), [], spec
    cks = sorted(glob.glob(os.path.join(spec, "checkpoint-*", "training_args.bin")),
                 key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    for p in cks[-1:] + [os.path.join(spec, "training_args.bin")]:
        if os.path.isfile(p):
            return load_targs(p), [], p
    cfgs = sorted(glob.glob(os.path.join(spec, "**/files/config.yaml"), recursive=True))
    if not cfgs:
        sys.exit(f"no training_args.bin and no offline wandb config under {spec}")
    import yaml
    c = yaml.safe_load(open(cfgs[-1]))
    cfg = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in c.items()}
    meta = os.path.join(os.path.dirname(cfgs[-1]), "wandb-metadata.json")
    args = json.load(open(meta)).get("args", []) if os.path.exists(meta) else []
    return cfg, args, cfgs[-1]


def load_ref(spec):
    kind, val = spec.split(":", 1)
    if kind == "targs":
        return load_targs(val), []
    if kind == "json":
        d = json.load(open(val))
        return dict(d["config"]), d.get("args", [])
    if kind == "wandb":
        import wandb
        path = val if val.count("/") == 2 else f"logan-yang2002-johns-hopkins-university/Co-learning/{val}"
        r = wandb.Api(timeout=90).run(path)
        return dict(r.config), (r.metadata or {}).get("args", [])
    sys.exit(f"unknown reference kind {kind!r} (use targs:, json: or wandb:)")


def norm(v):
    if isinstance(v, _EnumRef): return v.member.lower()
    if isinstance(v, _Any):
        a = getattr(v, "_args", ())
        if len(a) == 1 and isinstance(a[0], (str, int, float)): return norm(a[0])
        return f"{type(v).__name__}{norm(list(a))}"
    if isinstance(v, bool) or v is None: return str(v).lower()
    if isinstance(v, int): return v
    if isinstance(v, float): return int(v) if v == int(v) else v
    if isinstance(v, (list, tuple)): return [norm(x) for x in v]
    if isinstance(v, dict): return {k: norm(x) for k, x in sorted(v.items())}
    s = str(v)
    s = re.sub(r"^(IntervalStrategy|SchedulerType|SaveStrategy|OptimizerNames|HubStrategy|EvaluationStrategy|FSDPOption)\.", "", s)
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", s): s = s.lower()          # enum member names: STEPS -> steps
    if s in ("True", "False", "None"): s = s.lower()
    return s


def main():
    if len(sys.argv) != 3: sys.exit(__doc__)
    new_cfg, new_args, path = load_new(sys.argv[1])
    ref_cfg, ref_args = load_ref(sys.argv[2])
    print(f"new: {path}  ({len(new_cfg)} fields)\nref: {sys.argv[2]}  ({len(ref_cfg)} fields)")
    shared = sorted((set(new_cfg) & set(ref_cfg)) - IGNORE)
    diffs = [(k, ref_cfg[k], new_cfg[k]) for k in shared if norm(ref_cfg[k]) != norm(new_cfg[k])]
    print(f"\n== (1) {len(diffs)} fields on both sides with different values ==")
    for k, a, b in diffs:
        print(f"  {k:42s} paper={str(norm(a))[:60]:62s} new={str(norm(b))[:60]}")
    only_ref = sorted(set(ref_cfg) - set(new_cfg) - IGNORE)
    only_new = sorted(set(new_cfg) - set(ref_cfg) - IGNORE)
    print(f"\n== (2) {len(only_ref)} fields only in the paper record ==\n  " + (", ".join(only_ref) if only_ref else "(none)"))
    print(f"\n== (3) {len(only_new)} fields only in the new run ==\n  " + (", ".join(only_new) if only_new else "(none)"))
    if ref_args and new_args:
        def pairs(a):
            out, i = {}, 0
            while i < len(a):
                if a[i].startswith("--"):
                    val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else "<flag>"
                    out[a[i]] = val; i += 2 if val != "<flag>" else 1
                else:
                    i += 1
            return out
        ra, na = pairs(ref_args), pairs(new_args)
        ad = [(k, ra.get(k, "<absent>"), na.get(k, "<absent>")) for k in sorted(set(ra) | set(na)) if ra.get(k) != na.get(k)]
        print(f"\n== {len(ad)} differing command-line args ==")
        for k, a, b in ad:
            print(f"  {k:36s} paper={str(a)[:70]:72s} new={str(b)[:70]}")


if __name__ == "__main__":
    main()
