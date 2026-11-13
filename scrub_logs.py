#!/usr/bin/env python3
"""Make text logs safe for a PUBLIC repo: copy each file with machine/person identifiers replaced, and withhold any
file in which an identifier or a secret is still found afterwards. Only needs python3 (stdlib).

  python3 scrub_logs.py --dest DIR [--extra STRING]... [--vocab-file F] [--secret-file F]... SRC=REL [SRC=REL ...]

  SRC=REL     copy SRC to DIR/REL. A training_args.bin is decoded (restricted unpickler, no torch) and written as JSON.
  --extra     an exact string to replace wherever it occurs, case-insensitive (user name, home directory, host names...).
              Strings shorter than 3 characters are ignored.
  --vocab-file  extra rules, one per line: <python regex><TAB><replacement>; also checked after scrubbing.
  --secret-file a file whose content is a secret (token / API key): a file containing it is withheld, never written.
              WANDB_API_KEY / HF_TOKEN / HUGGING_FACE_HUB_TOKEN from the environment are treated the same way.

What is replaced in every file:
  - absolute paths (/a/b/c.txt -> <path>/c.txt), e-mail addresses, IPv4 addresses
  - host names found in the logs themselves (host=..., "host": ..., NCCL line prefixes, MASTER_ADDR=..., nodelists)
  - user names / e-mails found in wandb metadata ("username", "email"), the current user, home directory, hostname
  - every --extra string and every --vocab-file rule
Withheld (not written; listed in the report by path only): a file that still contains any of these after scrubbing,
internal project words, a secret (incl. HF/wandb key patterns), or that is not text; also a file whose output
name (REL) names any of these. Project words are first replaced by built-in rules.
Exit status: 0 = every file written, 3 = some files withheld (the others are written), 2 = usage error.
"""
import argparse, getpass, io, json, os, pickle, re, socket, sys, zipfile

PATH_RE = re.compile(r"(?<![\w.~/-])/(?!/)(?:[\w.+@=,-]+/)+[\w.+@=,-]*")   # also after ":" (json:/a/b), not URLs
FS_ROOT_RE = re.compile(r"(?<![\w.~/<-])/(?:home|scratch|weka|users?|mnt|raid|lustre|gpfs|data|work|projects?)\b(?!/)", re.I)   # bare mount roots, e.g. -B /weka:/weka
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IP_RE = re.compile(r"(?<![\w.v-])(?<!==)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\w)(?!\.\d)")
GPU_UUID_RE = re.compile(r"\b(GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
WANDB_KEY_RE = re.compile(r"(?i)(?:wandb_api_key|api[_-]?key|password)[\"']?\s*[=:\s]\s*[\"']?(?:local-)?[0-9a-f]{40}\b")
HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{30,}")
WORDS_RE = re.compile(r"rebuttal|repro(?!duc|cess)|corl|\bco[-_ ]rl\b|anvil", re.I)
BUILTIN_RULES = [(re.compile(r"corlplus", re.I), "cogrpoplus"), (re.compile(r"\bco[-_ ]?rl(?![a-z])", re.I), "co-GRPO"),
                 (re.compile(r"corl", re.I), "cogrpo"), (re.compile(r"anvil", re.I), "<cluster>")]
HOST_PATTERNS = [
    re.compile(r"\bhost(?:name)?\s*[=:]\s*[\"']?([A-Za-z][\w.-]{2,})", re.I),
    re.compile(r"\"host\"\s*:\s*\"([^\"]{3,})\""),
    re.compile(r"(?m)^([A-Za-z][\w.-]{2,}):\d+:\d+ \[\d+\] NCCL"),
    re.compile(r"\b(?:MASTER_ADDR|SLURMD_NODENAME|SLURM_JOB_NODELIST|SLURM_NODELIST|SLURM_SUBMIT_HOST|NodeList|BatchHost)\s*[=:]\s*\"?([A-Za-z][\w.\-\[\],]{2,})"),
    re.compile(r"tcp://([A-Za-z][\w.-]{2,}):\d+"),
]
USER_PATTERNS = [re.compile(r"\"(?:username|email|user)\"\s*:\s*\"([^\"]{3,})\""),
                 re.compile(r"\b(?:USER|LOGNAME|SLURM_JOB_USER)\s*=\s*([\w.-]{3,})")]
NOT_IDENTIFIERS = {"localhost", "none", "null", "true", "false", "unknown", "0.0.0.0", "127.0.0.1"}


class Restricted(pickle.Unpickler):
    """Turns every class into a plain record; only harmless builtins are real. Never executes code from the file."""
    SAFE = {("builtins", n) for n in ("set", "frozenset", "slice", "complex", "bytearray", "range")} | \
           {("collections", "OrderedDict"), ("copyreg", "_reconstructor"), ("_codecs", "encode")} | \
           {("datetime", n) for n in ("datetime", "date", "time", "timedelta", "timezone")}

    def find_class(self, mod, name):
        if (mod, name) in self.SAFE:
            return super().find_class(mod, name)
        if (mod, name) == ("builtins", "object"):
            return object
        return type(name, (Record,), {"__module__": mod})

    def persistent_load(self, pid):
        return "<tensor>"


class Record:
    def __init__(self, *a, **k):
        self._args = list(a)

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"_state": st})


def plain(o, depth=0):
    if depth > 8:
        return repr(o)
    if isinstance(o, Record):
        d = {k: plain(v, depth + 1) for k, v in o.__dict__.items() if k != "_args" or v}
        if set(d) == {"_args"}:
            return d["_args"][0] if len(d["_args"]) == 1 else d["_args"]
        if set(d) == {"_value_"} or set(d) == {"_state"}:
            return next(iter(d.values()))
        return d
    if isinstance(o, dict):
        return {str(k): plain(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [plain(v, depth + 1) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def read_text(src):
    if os.path.basename(src) == "training_args.bin":
        z = zipfile.ZipFile(src)
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        obj = Restricted(io.BytesIO(z.read(name))).load()
        return json.dumps(plain(obj), indent=1, sort_keys=True, default=str) + "\n"
    raw = open(src, "rb").read()
    if b"\x00" in raw[:65536]:
        raise ValueError("not a text file")
    return raw.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--vocab-file")
    ap.add_argument("--secret-file", action="append", default=[])
    ap.add_argument("pairs", nargs="+", metavar="SRC=REL")
    a = ap.parse_args()
    pairs = []
    for p in a.pairs:
        src, sep, rel = p.rpartition("=")
        if not sep or not rel or rel.startswith("/") or ".." in rel.split("/") or not os.path.isfile(src):
            print("usage error: bad pair %r (need an existing SRC file and a relative REL)" % p, file=sys.stderr)
            sys.exit(2)
        pairs.append((src, rel))
    rules = []
    if a.vocab_file:
        for line in open(a.vocab_file, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.strip() and not line.lstrip().startswith("#"):
                pat, _, rep = line.partition("\t")
                rules.append((re.compile(pat, re.I), rep or "<redacted>"))
    secrets = []
    for f in a.secret_file:
        try:
            s = open(os.path.expanduser(f)).read().strip()
            if len(s) >= 8:
                secrets.append(s)
        except OSError:
            pass
    try:
        import netrc
        for host in ("api.wandb.ai", "huggingface.co"):
            auth = netrc.netrc().authenticators(host)
            if auth and auth[2] and len(auth[2]) >= 8:
                secrets.append(auth[2])
    except Exception:
        pass
    for k in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if len(os.environ.get(k, "")) >= 8:
            secrets.append(os.environ[k])
    texts, withheld = {}, []
    for src, rel in pairs:
        try:
            texts[rel] = read_text(src)
        except Exception as e:
            withheld.append((rel, "unreadable: %s" % type(e).__name__))

    # identifiers: given, about this machine, and found in the logs themselves
    ids = set(x for x in a.extra if len(x) >= 3)
    home = os.path.expanduser("~")
    for x in (getpass.getuser(), home, socket.gethostname(), socket.gethostname().split(".")[0], socket.getfqdn()):
        if x and len(x) >= 3:
            ids.add(x)
    for t in texts.values():
        for pat in HOST_PATTERNS + USER_PATTERNS:
            for m in pat.finditer(t):
                for h in re.split(r"[,\[\]]", m.group(1)):
                    h = h.strip().strip('"')
                    if len(h) >= 3 and h.lower() not in NOT_IDENTIFIERS and not h.isdigit():
                        ids.add(h)
                        if "." in h and not EMAIL_RE.fullmatch(h) and len(h.split(".")[0]) >= 3:
                            ids.add(h.split(".")[0])
                        if "@" in h and len(h.split("@")[0]) >= 3:
                            ids.add(h.split("@")[0])
    id_re = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(x) for x in sorted(ids, key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.I) if ids else None

    def scrub(t):
        t = PATH_RE.sub(lambda m: "<path>/" + m.group(0).rstrip("/").rsplit("/", 1)[-1] if not m.group(0).endswith("/") else "<path>/", t)
        t = FS_ROOT_RE.sub("<path>", t)
        t = EMAIL_RE.sub("<email>", t)
        t = IP_RE.sub(lambda m: m.group(0) if m.group(0) in NOT_IDENTIFIERS else "<ip>", t)
        t = GPU_UUID_RE.sub(lambda m: m.group(1) + "-<uuid>", t)
        for pat, rep in rules + BUILTIN_RULES:
            t = pat.sub(rep, t)
        if id_re:
            t = id_re.sub("<redacted>", t)
        return t

    def residue(t):
        found = []
        if any(s in t for s in secrets) or HF_TOKEN_RE.search(t) or WANDB_KEY_RE.search(t):
            return ["secret"]
        if WORDS_RE.search(t):
            found.append("project word")
        if id_re and id_re.search(t):
            found.append("identifier")
        if any(p.search(t) for p, _ in rules):
            found.append("vocab rule")
        if re.search(r"(?<![\w<>])/(?:home|scratch|weka|users?|mnt|raid|lustre|gpfs)\b", t, re.I):
            found.append("path")
        return found

    written = []
    for rel, t in sorted(texts.items()):
        if WORDS_RE.search(rel) or (id_re and id_re.search(rel)) or any(p.search(rel) for p, _ in rules):
            withheld.append((rel, "the output file name itself names a project word or identifier"))
            continue
        if any(s in t for s in secrets) or HF_TOKEN_RE.search(t) or WANDB_KEY_RE.search(t):
            withheld.append((rel, "secret"))
            continue
        out = scrub(t)
        left = residue(out)
        if left:
            withheld.append((rel, "still contains: " + ", ".join(left)))
            continue
        dst = os.path.join(a.dest, rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(out)
        written.append(rel)
    print("[scrub] %d written, %d withheld -> %s" % (len(written), len(withheld), a.dest))
    for rel, why in sorted(withheld):
        print("[scrub] WITHHELD %s (%s)" % (rel, why))
    sys.exit(3 if withheld else 0)


if __name__ == "__main__":
    main()
