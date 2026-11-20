"""Keep OUT (a fast local folder) synchronised with persistent storage across Colab sessions.

Training writes only to the local OUT folder (fast, no network file system in the training loop). This module
copies new or changed result files to persistent storage, and copies them back at the start of a new session.

Backend, chosen by the OUT_SYNC environment variable (set by the Colab notebook):
    hf:<user>/<repo>   private Hugging Face *dataset* repository (token from the HF_TOKEN environment variable)
    dir:<path>         any folder, e.g. a mounted Google Drive folder (the fallback)
    unset or "none"    no syncing (results live only on the local disk)

What is synced (paths relative to OUT):
    eager  (pulled at session start): split/, env/, figures/, results/, runs/**/ (logs, configs, metrics, CSVs),
           data_cache/train32.pt, test32.pt, fomixup_masks.pt
    lazy   (pulled only when a script needs them, via ensure()): runs/**/checkpoints/*.pt, data_cache/gtsrb_c/*
Scripts push automatically (throttled to one push per OUT_SYNC_INTERVAL seconds, default 600, during training
and a final blocking push at the end); the notebook also calls `python -m src.storage push` after every step.

CLI:
    python -m src.storage pull   --out OUT     # copy remote files that are missing locally (never overwrites)
    python -m src.storage push   --out OUT     # upload new/changed files
    python -m src.storage status --out OUT     # show backend and what would be pushed
"""
import argparse
import fnmatch
import json
import os
import shutil
import threading
import time

INCLUDE = [
    "split/*", "env/*", "figures/*", "figures/*/*", "results/*", "runs/*/*/*", "runs/*/*/checkpoints/*.pt",
    "data_cache/train32.pt", "data_cache/test32.pt", "data_cache/fomixup_masks.pt",
    "data_cache/gtsrb_c/*.npy",
]
LAZY = ["runs/*/*/checkpoints/*.pt", "data_cache/gtsrb_c/*"]
EXCLUDE = ["*.tmp", "*.tmp.npy", "*.tmp.*", ".sync*", "*/.sync*", "*.tmp*"]
MANIFEST = ".sync_manifest.json"


def _match(rel, patterns):
    return any(fnmatch.fnmatchcase(rel, p) for p in patterns)


def _log(msg):
    print(f"[sync] {msg}", flush=True)


def selected_files(out):
    """Relative paths (forward slashes) of every local file that belongs in persistent storage."""
    rels = []
    for root, dirs, files in os.walk(out):
        dirs[:] = [d for d in dirs if not d.startswith(".sync")]
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), out).replace(os.sep, "/")
            if _match(rel, INCLUDE) and not _match(rel, EXCLUDE) and not os.path.basename(rel).startswith("."):
                rels.append(rel)
    return sorted(rels)


# ----------------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------------
class DirBackend:
    """Copy files to/from a folder (e.g. /content/drive/MyDrive/gtsrb_freq_robust)."""

    def __init__(self, target):
        self.target = target
        self.name = f"dir:{target}"
        os.makedirs(target, exist_ok=True)

    def upload(self, out, rels):
        for rel in rels:
            dst = os.path.join(self.target, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".tmp"
            shutil.copy2(os.path.join(out, rel), tmp)
            os.replace(tmp, dst)

    def remote_files(self):
        return selected_files(self.target)

    def download(self, rels, out):
        for rel in rels:
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(self.target, rel), dst + ".tmp")
            os.replace(dst + ".tmp", dst)

    def exists(self, rel):
        return os.path.isfile(os.path.join(self.target, rel))

    def squash(self):
        pass


class HFBackend:
    """Private Hugging Face dataset repository. Authentication: HF_TOKEN environment variable."""

    def __init__(self, repo_id):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.name = f"hf:{repo_id}"
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def upload(self, out, rels):
        # stage hard links (or copies) so files replaced during the upload cannot change under the hasher
        stage = os.path.join(out, f".sync_stage_{os.getpid()}_{int(time.time() * 1000)}")
        try:
            for rel in rels:
                dst = os.path.join(stage, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    os.link(os.path.join(out, rel), dst)
                except OSError:
                    shutil.copy2(os.path.join(out, rel), dst)
            self.api.upload_folder(repo_id=self.repo_id, repo_type="dataset", folder_path=stage, path_in_repo="",
                                   commit_message=f"sync {len(rels)} files")
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def remote_files(self):
        return [f for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
                if _match(f, INCLUDE) and not _match(f, EXCLUDE)]

    def download(self, rels, out):
        from huggingface_hub import hf_hub_download
        stage = os.path.join(out, f".sync_pull_{os.getpid()}")
        try:
            for rel in rels:
                p = hf_hub_download(self.repo_id, rel, repo_type="dataset", local_dir=stage)
                dst = os.path.join(out, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(p, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def exists(self, rel):
        return self.api.file_exists(self.repo_id, rel, repo_type="dataset")

    def squash(self):
        """Collapse the repository history so superseded checkpoints stop counting towards storage."""
        self.api.super_squash_history(self.repo_id, repo_type="dataset",
                                      commit_message="results (history squashed)")


def backend_from_env():
    spec = os.environ.get("OUT_SYNC", "").strip()
    if not spec or spec.lower() == "none":
        return None
    kind, _, arg = spec.partition(":")
    if kind == "hf":
        return HFBackend(arg)
    if kind == "dir":
        return DirBackend(arg)
    raise ValueError(f"OUT_SYNC must be 'hf:<user>/<repo>', 'dir:<path>' or 'none', got {spec!r}")


# ----------------------------------------------------------------------------------------------
# Syncer: change tracking, throttled background pushes, lazy fetches
# ----------------------------------------------------------------------------------------------
class Syncer:
    def __init__(self, out, backend, interval=None):
        self.out = out
        self.backend = backend
        self.interval = float(os.environ.get("OUT_SYNC_INTERVAL", 600) if interval is None else interval)
        self.last_push = 0.0
        self.n_pushes = 0
        self._thread = None
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return self.backend is not None

    # manifest of what the remote already has: rel -> [size, mtime_ns]
    def _manifest_path(self):
        return os.path.join(self.out, MANIFEST)

    def _load_manifest(self):
        try:
            with open(self._manifest_path()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, m):
        os.makedirs(self.out, exist_ok=True)
        tmp = self._manifest_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f)
        os.replace(tmp, self._manifest_path())

    def _stat(self, rel):
        st = os.stat(os.path.join(self.out, rel))
        return [st.st_size, st.st_mtime_ns]

    def changed_files(self):
        m = self._load_manifest()
        return [r for r in selected_files(self.out) if m.get(r) != self._stat(r)]

    def _push(self, squash=False):
        with self._lock:
            rels = self.changed_files()
            if not rels:
                return True
            stats = {r: self._stat(r) for r in rels}
            mb = sum(s[0] for s in stats.values()) / 2 ** 20
            t0 = time.time()
            try:
                self.backend.upload(self.out, rels)
            except Exception as e:  # never kill training because of storage
                _log(f"push to {self.backend.name} FAILED ({type(e).__name__}: {e}); will retry later")
                return False
            m = self._load_manifest()
            m.update(stats)
            self._save_manifest(m)
            self.n_pushes += 1
            _log(f"pushed {len(rels)} files ({mb:.1f} MB) to {self.backend.name} in {time.time() - t0:.0f}s")
            if squash or self.n_pushes % 6 == 0:
                try:
                    self.backend.squash()
                except Exception as e:
                    _log(f"history squash skipped ({type(e).__name__}: {e})")
            return True

    def maybe_push(self):
        """Start a background push if the throttle interval has passed and no push is running."""
        if not self.enabled or time.time() - self.last_push < self.interval:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self.last_push = time.time()
        self._thread = threading.Thread(target=self._push, daemon=True)
        self._thread.start()

    def push_now(self, squash=False):
        """Blocking push (end of a run). Returns True on success or when syncing is disabled."""
        if not self.enabled:
            return True
        if self._thread is not None:
            self._thread.join()
        self.last_push = time.time()
        ok = self._push(squash=squash)
        if not ok:
            print("\033[91m[sync] results are only on the local disk; re-run the push cell before the "
                  "session ends\033[0m", flush=True)
        return ok

    def ensure(self, rel):
        """Make sure OUT/rel exists locally, fetching it from the backend if needed. Returns True if present."""
        rel = rel.replace(os.sep, "/")
        if os.path.exists(os.path.join(self.out, rel)):
            return True
        if not self.enabled:
            return False
        try:
            if not self.backend.exists(rel):
                return False
            _log(f"fetching {rel} from {self.backend.name}")
            self.backend.download([rel], self.out)
        except Exception as e:
            _log(f"could not fetch {rel} ({type(e).__name__}: {e})")
            return False
        m = self._load_manifest()
        m[rel] = self._stat(rel)
        self._save_manifest(m)
        return True

    def pull(self, include_lazy=False):
        """Copy remote files that are missing locally. Never overwrites a local file."""
        if not self.enabled:
            _log("no persistent storage configured (OUT_SYNC unset); nothing to pull")
            return 0
        remote = self.backend.remote_files()
        missing = [r for r in remote if not os.path.exists(os.path.join(self.out, r))
                   and (include_lazy or not _match(r, LAZY))]
        if missing:
            _log(f"pulling {len(missing)} files from {self.backend.name}")
            self.backend.download(missing, self.out)
        m = self._load_manifest()
        for r in missing:
            m[r] = self._stat(r)
        self._save_manifest(m)
        lazy = sum(1 for r in remote if _match(r, LAZY))
        _log(f"pull done: {len(missing)} new files; {len(remote)} files in storage "
             f"({lazy} checkpoints/corruption arrays fetched on demand)")
        return len(missing)


def rel_to_out(out, path):
    return os.path.relpath(path, out).replace(os.sep, "/")


_SYNCERS = {}


def get_syncer(out):
    key = os.path.abspath(out)
    if key not in _SYNCERS:
        try:
            backend = backend_from_env()
        except Exception as e:
            _log(f"storage backend unavailable ({type(e).__name__}: {e}); results stay on the local disk")
            backend = None
        _SYNCERS[key] = Syncer(out, backend)
    return _SYNCERS[key]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pull", "push", "status"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="pull: also fetch checkpoints and corruption arrays")
    args = ap.parse_args(argv)
    s = get_syncer(args.out)
    if args.action == "pull":
        s.pull(include_lazy=args.all)
    elif args.action == "push":
        ok = s.push_now()
        raise SystemExit(0 if ok else 1)
    else:
        name = s.backend.name if s.enabled else "none"
        ch = s.changed_files()
        print(f"backend: {name}\nlocal files tracked: {len(selected_files(args.out))}\nnot yet pushed: {len(ch)}")
        for r in ch[:20]:
            print("  ", r)


if __name__ == "__main__":
    main()
