"""Persistent-storage sync (src/storage.py): folder backend end to end, Hugging Face backend against a fake Hub."""
import os
import shutil

import pytest

from src import storage


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _populate(out):
    _write(os.path.join(out, "split", "train.txt"), "a 1\n")
    _write(os.path.join(out, "runs", "E0_standard", "seed0", "train_log.csv"), "epoch\n1\n")
    _write(os.path.join(out, "runs", "E0_standard", "seed0", "checkpoints", "last.pt"), "ckpt")
    _write(os.path.join(out, "runs", "E0_standard", "seed0", "checkpoints", "last.pt.tmp"), "partial")
    _write(os.path.join(out, "data_cache", "train32.pt"), "cache")
    _write(os.path.join(out, "data_cache", "gtsrb_c", "fog_s1.npy"), "arr")
    _write(os.path.join(out, ".smoke", "data_cache", "gtsrb_c", "fog_s1.npy"), "smoke")
    _write(os.path.join(out, "scratch", "junk.bin"), "junk")


def test_selected_files_patterns(tmp_path):
    out = str(tmp_path / "out")
    _populate(out)
    sel = storage.selected_files(out)
    assert "split/train.txt" in sel
    assert "runs/E0_standard/seed0/train_log.csv" in sel
    assert "runs/E0_standard/seed0/checkpoints/last.pt" in sel
    assert "data_cache/train32.pt" in sel and "data_cache/gtsrb_c/fog_s1.npy" in sel
    assert not any(s.endswith(".tmp") for s in sel)
    assert not any(s.startswith(".smoke") or s.startswith("scratch") for s in sel)


def test_dir_backend_push_pull_ensure(tmp_path, monkeypatch):
    out, remote = str(tmp_path / "out"), str(tmp_path / "drive")
    monkeypatch.setenv("OUT_SYNC", f"dir:{remote}")
    _populate(out)
    s = storage.Syncer(out, storage.backend_from_env(), interval=0)
    assert s.push_now()
    assert os.path.exists(os.path.join(remote, "runs", "E0_standard", "seed0", "checkpoints", "last.pt"))
    assert s.changed_files() == []                      # manifest records what was pushed
    # change one file -> only that one is pending
    _write(os.path.join(out, "runs", "E0_standard", "seed0", "train_log.csv"), "epoch\n1\n2\n")
    assert s.changed_files() == ["runs/E0_standard/seed0/train_log.csv"]
    assert s.push_now()

    # new session: empty local folder; pull fetches eager files, not checkpoints / corruption arrays
    out2 = str(tmp_path / "out2")
    s2 = storage.Syncer(out2, storage.backend_from_env(), interval=0)
    n = s2.pull()
    assert n > 0
    assert open(os.path.join(out2, "runs", "E0_standard", "seed0", "train_log.csv")).read() == "epoch\n1\n2\n"
    assert os.path.exists(os.path.join(out2, "data_cache", "train32.pt"))
    assert not os.path.exists(os.path.join(out2, "runs", "E0_standard", "seed0", "checkpoints", "last.pt"))
    assert not os.path.exists(os.path.join(out2, "data_cache", "gtsrb_c", "fog_s1.npy"))
    # lazy fetch on demand
    assert s2.ensure("runs/E0_standard/seed0/checkpoints/last.pt")
    assert open(os.path.join(out2, "runs", "E0_standard", "seed0", "checkpoints", "last.pt")).read() == "ckpt"
    assert not s2.ensure("runs/E9/seed0/checkpoints/best.pt")
    # pull never overwrites a newer local file
    _write(os.path.join(out2, "split", "train.txt"), "local edit")
    s2.pull()
    assert open(os.path.join(out2, "split", "train.txt")).read() == "local edit"
    assert s2.changed_files() == ["split/train.txt"]


def test_disabled_syncer_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("OUT_SYNC", raising=False)
    out = str(tmp_path / "out")
    _populate(out)
    s = storage.Syncer(out, storage.backend_from_env())
    assert not s.enabled
    assert s.push_now() is True
    assert s.pull() == 0
    s.maybe_push()
    assert s.ensure("split/train.txt") and not s.ensure("runs/x/seed0/checkpoints/best.pt")


def test_failed_push_does_not_raise(tmp_path):
    class Broken:
        name = "broken"

        def upload(self, out, rels):
            raise ConnectionError("network down")

        def squash(self):
            pass

    out = str(tmp_path / "out")
    _populate(out)
    s = storage.Syncer(out, Broken(), interval=0)
    assert s.push_now() is False
    assert len(s.changed_files()) > 0                   # still pending, retried next time


def test_throttle(tmp_path, monkeypatch):
    out, remote = str(tmp_path / "out"), str(tmp_path / "drive")
    _populate(out)
    s = storage.Syncer(out, storage.DirBackend(remote), interval=3600)
    s.last_push = 0
    s.maybe_push()                                      # first push starts
    s._thread.join()
    _write(os.path.join(out, "split", "val.txt"), "b 2\n")
    s.maybe_push()                                      # within the interval: nothing
    assert "split/val.txt" in s.changed_files()
    assert s.push_now() and s.changed_files() == []


class FakeHfApi:
    """Stands in for huggingface_hub.HfApi; stores files in a folder. Keyword names mirror the real API."""
    store = None
    created = []
    squashed = 0

    def __init__(self, *a, **k):
        pass

    def create_repo(self, repo_id, *, repo_type=None, private=None, exist_ok=False, **k):
        assert repo_type == "dataset" and private is True and exist_ok is True
        FakeHfApi.created.append(repo_id)

    def upload_folder(self, *, repo_id, folder_path, path_in_repo=None, commit_message=None, repo_type=None, **k):
        assert repo_type == "dataset" and path_in_repo == ""
        for root, _, files in os.walk(folder_path):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), folder_path)
                dst = os.path.join(self.store, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(root, fn), dst)

    def list_repo_files(self, repo_id, *, repo_type=None, **k):
        return storage.selected_files(self.store) + [".gitattributes"]

    def file_exists(self, repo_id, filename, *, repo_type=None, **k):
        return os.path.isfile(os.path.join(self.store, filename))

    def super_squash_history(self, repo_id, *, repo_type=None, commit_message=None, **k):
        FakeHfApi.squashed += 1


def fake_hf_hub_download(repo_id, filename, *, repo_type=None, local_dir=None, **k):
    dst = os.path.join(local_dir, filename)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(FakeHfApi.store, filename), dst)
    return dst


def test_hf_backend_with_fake_hub(tmp_path, monkeypatch):
    import huggingface_hub
    FakeHfApi.store = str(tmp_path / "hub")
    os.makedirs(FakeHfApi.store)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setenv("OUT_SYNC", "hf:student/gtsrb-runs")
    out = str(tmp_path / "out")
    _populate(out)
    s = storage.Syncer(out, storage.backend_from_env(), interval=0)
    assert FakeHfApi.created == ["student/gtsrb-runs"]
    assert s.push_now(squash=True) and FakeHfApi.squashed == 1
    assert not any(d.startswith(".sync_stage") for d in os.listdir(out))   # staging cleaned up
    out2 = str(tmp_path / "out2")
    s2 = storage.Syncer(out2, storage.backend_from_env(), interval=0)
    s2.pull()
    assert os.path.exists(os.path.join(out2, "split", "train.txt"))
    assert not os.path.exists(os.path.join(out2, "runs", "E0_standard", "seed0", "checkpoints", "last.pt"))
    assert s2.ensure("runs/E0_standard/seed0/checkpoints/last.pt")
    assert not any(d.startswith(".sync_pull") for d in os.listdir(out2))


def test_bad_spec_rejected(monkeypatch):
    monkeypatch.setenv("OUT_SYNC", "s3:bucket")
    with pytest.raises(ValueError):
        storage.backend_from_env()
