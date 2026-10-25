"""Backups of run directories (runs/ is gitignored, so paid data has no second copy otherwise).

python -m agent.common.backup runs/<run> [--note "..."]      -> zip + SHA-256, logged in two manifests
python -m agent.common.backup --verify                       -> re-hash every logged backup and compare

Each backup is a zip of the whole run directory in DEST (outside the repository); its SHA-256 is appended to DEST/
MANIFEST.tsv and to freeze/backups.tsv, which is committed, so the integrity of every copy can be checked later from the
repository alone. Backups are never overwritten (the file name carries the time). DEST is on the same machine: copy it
to another device as well.
"""
import argparse
import hashlib
import os
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(os.path.dirname(ROOT), "thesis-agent-control_archive", "backups")
LOG = os.path.join(ROOT, "freeze", "backups.tsv")
HEADER = "time\trun\tzip\tfiles\tbytes\tsha256\tnote\n"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _append(path, line):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(HEADER)
        f.write(line)


def backup_run(run_dir, dest=DEST, note="", log=LOG):
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    name = os.path.basename(os.path.normpath(run_dir))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(dest, exist_ok=True)
    zpath = os.path.join(dest, f"{name}_{stamp}.zip")
    if os.path.exists(zpath):
        raise FileExistsError(zpath)
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(run_dir):
            for fn in sorted(files):
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, os.path.dirname(run_dir)))
                n += 1
    with zipfile.ZipFile(zpath) as z:                         # the archive must read back completely
        bad = z.testzip()
        if bad is not None:
            raise IOError(f"corrupt member {bad} in {zpath}")
    digest, size = sha256(zpath), os.path.getsize(zpath)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{os.path.basename(zpath)}\t{n}\t{size}\t{digest}\t{note}\n"
    _append(os.path.join(dest, "MANIFEST.tsv"), line)
    if log:
        _append(log, line)
    return {"zip": zpath, "files": n, "bytes": size, "sha256": digest}


def verify(dest=DEST, log=LOG):
    """Re-hash every backup listed in the committed log; returns [(zip, ok or reason)]."""
    out = []
    if not os.path.exists(log):
        return out
    with open(log, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            zpath = os.path.join(dest, parts[2])
            if not os.path.exists(zpath):
                out.append((parts[2], "missing"))
            else:
                out.append((parts[2], "ok" if sha256(zpath) == parts[5] else "hash differs"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--note", default="")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        res = verify()
        for z, status in res:
            print(f"{status}\t{z}")
        return 0 if all(s == "ok" for _, s in res) else 1
    for r in a.runs:
        info = backup_run(r, note=a.note)
        print(f"{info['zip']}  files={info['files']}  bytes={info['bytes']}  sha256={info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
