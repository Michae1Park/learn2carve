#!/usr/bin/env python3
"""Back up the saved runs (checkpoints + configs) to a Hugging Face model repo.

The repo keeps only the final G1 policy's weights; every stage's weights live on the Hub, at the same paths as here
(`saved/g1/chain/NN_<stage>/checkpoints/best_agent.pt`, ...), with `scripts/hf_model_card.md` as the Hub README.

Credentials: none are passed to or stored by this script. Log in once with `hf auth login`, which keeps the token in
`~/.cache/huggingface/token` (outside the repo); huggingface_hub reads it from there (or from $HF_TOKEN).

    uv run --no-project --with huggingface_hub python scripts/upload_hf.py <user>/learn2carve            # dry run
    uv run --no-project --with huggingface_hub python scripts/upload_hf.py <user>/learn2carve --upload   # private
    ... --upload --public                                                                                 # public
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "saved"
CARD = ROOT / "scripts" / "hf_model_card.md"
ALLOWED = {".pt", ".yaml", ".md", ".csv", ".png"}
ALLOWED_NAMES = {"SHA256SUMS"}
TEXT = {".yaml", ".md", ".csv"}
# Hub/GitHub/AWS tokens, private keys, absolute home paths
LEAK = re.compile(r"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|/home/\w|/Users/\w")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("repo_id", help="<user or org>/<name> on the Hub")
p.add_argument("--upload", action="store_true", help="actually upload (default: dry run, lists the files)")
p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
args = p.parse_args()

def git(*cmd):
    return subprocess.run(["git", "-C", str(ROOT), *cmd], check=True, capture_output=True, text=True).stdout.strip()


# Tie the upload to a git commit: the Hub commit message names it, so saved/ must match that commit exactly
# (ignored files, i.e. the weights kept only on the Hub, are not counted).
head = git("rev-parse", "HEAD")
dirty = git("status", "--porcelain", "--", "saved", str(CARD.relative_to(ROOT)))
if dirty:
    sys.exit("saved/ or the model card has uncommitted changes; commit them first so the upload matches a commit:\n"
             + dirty)


def allowed(f):
    return f.suffix in ALLOWED or f.name in ALLOWED_NAMES


files = sorted(f for f in SRC.rglob("*") if f.is_file())
skipped = [f for f in files if not allowed(f)]
files = [f for f in files if allowed(f)]

leaks = []
for f in files + [CARD]:
    if f.suffix in TEXT or f.name in ALLOWED_NAMES:
        for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if LEAK.search(line):
                leaks.append(f"{f.relative_to(ROOT)}:{n}")
if leaks:
    sys.exit("Refusing to upload, possible credential or local path in:\n  " + "\n  ".join(leaks))

total = sum(f.stat().st_size for f in files)
for f in files:
    print(f"{f.stat().st_size / 1e6:7.2f} MB  {f.relative_to(ROOT)}")
print(f"{total / 1e6:7.2f} MB  total, {len(files)} files, + README.md from {CARD.relative_to(ROOT)}")
print(f"git commit: {head}")
for f in skipped:
    print(f"  skipped (extension not allowed): {f.relative_to(ROOT)}")
if not args.upload:
    print(f"\nDry run. Add --upload to push to {args.repo_id} (if it does not exist yet, it is created "
          f"{'public' if args.public else 'private'}; an existing repo keeps its visibility).")
    sys.exit(0)

from huggingface_hub import HfApi  # noqa: E402  (only needed for a real upload)

api = HfApi()  # token from `hf auth login` / $HF_TOKEN, never from this script
try:
    user = api.whoami()["name"]
except Exception:
    sys.exit("Not logged in to Hugging Face: run `hf auth login` first.")
print(f"\nLogged in as {user}")

url = api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
link = f"from github.com/Michae1Park/learn2carve@{head}"
api.upload_folder(repo_id=args.repo_id, folder_path=str(SRC), path_in_repo="saved",
                  allow_patterns=[f"**/*{e}" for e in sorted(ALLOWED)] + [f"**/{n}" for n in sorted(ALLOWED_NAMES)],
                  commit_message=f"Saved runs: checkpoints and configs, {link}")
card = CARD.read_text().replace("<this repo>", args.repo_id).encode()
info = api.upload_file(repo_id=args.repo_id, path_or_fileobj=card, path_in_repo="README.md",
                       commit_message=f"Model card, {link}")
print(f"Uploaded to {url} at revision {info.oid}  (git {head})")
