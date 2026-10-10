"""Reject oversized Git blobs before publishing (also checks full tree in CI)."""
import argparse
import subprocess

LIMIT = 50 * 1024 * 1024


def check(all_files=False):
    command = (["git", "ls-files", "-z"] if all_files else
               ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])
    names = subprocess.check_output(command).decode().split("\0")
    oversized = []
    for name in filter(None, names):
        size = int(subprocess.check_output(["git", "cat-file", "-s", f":{name}"]))
        if size >= LIMIT:
            oversized.append(f"{name}: {size} bytes")
    if oversized:
        raise SystemExit("Files exceed the 50 MiB budget:\n" + "\n".join(oversized))
    print("Git blob size check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    check(parser.parse_args().all)
