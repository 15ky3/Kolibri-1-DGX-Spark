#!/usr/bin/env python3
"""Evict the checkpoint's pages from the OS page cache (no root needed).

posix_fadvise(DONTNEED) on every snapshot file. Used to measure cold-start
load times honestly; nothing else needs it.
Usage: scripts/drop-model-cache.py <snapshot dir>
"""
import os
import sys

snap = sys.argv[1]
for name in sorted(os.listdir(snap)):
    path = os.path.realpath(os.path.join(snap, name))
    if not os.path.isfile(path):
        continue
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
print("evicted", snap)
