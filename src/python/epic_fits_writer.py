"""
Writes the imager's long accumulations (FITS, quick look, daily digest) in a
process of its own.

epic++ embeds Python, and every call into it holds the one interpreter lock.
Written from inside the imager, a 201 MB accumulation (192 channels of
256 x 256) held that lock for seconds: the pixel extractor, which calls into
Python for every image, stopped, the images backed up and the correlator ran
out of output buffers. The disk saver therefore starts this module once
(src/ex/fits_writer_proc.hpp) and hands it each accumulation over its
standard input; the writing happens under this process's own lock.

Protocol, one job at a time:
  epic++ -> writer   one line of JSON {"grid_size", "nchan", "out_dir",
                     "nbytes", "meta"}, then nbytes of float32 data in the
                     imager's order [chan][a][b][4]
  writer -> epic++   one line of JSON {"filename": path or ""}

The writer exits at the end of its input, so it ends with the imager. Its
standard output carries only the replies: anything printed while writing goes
to standard error.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def _read_exact(stream, n: int) -> bytearray | None:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = stream.readinto(view[got:])
        if not k:
            return None
        got += k
    return buf


def serve(inp, reply) -> None:
    """Answer jobs from the binary stream inp on the text stream reply until inp ends"""
    import epic_utils  # after the redirection of stdout, it prints while importing

    while True:
        line = inp.readline()
        if not line:
            return
        try:
            job = json.loads(line)
            data = _read_exact(inp, int(job["nbytes"]))
        except (ValueError, KeyError) as e:
            print(f"epic_fits_writer: bad job header: {e!r}", file=sys.stderr, flush=True)
            return
        if data is None:
            return
        try:
            arr = np.frombuffer(data, dtype=np.float32)
            name = epic_utils.save_output(arr, int(job["grid_size"]), int(job["nchan"]),
                                          job["out_dir"], job["meta"])
        except Exception as e:  # noqa: BLE001 - one bad job must not end the writer
            print(f"epic_fits_writer: accumulation not written: {e!r}", file=sys.stderr, flush=True)
            name = ""
        reply.write(json.dumps({"filename": name or ""}) + "\n")
        reply.flush()


def main() -> None:
    # the replies get their own copy of stdout; everything else printed goes to stderr
    reply = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    serve(sys.stdin.buffer, reply)


if __name__ == "__main__":
    main()
