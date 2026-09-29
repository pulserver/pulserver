"""Drive a console as a scanner console does: an exam, then one scan.

Usage: ``python docker/smoke.py [ADDRESS [COIL]] [--subject SUBJECT]``, the
exam started on the subject named, the vials by default, which a console with
field maps examines on BrainWeb, in the coil named, or in the console's own.
Exits 1 unless the localizer holds three planes and the scan returns at least
one DICOM image with pixels.
"""

import argparse
import base64
import io
import json
import sys
import time

import pydicom
from websockets.exceptions import InvalidHandshake
from websockets.sync.client import connect

PROTOCOL_BEGIN, PROTOCOL_END = "[NimPulseqGUI Protocol]", "[NimPulseqGUI Protocol End]"
PRESCRIPTION = [f"fov_offset_{a}: 0.0" for a in "xyz"] + [
    f"fov_rotation_{i}{j}: {1.0 if i == j else 0.0}"
    for i in (1, 2, 3)
    for j in (1, 2, 3)
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("address", nargs="?", default="ws://127.0.0.1:8765")
    parser.add_argument("coil", nargs="?")
    parser.add_argument("--subject", default="vials")
    args = parser.parse_args()
    deadline = time.monotonic() + 600.0
    while True:
        try:
            socket = connect(args.address, max_size=None, open_timeout=10).__enter__()
            break
        # A published port may accept a connection before the console listens,
        # and then close it unanswered.
        except (OSError, InvalidHandshake):
            if time.monotonic() > deadline:
                raise
            time.sleep(5.0)
    ident = 0

    def call(name, **fields):
        nonlocal ident
        ident += 1
        socket.send(json.dumps({"id": ident, "call": name, **fields}))
        messages = []
        while True:
            reply = json.loads(socket.recv(timeout=900))
            if "error" in reply:
                raise RuntimeError(reply["error"])
            if name != "scan" or "done" in reply:
                return reply, messages
            messages.append(reply)

    coil = {} if args.coil is None else {"coil": args.coil}
    exam, _ = call("exam", subject=args.subject, **coil)
    print(f"localizer: {len(exam['localizer'])} planes")
    block = (
        "\n".join([PROTOCOL_BEGIN, "nx: 32", "ny: 32", *PRESCRIPTION, PROTOCOL_END])
        + "\n"
    )
    generated, _ = call("generate", plugin="gre2d", block=block)
    print(generated["reply"].strip())
    started = time.monotonic()
    done, messages = call(
        "scan",
        design=generated["design"],
        rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        centre_mm=[0.0, 0.0, 0.0],
    )
    took = time.monotonic() - started
    duration = max((m["duration"] for m in messages if "clock" in m), default=0.0)
    preparing = sum("preparing" in m for m in messages)
    print(f"scan of {duration:.1f} s took {took:.1f} s, {preparing} preparing messages")
    for message in messages:
        if "text" in message:
            print("text:", message["text"].strip())
    images = [
        pydicom.dcmread(io.BytesIO(base64.b64decode(m["dicom"])))
        for m in messages
        if "dicom" in m
    ]
    print(f"scan status {done['done']}, {len(images)} DICOM images")
    socket.close()
    ok = len(exam["localizer"]) == 3 and images and images[0].pixel_array.max() > 0
    # The image's console plays a scan on the scanner's clock.
    ok = ok and duration > 0.0 and took >= duration and preparing > 0
    return 0 if ok and done["done"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
