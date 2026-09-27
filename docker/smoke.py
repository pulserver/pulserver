"""Drive the composed console as a scanner console does: an exam, then one scan through Gadgetron.

Exits 1 unless the localizer holds three planes and the scan returns at least
one DICOM image with pixels.
"""

import base64
import io
import json
import sys
import time

import pydicom
from websockets.sync.client import connect

ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
PROTOCOL_BEGIN, PROTOCOL_END = "[NimPulseqGUI Protocol]", "[NimPulseqGUI Protocol End]"
PRESCRIPTION = [f"fov_offset_{a}: 0.0" for a in "xyz"] + [
    f"fov_rotation_{i}{j}: {1.0 if i == j else 0.0}"
    for i in (1, 2, 3)
    for j in (1, 2, 3)
]


def main() -> int:
    deadline = time.monotonic() + 600.0
    while True:
        try:
            socket = connect(ADDRESS, max_size=None, open_timeout=10).__enter__()
            break
        except OSError:
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

    exam, _ = call("exam", subject="vials")
    print(f"localizer: {len(exam['localizer'])} planes")
    block = (
        "\n".join([PROTOCOL_BEGIN, "nx: 32", "ny: 32", *PRESCRIPTION, PROTOCOL_END])
        + "\n"
    )
    generated, _ = call("generate", plugin="gre2d", block=block)
    print(generated["reply"].strip())
    done, messages = call(
        "scan",
        design=generated["design"],
        rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        centre_mm=[0.0, 0.0, 0.0],
    )
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
    return 0 if ok and done["done"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
