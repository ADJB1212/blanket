from __future__ import annotations

import os
import subprocess
import sys
from textwrap import dedent


def test_deterministic_execution_across_processes() -> None:
    code = dedent("""
        from hashlib import sha256
        from io import BytesIO
        from random import Random, getstate, seed
        from threading import current_thread

        from blanket import Image, ImageBatch, ImagePalette
        from blanket.Compressor import LosslessImageCompressor

        size = (769, 513)
        colors = [bytes((i, i * 37 % 256, i * 71 % 256)) for i in range(32)]
        pixels = b"".join(colors[i % 32] for i in Random(7).randbytes(size[0] * size[1]))
        image = Image.frombytes("RGB", size, pixels)

        spread = image.effect_spread(3).tobytes()
        assert spread != pixels
        assert image.effect_spread(3).tobytes() == spread
        seed(97)
        state = getstate()
        palette = bytes(ImagePalette.random("RGBA").palette)
        assert bytes(ImagePalette.random("RGBA").palette) == palette
        assert getstate() == state

        events = []
        caller = current_thread()

        def sources():
            for value in range(3):
                events.append(("source", value))
                yield Image.new("L", (1, 1), value)

        def step(frame):
            assert current_thread() is caller
            events.append(("step", frame.getpixel((0, 0))))
            return frame

        results = ImageBatch.Pipeline([step]).map(sources(), max_workers=4)
        assert [frame.getpixel((0, 0)) for frame in results] == [0, 1, 2]
        assert events == [(event, i) for i in range(3) for event in ("source", "step")]

        compressed = image.encode("PNG", compressor=LosslessImageCompressor(effort=7))
        tiff = image.encode("TIFF", compression="tiff_lzw")
        for encoded in (compressed, tiff):
            with Image.open(BytesIO(encoded)) as reopened:
                assert reopened.convert("RGB").tobytes() == pixels
        for data in (spread, palette, compressed, tiff):
            print(sha256(data).hexdigest())
        """)
    outputs = []
    for workers in (1, 4):
        env = {**os.environ, "BLANKET_DETERMINISTIC": "1", "RAYON_NUM_THREADS": str(workers), "PYTHONHASHSEED": str(workers)}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
