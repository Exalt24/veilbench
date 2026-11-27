"""Record `veilbench demo` into docs/demo.gif.

The command is really run and its output is read line by line with the time each line arrived. The lines are then drawn into
terminal looking frames with a monospaced font, in the order and at the pace (long waits shortened) they arrived. Nothing in the
picture is typed by hand: change what `veilbench demo` prints and the GIF changes with it.

    python scripts/make_demo_gif.py            # writes docs/demo.gif and .work/demo.txt (the transcript; it is not committed because it holds a key shaped string)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
W, H, PAD = 1040, 700, 16
COLS, LINE_H, FONT_PX = 100, 20, 16
BG, FG, DIM, BOLD, GOOD = (22, 24, 29), (200, 205, 214), (120, 128, 140), (255, 255, 255), (130, 200, 150)


def capture() -> list[tuple[float, str]]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    p = subprocess.Popen([sys.executable, "-m", "veilbench.cli", "demo", "--pause", "0.9"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
    t0, out = time.perf_counter(), []
    for line in p.stdout:
        out.append((time.perf_counter() - t0, line.rstrip("\n")))
    if p.wait() != 0:
        raise SystemExit("the demo command failed")
    return out


def wrap(lines: list[tuple[float, str]]) -> list[tuple[float, str, bool]]:
    """(arrival time, text, is a heading) for every drawn row."""
    rows = []
    for t, line in lines:
        heading = line.startswith("\x1b[1m")
        text = re.sub(r"\x1b\[[0-9;]*m", "", line)
        for part in (textwrap.wrap(text, COLS) or [""]):
            rows.append((t, part, heading))
    return rows


def draw(rows: list[tuple[float, str, bool]], upto: int, font, bold, prompt: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill=(40, 43, 51))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 22, 9, 26 + i * 22, 21], fill=c)
    d.text((W // 2 - 90, 7), "veilbench demo", font=font, fill=DIM)
    visible = (H - 30 - 2 * PAD) // LINE_H
    shown = ([("$ veilbench demo", True)] if prompt else []) + [(r[1], r[2]) for r in rows[:upto]]
    for i, (text, heading) in enumerate(shown[-visible:]):
        y = 30 + PAD + i * LINE_H
        if text.startswith("$ "):
            d.text((PAD, y), "$", font=bold, fill=GOOD)
            d.text((PAD + 2 * 9, y), text[2:], font=bold, fill=BOLD)
        elif heading:
            d.text((PAD, y), text, font=bold, fill=BOLD)
        else:
            d.text((PAD, y), text, font=font, fill=FG)
    return img


def main() -> int:
    lines = capture()
    rows = wrap(lines)
    font = ImageFont.truetype(str(FONT / "consola.ttf"), FONT_PX)
    bold = ImageFont.truetype(str(FONT / "consolab.ttf"), FONT_PX)
    # real arrival times, with the model training wait and every long gap cut to 1.1 seconds
    frames, durations = [], []
    times = sorted({t for t, _, _ in rows})
    gaps = {t: min(max(b - a, 0.25), 1.1) for a, b, t in zip([0.0] + times[:-1], times, times)}
    # frame 0 already shows the raw record and what was detected: the first four lines the command printed
    cut = times[min(3, len(times) - 1)]
    i = sum(1 for r in rows if r[0] <= cut)
    frames.append(draw(rows, i, font, bold, True))
    durations.append(1400)
    while i < len(rows):
        t = rows[i][0]
        j = i
        while j < len(rows) and rows[j][0] == t:
            j += 1
        frames.append(draw(rows, j, font, bold, True))
        durations.append(int(gaps.get(t, 0.4) * 1000) + 500)
        i = j
    durations[-1] = 4500
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=48) for f in frames]
    out = ROOT / "docs" / "demo.gif"
    out.parent.mkdir(exist_ok=True)
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    (ROOT / ".work").mkdir(exist_ok=True)
    (ROOT / ".work" / "demo.txt").write_text("$ veilbench demo\n" + "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", ln) for _, ln in lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
