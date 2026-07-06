from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("times", nargs="+", type=float)
    parser.add_argument("--tile-width", type=int, default=480)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{args.video}")
    frames: list[np.ndarray] = []
    for seconds in args.times:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        scale = args.tile_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.rectangle(frame, (0, 0), (165, 34), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{seconds:.1f}s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("没有读到视频帧")
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    tile_height, tile_width = frames[0].shape[:2]
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = frame
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"无法写入：{args.output}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
