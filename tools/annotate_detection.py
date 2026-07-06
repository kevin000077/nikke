from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from marble_aim.config import VisionConfig
from marble_aim.vision import BoardDetector


def annotate(input_path: Path, output_path: Path) -> tuple[int, bool]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"无法读取图片：{input_path}")
    result = BoardDetector(VisionConfig()).detect(image)
    if result.board is None:
        raise RuntimeError("没有识别到白色碰撞范围")

    output = image.copy()
    tint = image.copy()
    board = result.board
    cv2.rectangle(
        tint,
        (round(board.left), round(board.top)),
        (round(board.right), round(board.bottom)),
        (255, 220, 0),
        -1,
    )
    output = cv2.addWeighted(tint, 0.10, output, 0.90, 0)
    cv2.rectangle(
        output,
        (round(board.left), round(board.top)),
        (round(board.right), round(board.bottom)),
        (255, 230, 0),
        5,
    )
    cv2.putText(
        output,
        "COLLISION AREA",
        (round(board.left) + 12, round(board.top) + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 230, 0),
        3,
        cv2.LINE_AA,
    )

    for index, obstacle in enumerate(result.obstacles, start=1):
        rect = obstacle.rect
        left, top = round(rect.left), round(rect.top)
        right, bottom = round(rect.right), round(rect.bottom)
        cv2.rectangle(output, (left, top), (right, bottom), (30, 255, 80), 3)
        cv2.circle(output, (left + 13, top + 13), 12, (20, 80, 20), -1)
        cv2.putText(
            output,
            str(index),
            (left + 5, top + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if result.aim_line:
        lower, upper = result.aim_line
        cv2.line(
            output,
            (round(lower[0]), round(lower[1])),
            (round(upper[0]), round(upper[1])),
            (255, 30, 255),
            5,
            cv2.LINE_AA,
        )
        for point in (lower, upper):
            cv2.circle(
                output,
                (round(point[0]), round(point[1])),
                10,
                (255, 30, 255),
                3,
                cv2.LINE_AA,
            )
        midpoint = (
            round((lower[0] + upper[0]) / 2),
            round((lower[1] + upper[1]) / 2),
        )
        cv2.putText(
            output,
            "AIM LINE",
            (midpoint[0] + 14, midpoint[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 30, 255),
            3,
            cv2.LINE_AA,
        )

    legend_x, legend_y = 60, 70
    cv2.rectangle(output, (legend_x - 20, legend_y - 40), (430, legend_y + 85), (20, 20, 20), -1)
    legend = [
        ((30, 255, 80), f"BLOCKS: {len(result.obstacles)}"),
        ((255, 230, 0), "WHITE COLLISION FRAME"),
        ((255, 30, 255), f"AIM LINE: {'FOUND' if result.aim_line else 'NOT FOUND'}"),
    ]
    for row, (color, label) in enumerate(legend):
        y = legend_y + row * 36
        cv2.line(output, (legend_x, y), (legend_x + 38, y), color, 6)
        cv2.putText(
            output,
            label,
            (legend_x + 52, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"无法写入结果图片：{output_path}")
    return len(result.obstacles), result.aim_line is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="标注方块、碰撞框和发射线")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    blocks, aim_found = annotate(args.input, args.output)
    print(f"blocks={blocks} aim_line={aim_found} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
