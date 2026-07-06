from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from marble_aim.config import VisionConfig
from marble_aim.geometry import direction_from_angle, simulate_trajectory, vec
from marble_aim.vision import BoardDetector, DetectionResult


def _angle_from_line(
    line: tuple[tuple[float, float], tuple[float, float]]
) -> float:
    lower, upper = line
    return math.degrees(
        math.atan2(upper[0] - lower[0], -(upper[1] - lower[1]))
    )


def _read_detection(
    capture: cv2.VideoCapture,
    detector: BoardDetector,
    frame_index: int,
) -> tuple[np.ndarray, DetectionResult]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"无法读取第 {frame_index} 帧")
    return frame, detector.detect(frame)


def stabilise_final_aim(
    capture: cv2.VideoCapture,
    detector: BoardDetector,
    aim_frame: int,
    window: int,
) -> tuple[DetectionResult, float, float, int]:
    observations: list[tuple[int, float, float, DetectionResult]] = []
    for frame_index in range(max(0, aim_frame - window + 1), aim_frame + 1):
        _, detection = _read_detection(capture, detector, frame_index)
        if detection.board is None or detection.aim_line is None:
            continue
        lower, _ = detection.aim_line
        observations.append(
            (
                frame_index,
                _angle_from_line(detection.aim_line),
                lower[0],
                detection,
            )
        )
    if len(observations) < max(3, window // 3):
        raise RuntimeError("最终瞄准窗口中的有效发射线样本不足")
    median_angle = float(np.median([item[1] for item in observations]))
    filtered = [
        item for item in observations if abs(item[1] - median_angle) <= 3.0
    ]
    if len(filtered) < 3:
        filtered = observations
    angle = float(np.median([item[1] for item in filtered]))
    lower_x = float(np.median([item[2] for item in filtered]))
    selected = max(filtered, key=lambda item: item[0])[3]
    return selected, angle, lower_x, len(filtered)


def dashed_polyline(
    image: np.ndarray,
    points: list[np.ndarray],
    color: tuple[int, int, int],
    thickness: int = 3,
    dash: float = 12.0,
    gap: float = 8.0,
) -> None:
    phase = 0.0
    period = dash + gap
    for first, second in zip(points, points[1:]):
        delta = second - first
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            continue
        direction = delta / length
        position = 0.0
        while position < length:
            cycle_position = phase % period
            if cycle_position < dash:
                draw_length = min(dash - cycle_position, length - position)
                start = first + direction * position
                end = first + direction * (position + draw_length)
                cv2.line(
                    image,
                    tuple(np.round(start).astype(int)),
                    tuple(np.round(end).astype(int)),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                position += draw_length
                phase += draw_length
            else:
                skip = min(period - cycle_position, length - position)
                position += skip
                phase += skip


def draw_overlay(
    frame: np.ndarray,
    points: list[np.ndarray],
    angle: float,
    source_time: float,
    selected_time: float,
) -> None:
    dashed_polyline(frame, points, (255, 30, 255), 3)
    cv2.circle(
        frame,
        tuple(np.round(points[0]).astype(int)),
        7,
        (255, 30, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (20, 20), (475, 105), (15, 15, 15), -1)
    cv2.putText(
        frame,
        "PREDICTED PATH (DASHED)",
        (38, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 30, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FINAL AIM: {angle:+.2f} deg   selected {selected_time:.2f}s",
        (38, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"source {source_time:.2f}s",
        (frame.shape[1] - 205, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把最终选定角度的预测反弹轨迹叠加到实录视频"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--aim-frame", type=int, required=True)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--stabilise-frames", type=int, default=12)
    parser.add_argument("--ball-radius", type=float, default=7.0)
    parser.add_argument("--max-collisions", type=int, default=40)
    parser.add_argument("--reflection-bias", type=float, default=0.0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector = BoardDetector(VisionConfig())
    detection, angle, lower_x, sample_count = stabilise_final_aim(
        capture,
        detector,
        args.aim_frame,
        args.stabilise_frames,
    )
    assert detection.board is not None
    board = detection.board
    launch_y = board.top + board.height * 0.94
    direction = direction_from_angle(angle)
    if abs(direction[1]) < 1e-9:
        raise RuntimeError("最终方向接近水平，无法反推发射点")
    travel_from_bottom = (launch_y - board.bottom) / direction[1]
    launch_x = lower_x + travel_from_bottom * direction[0]
    origin = vec(launch_x, launch_y)
    trajectory = simulate_trajectory(
        origin,
        direction,
        board,
        detection.obstacles,
        ball_radius=args.ball_radius,
        max_collisions=args.max_collisions,
        max_distance=board.diagonal * 12.0,
        reflection_bias_deg=args.reflection_bias,
        epsilon=0.05,
    )

    start_frame = max(0, round(args.start_time * fps))
    end_frame = (
        min(total_frames, round(args.end_time * fps))
        if args.end_time is not None
        else total_frames
    )
    if end_frame <= start_frame:
        raise RuntimeError("输出时间范围为空")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{args.output}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    preview: np.ndarray | None = None
    selected_time = args.aim_frame / fps
    for frame_index in range(start_frame, end_frame):
        ok, frame = capture.read()
        if not ok:
            break
        draw_overlay(
            frame,
            trajectory.points,
            angle,
            frame_index / fps,
            selected_time,
        )
        if preview is None or frame_index == min(end_frame - 1, args.aim_frame + round(fps * 3)):
            preview = frame.copy()
        writer.write(frame)
    writer.release()
    capture.release()

    preview_path = args.output.with_suffix(".preview.png")
    if preview is not None:
        cv2.imwrite(str(preview_path), preview)
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "fps": fps,
        "source_frame_range": [start_frame, end_frame],
        "aim_frame": args.aim_frame,
        "selected_time_seconds": selected_time,
        "stabilisation_samples": sample_count,
        "final_angle_degrees": angle,
        "launch_origin": origin.tolist(),
        "collision_frame": [
            board.left,
            board.top,
            board.right,
            board.bottom,
        ],
        "blocks": len(detection.obstacles),
        "predicted_block_hits": trajectory.block_hits,
        "predicted_collisions": len(trajectory.collisions),
        "loop_stopped": trajectory.looped,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
