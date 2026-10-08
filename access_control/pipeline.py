from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from .access import AccessController, signal_for_tracks
from .access_store import AccessStore
from .drawing import draw_access_overlay, draw_scene
from .capture import is_live_source, open_capture
from .display import WINDOW_NAME, PreviewRenderer, create_display_window, poll_display_key
from .faces import FaceRecognizer, IdentityMemory
from .face_worker import FaceMetrics, FaceWorker, project_faces
from .esp32 import DEFAULT_ESP_URL, EspGpioClient
from .line_crossing import LineCounter, LineSettings
from .pose import PoseDetector
from .recording import RealtimeVideoWriter
from .tracking import OCSortPersonTracker
from .timing import FrameRateLimiter, FrameRateMeter, processing_fps_limit


@dataclass(slots=True)
class PipelineConfig:
    input_source: str
    output_dir: Path
    pose_model: Path
    face_detector_model: Path
    face_recognizer_model: Path
    gallery_dir: Path
    device: str = "0"
    image_size: int = 640
    pose_confidence: float = 0.25
    face_threshold: float | None = None
    face_detector_backend: str = "scrfd"
    face_recognizer_backend: str = "sface"
    face_recognizer_engine: Path | None = None
    face_detect_score: float = 0.50
    face_interval: int = 3
    face_recheck_interval: int = 0
    face_async: bool = True
    face_retry_seconds: float = 0.5
    face_max_tracks: int = 1
    face_rotations: tuple[int, ...] = (0, 90, 180, -90)
    face_runtime: str = "auto"
    face_engine_model: Path | None = None
    tracker_iou: float = 0.25
    tracker_max_age: int = 20
    show: bool = False
    save_video: bool = False
    rotate: int = 0
    start_frame: int = 0
    max_frames: int = 0
    max_fps: float = 0.0
    preview_max_side: int = 1280
    capture_threads: int = 1
    database: Path = Path(__file__).resolve().parents[1] / "data" / "acessos.sqlite3"
    line_config: Path | None = Path(__file__).resolve().parents[1] / "outputs" / "linha" / "linha.json"
    invert_line: bool = False
    esp_url: str | None = DEFAULT_ESP_URL
    alarm_seconds: float = 3.0


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    live = is_live_source(config.input_source)
    settings = LineSettings.load(config.line_config, config.invert_line)
    with ExitStack() as resources:
        store = AccessStore(config.database)
        resources.callback(store.close)
        esp = EspGpioClient(config.esp_url) if config.esp_url else None
        if esp is not None:
            resources.callback(esp.close)
        capture = open_capture(config.input_source, decoder_threads=config.capture_threads)
        resources.callback(capture.release)
        resources.callback(cv2.destroyAllWindows)
        return _run_pipeline(config, capture, live, store, settings, esp)


def _run_pipeline(config, capture, live, store, settings, esp):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(config.rotate) == 90:
        width, height = height, width
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps_limit = processing_fps_limit(reported_fps, config.max_fps, live)
    source_fps = reported_fps
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 10.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) if not live else 0
    total_frames = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else 0
    if config.start_frame > 0 and not live:
        capture.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    pose_detector = PoseDetector(config.pose_model, config.device, config.image_size, config.pose_confidence)
    face_recognizer = FaceRecognizer(
        config.face_detector_model,
        config.face_recognizer_model,
        config.gallery_dir,
        threshold=config.face_threshold,
        detect_score=config.face_detect_score,
        detector_backend=config.face_detector_backend,
        rotations=config.face_rotations,
        runtime_backend=config.face_runtime,
        engine_model=config.face_engine_model,
        recognizer_backend=config.face_recognizer_backend,
        recognizer_engine=config.face_recognizer_engine,
    )
    tracker = OCSortPersonTracker(iou_threshold=config.tracker_iou, max_age=config.tracker_max_age)
    memory = IdentityMemory()

    video_path = config.output_dir / "resultado.mp4"
    writer = _create_writer(video_path, source_fps, (width, height)) if config.save_video else None
    if writer is not None and live:
        writer = RealtimeVideoWriter(writer, source_fps)
    events_path = config.output_dir / "eventos.jsonl"
    started = time.perf_counter()
    processed = 0
    last_faces = []
    observed_track_ids: set[int] = set()
    last_face_check_by_track: dict[int, int] = {}
    last_face_attempt_time: dict[int, float] = {}
    face_snapshot = []
    face_result_time = 0.0
    face_metrics = FaceMetrics()
    rate_limiter = FrameRateLimiter(fps_limit)
    rate_meter = FrameRateMeter()
    timing_totals: dict[str, float] = defaultdict(float)
    yolo_timing_totals: dict[str, float] = defaultdict(float)
    preview = PreviewRenderer(config.preview_max_side) if config.show else None
    worker = FaceWorker(face_recognizer) if live and config.face_async else None
    face_mode = "async" if worker is not None else "sync"
    print(f"Processamento facial: {face_mode} | intervalo: {config.face_interval} quadros", flush=True)
    print(f"Detector facial: {', '.join(face_recognizer.detector_providers)}", flush=True)
    print(f"Reconhecedor facial: {config.face_recognizer_backend} "
          f"({', '.join(face_recognizer.recognizer_providers)}) | limiar: {face_recognizer.threshold}",
          flush=True)
    display_initialized = False
    access = None
    state = None
    timeline_start = datetime.now().astimezone()

    try:
        with events_path.open("w", encoding="utf-8") as events:
            while True:
                cycle_started = time.perf_counter()
                rate_limiter.wait()
                limiter_ms = (time.perf_counter() - cycle_started) * 1000
                capture_started = time.perf_counter()
                if live and config.show and display_initialized:
                    ok, frame = capture.read(on_wait=lambda: poll_display_key() not in (27, ord("q")))
                else:
                    ok, frame = capture.read()
                capture_ms = (time.perf_counter() - capture_started) * 1000
                if not ok:
                    break
                rotation_started = time.perf_counter()
                timestamp = (datetime.now().astimezone() if live else
                             timeline_start + timedelta(seconds=processed / source_fps))
                frame = _rotate_frame(frame, config.rotate)
                height, width = frame.shape[:2]
                if access is None:
                    counter = LineCounter(*settings.pixels(width, height),
                                          settings.hysteresis, settings.invert)
                    access = AccessController(counter, store, config.alarm_seconds)
                rotation_ms = (time.perf_counter() - rotation_started) * 1000
                frame_started = time.perf_counter()
                detections = pose_detector.detect(frame)
                person_ms = (time.perf_counter() - frame_started) * 1000
                yolo_timings = getattr(pose_detector, "last_timings", {})
                if not isinstance(yolo_timings, dict):
                    yolo_timings = {}
                tracking_started = time.perf_counter()
                tracks = tracker.update(detections)
                tracking_ms = (time.perf_counter() - tracking_started) * 1000
                access_started = time.perf_counter()
                observed_track_ids.update(track.track_id for track in tracks)
                if worker is not None:
                    batch = worker.poll()
                    if batch is not None:
                        # Use boxes from the submitted frame for association. A delayed
                        # face must never be assigned to another person's current box.
                        last_faces = batch.faces
                        face_snapshot = batch.tracks
                        memory.update(last_faces, face_snapshot)
                        face_metrics.record(last_faces, batch.timings)
                        face_result_time = time.monotonic()
                memory.prune(tracker.active_track_ids)
                if esp is not None:
                    initial_identities = {track.track_id: memory.get(track.track_id) for track in tracks}
                    esp.update(signal_for_tracks(tracks, initial_identities), access.alarm_until)

                if (processed % max(1, config.face_interval) == 0
                        and (worker is None or not worker.busy)):
                    tracks_to_check = _tracks_requiring_face_check(
                        tracks,
                        memory,
                        last_face_check_by_track,
                        processed,
                        config.face_recheck_interval,
                    )
                    if worker is not None:
                        now = time.monotonic()
                        tracks_to_check = _limit_face_checks(
                            tracks_to_check, last_face_attempt_time, now,
                            config.face_retry_seconds, config.face_max_tracks,
                        )
                        if worker.submit(frame, tracks_to_check, tracks, processed):
                            for track in tracks_to_check:
                                last_face_attempt_time[track.track_id] = now
                                last_face_check_by_track[track.track_id] = processed
                    else:
                        last_faces = (face_recognizer.detect_and_recognize(frame, tracks_to_check)
                                      if tracks_to_check else [])
                        memory.update(last_faces, tracks)
                        if tracks_to_check:
                            face_metrics.record(last_faces, dict(face_recognizer.last_timings))
                        for track in tracks_to_check:
                            last_face_check_by_track[track.track_id] = processed
                memory.prune(tracker.active_track_ids)
                for track_id in list(last_face_check_by_track):
                    if track_id not in tracker.active_track_ids:
                        del last_face_check_by_track[track_id]
                        last_face_attempt_time.pop(track_id, None)
                identities = {track.track_id: memory.get(track.track_id) for track in tracks}

                state = access.update(tracks, identities, tracker.active_track_ids, timestamp)
                if esp is not None:
                    esp.update(state.lamp, state.alarm_until)
                for crossing in state.events:
                    name = crossing["nome"] or "Desconhecido"
                    print(f"ID {crossing['tracking_id']} | {name} | {crossing['direcao']} | "
                          f"{crossing['acao']}", flush=True)

                elapsed = max(time.perf_counter() - frame_started, 1e-9)
                access_ms = (time.perf_counter() - access_started) * 1000
                drawing_started = time.perf_counter()
                display_faces = last_faces
                if worker is not None:
                    display_faces = (project_faces(last_faces, face_snapshot, tracks)
                                     if time.monotonic() - face_result_time <= 1.0 else [])
                delivered_fps = rate_meter.update()
                render_frame = frame
                coordinate_scale = (1.0, 1.0)
                if preview is not None and writer is None:
                    render_frame = preview.prepare(frame)
                    coordinate_scale = (render_frame.shape[1] / width, render_frame.shape[0] / height)
                annotated = draw_scene(render_frame, detections, tracks, identities, display_faces,
                                       delivered_fps, processing_ms=elapsed * 1000,
                                       coordinate_scale=coordinate_scale)
                draw_access_overlay(annotated, tracks, access.counter, state,
                                    esp.status if esp is not None else None,
                                    coordinate_scale=coordinate_scale)
                drawing_ms = (time.perf_counter() - drawing_started) * 1000
                video_started = time.perf_counter()
                if writer is not None:
                    writer.write(annotated)
                video_ms = (time.perf_counter() - video_started) * 1000
                display_started = time.perf_counter()
                if config.show:
                    displayed = preview.prepare(annotated) if writer is not None else annotated
                    if not display_initialized:
                        create_display_window(displayed)
                        display_initialized = True
                    cv2.imshow(WINDOW_NAME, displayed)
                    if poll_display_key() in (27, ord("q")):
                        break
                display_ms = (time.perf_counter() - display_started) * 1000
                frame_timings = {
                    "person_detection": person_ms, "tracking": tracking_ms,
                    "frame": elapsed * 1000, "limiter_wait": limiter_ms,
                    "capture_wait": capture_ms, "rotation": rotation_ms,
                    "access_and_faces": access_ms, "drawing": drawing_ms,
                    "video_write": video_ms, "display": display_ms,
                }
                capture_stats = _capture_stats(capture)
                event_started = time.perf_counter()

                event = {
                    "frame": config.start_frame + processed,
                    "time_seconds": round(time.perf_counter() - started if live else
                                          (config.start_frame + processed) / source_fps, 3),
                    "tracks": [
                        {
                            "track_id": track.track_id,
                            "bbox_xyxy": [round(float(value), 2) for value in track.bbox_xyxy],
                            "pose_confidence": round(track.confidence, 4),
                            "identity": identities[track.track_id].label,
                            "identity_score": round(identities[track.track_id].score, 4),
                            "known": identities[track.track_id].known,
                            "keypoints_xyc": [
                                [round(float(value), 2) for value in row]
                                for row in detections[track.pose_index].keypoints_xyc
                            ],
                        }
                        for track in tracks
                    ],
                    "crossings": state.events,
                    "processing_fps": round(delivered_fps, 3),
                    "capture": capture_stats,
                    "timings_ms": {name: round(value, 3) for name, value in frame_timings.items()},
                    "yolo_timings_ms": yolo_timings,
                    "face_processing": {"mode": face_mode,
                                        "busy": worker.busy if worker is not None else False,
                                        "last_ms": {name: round(value, 3)
                                                    for name, value in face_metrics.latest.items()}},
                    "access": {"entries": state.entries, "exits": state.exits,
                               "open_visits": state.open_visits, "signal": state.lamp,
                               "alarm": state.alarm_until > time.monotonic()},
                }
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                if state.events:
                    events.flush()
                frame_timings["event_write"] = (time.perf_counter() - event_started) * 1000
                frame_timings["cycle"] = (time.perf_counter() - cycle_started) * 1000
                for name, value in frame_timings.items():
                    timing_totals[name] += value
                for name, value in yolo_timings.items():
                    yolo_timing_totals[name] += value
                processed += 1
                if processed % 25 == 0:
                    print(f"Processados {processed}/{total_frames or '?'} quadros", flush=True)
                if config.max_frames > 0 and processed >= config.max_frames:
                    break
    finally:
        try:
            if worker is not None:
                worker.close()
                batch = worker.poll()
                if batch is not None:
                    face_metrics.record(batch.faces, batch.timings)
        finally:
            if writer is not None:
                writer.release()

    wall_time = time.perf_counter() - started
    summary = {
        "input": config.input_source,
        "frames_processed": processed,
        "source_fps": source_fps,
        "processing_fps_limit": fps_limit or None,
        "resolution": [width, height],
        "unique_track_ids": len(observed_track_ids),
        "gallery_identities": sorted(face_recognizer.gallery),
        "face_detector": config.face_detector_backend,
        "face_detect_score": config.face_detect_score,
        "face_recognizer": config.face_recognizer_backend,
        "face_threshold": face_recognizer.threshold,
        "face_recheck_interval": config.face_recheck_interval,
        "face_processing": {"mode": face_mode,
                            "detector_providers": list(face_recognizer.detector_providers),
                            "recognizer_providers": list(face_recognizer.recognizer_providers),
                            "runtime_requested": config.face_runtime,
                            "retry_seconds": config.face_retry_seconds if worker is not None else None,
                            "max_tracks_per_job": config.face_max_tracks if worker is not None else None,
                            "rotations": config.face_rotations,
                            **face_metrics.summary()},
        "average_timings_ms": {name: round(value / processed, 3)
                               for name, value in timing_totals.items()} if processed else {},
        "average_yolo_timings_ms": {name: round(value / processed, 3)
                                    for name, value in yolo_timing_totals.items()} if processed else {},
        "capture": _capture_stats(capture),
        "preview_max_side": config.preview_max_side if config.show else None,
        "capture_threads_requested": config.capture_threads if live else None,
        "processing_seconds": round(wall_time, 3),
        "average_fps": round(processed / wall_time, 3) if wall_time else 0.0,
        "output_video": str(video_path) if writer is not None else None,
        "events": str(events_path),
        "database": str(store.path),
        "run_id": access.run_id if access else None,
        "line_normalized": settings.normalized,
        "line_invert": settings.invert,
        "entries": state.entries if state else 0,
        "exits": state.exits if state else 0,
        "open_visits": store.open_count,
        "unknown_crossings": access.unknown_crossings if access else 0,
        "esp32": esp.status if esp is not None else {"enabled": False},
        "access_clock": "system" if live else "video_timeline",
    }
    (config.output_dir / "resumo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _capture_stats(capture) -> dict:
    reader = getattr(capture, "get_stats", None)
    stats = reader() if callable(reader) else {}
    return stats if isinstance(stats, dict) else {}


def _tracks_requiring_face_check(
    tracks,
    memory: IdentityMemory,
    last_check_by_track: dict[int, int],
    frame_index: int,
    recheck_interval: int,
):
    selected = []
    for track in tracks:
        if not memory.get(track.track_id).known:
            selected.append(track)
            continue
        last_check = last_check_by_track.get(track.track_id, frame_index)
        if recheck_interval > 0 and frame_index - last_check >= recheck_interval:
            selected.append(track)
    return selected


def _limit_face_checks(tracks, last_attempt: dict[int, float], now: float,
                       retry_seconds: float, max_tracks: int):
    # Oldest attempted track goes first so an unknown person cannot starve others.
    eligible = [track for track in tracks
                if now - last_attempt.get(track.track_id, -math.inf) >= max(0.0, retry_seconds)]
    eligible.sort(key=lambda track: last_attempt.get(track.track_id, -math.inf))
    return eligible[:max(1, max_tracks)]


def _create_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video: {path}")
    return writer


def _rotate_frame(frame, angle: int):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(angle) == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

