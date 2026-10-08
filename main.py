from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from access_control.pipeline import PipelineConfig, run_pipeline
from access_control.esp32 import DEFAULT_ESP_URL
from access_control.faces import RECOGNIZER_THRESHOLDS


ROOT = Path(__file__).resolve().parent
RECOGNIZER_MODELS = {
    "sface": ROOT / "models" / "face_recognition_sface_2021dec.onnx",
    "arcface": ROOT / "models" / "insightface_buffalo_l" / "w600k_r50.onnx",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconhecimento facial + deteccao de pessoas + tracking OC-SORT"
    )
    parser.add_argument("--input", required=True, help="DAV/MP4, URL RTSP ou indice da webcam")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "execucao")
    parser.add_argument("--model", "--pose-model", dest="pose_model", type=Path,
                        default=ROOT.parent / "models" / "yolo_v26" / ".engine" / "yolo26x_fp16.engine")
    parser.add_argument("--device", default="0", help="0 para GPU ou cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--face-recognizer", choices=tuple(RECOGNIZER_MODELS), default="arcface",
                        help="Modelo de reconhecimento facial")
    parser.add_argument("--face-threshold", type=float, default=None,
                        help="Similaridade minima; padrao por modelo: " + ", ".join(
                            f"{name} {value}" for name, value in RECOGNIZER_THRESHOLDS.items()))
    parser.add_argument("--face-detector", choices=("scrfd", "yunet"), default="scrfd")
    parser.add_argument("--face-runtime", choices=("auto", "cuda", "tensorrt"), default="auto",
                        help="SCRFD/ArcFace: auto usa a engine local quando compativel; cuda usa ONNX")
    parser.add_argument("--face-engine", type=Path,
                        default=ROOT / "models/insightface_buffalo_l/det_10g_fp16.engine")
    parser.add_argument("--face-recognizer-engine", type=Path,
                        default=ROOT / "models/insightface_buffalo_l/w600k_r50_fp16.engine",
                        help="Engine TensorRT do ArcFace")
    parser.add_argument("--face-detect-score", type=float, default=0.50)
    parser.add_argument("--face-interval", type=int, default=3)
    parser.add_argument("--face-recheck-interval", type=int, default=0,
                        help="Reverificar IDs conhecidos apos N quadros; 0 desativa")
    parser.add_argument("--face-sync", action="store_true",
                        help="Processamento facial sincrono tambem em cameras ao vivo")
    parser.add_argument("--face-retry-seconds", type=float, default=0.5,
                        help="Intervalo minimo entre tentativas por pessoa no modo ao vivo assincrono")
    parser.add_argument("--face-max-tracks", type=int, default=1,
                        help="Pessoas por tarefa facial no modo ao vivo assincrono")
    parser.add_argument("--face-rotations", type=int, nargs="+", choices=(-90, 0, 90, 180),
                        default=[0, 90, 180, -90], help="Rotacoes testadas pelo SCRFD")
    parser.add_argument("--tracker-iou", type=float, default=0.25)
    parser.add_argument("--tracker-max-age", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processa tudo")
    parser.add_argument("--max-fps", type=float, default=0.0,
                        help="Limite de processamento ao vivo; 0 usa o FPS informado pela camera")
    parser.add_argument("--start-frame", type=int, default=0, help="quadro inicial para videos")
    parser.add_argument("--rotate", type=int, choices=(-90, 0, 90, 180), default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--preview-max-side", type=int, default=1280,
                        help="Maior lado da previa; 0 envia a imagem completa para a janela")
    parser.add_argument("--capture-threads", type=int, default=1,
                        help="Threads FFmpeg da camera; 0 usa a escolha automatica do decodificador")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "acessos.sqlite3")
    parser.add_argument("--line-config", type=Path, default=ROOT / "outputs" / "linha" / "linha.json",
                        help="JSON da linha; usa as coordenadas padrao se o arquivo nao existir")
    parser.add_argument("--invert-line", action="store_true", help="Inverte entrada e saida da linha")
    parser.add_argument("--esp-url", default=DEFAULT_ESP_URL, help="Endereco HTTP do ESP32")
    parser.add_argument("--no-esp", action="store_true", help="Executa sem enviar comandos ao ESP32")
    parser.add_argument("--alarm-seconds", type=float, default=3.0,
                        help="Duracao do buzzer para cruzamentos desconhecidos")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--save-video", action="store_true", help="Gravar resultado.mp4 (desativado por padrao)")
    video.add_argument("--no-video", action="store_true", help="Compatibilidade: nao gravar video")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.max_fps) or args.max_fps < 0:
        parser.error("--max-fps deve ser finito e maior ou igual a zero")
    if args.capture_threads < 0:
        parser.error("--capture-threads deve ser maior ou igual a zero")
    config = PipelineConfig(
        input_source=args.input,
        output_dir=args.output.resolve(),
        pose_model=args.pose_model.resolve(),
        face_detector_model=(ROOT / "models" / "insightface_buffalo_l" / "det_10g.onnx"
                             if args.face_detector == "scrfd" else
                             ROOT / "models" / "face_detection_yunet_2023mar.onnx"),
        face_recognizer_model=RECOGNIZER_MODELS[args.face_recognizer],
        face_recognizer_backend=args.face_recognizer,
        face_recognizer_engine=args.face_recognizer_engine.resolve(),
        gallery_dir=ROOT / "faces",
        device=args.device,
        image_size=args.imgsz,
        pose_confidence=args.pose_conf,
        face_threshold=args.face_threshold,
        face_detector_backend=args.face_detector,
        face_runtime=args.face_runtime,
        face_engine_model=args.face_engine.resolve(),
        face_detect_score=args.face_detect_score,
        face_interval=args.face_interval,
        face_recheck_interval=max(0, args.face_recheck_interval),
        face_async=not args.face_sync,
        face_retry_seconds=max(0.0, args.face_retry_seconds),
        face_max_tracks=max(1, args.face_max_tracks),
        face_rotations=tuple(dict.fromkeys(args.face_rotations)),
        tracker_iou=args.tracker_iou,
        tracker_max_age=args.tracker_max_age,
        show=args.show,
        preview_max_side=max(0, args.preview_max_side),
        capture_threads=args.capture_threads,
        save_video=args.save_video,
        rotate=args.rotate,
        start_frame=max(0, args.start_frame),
        max_frames=args.max_frames,
        max_fps=args.max_fps,
        database=args.database.resolve(),
        line_config=args.line_config.resolve(),
        invert_line=args.invert_line,
        esp_url=None if args.no_esp else args.esp_url,
        alarm_seconds=args.alarm_seconds,
    )
    print(json.dumps(run_pipeline(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

