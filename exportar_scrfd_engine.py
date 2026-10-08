from __future__ import annotations

import argparse
import json
from pathlib import Path

from access_control.tensorrt_session import build_scrfd_engine


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Converter SCRFD ONNX para TensorRT FP16 (640x640)")
    parser.add_argument("--onnx", type=Path, default=ROOT / "models/insightface_buffalo_l/det_10g.onnx")
    parser.add_argument("--engine", type=Path, default=ROOT / "models/insightface_buffalo_l/det_10g_fp16.engine")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recriar uma engine existente")
    args = parser.parse_args()
    if args.engine.exists() and not args.force:
        parser.error("A engine ja existe; use --force para recriar")
    if args.workspace_mb < 1:
        parser.error("--workspace-mb deve ser positivo")
    print("Construindo engine SCRFD; a primeira exportacao pode levar alguns minutos...", flush=True)
    metadata = build_scrfd_engine(args.onnx.resolve(), args.engine.resolve(),
                                  fp16=not args.fp32, workspace_mb=args.workspace_mb)
    print(json.dumps({"engine": str(args.engine.resolve()), **metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
