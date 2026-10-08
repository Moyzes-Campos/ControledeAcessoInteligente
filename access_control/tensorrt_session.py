from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]


def build_scrfd_engine(onnx_path: Path, engine_path: Path, size: int = 640,
                       fp16: bool = True, workspace_mb: int = 1024) -> dict:
    import torch
    import tensorrt as trt

    if not torch.cuda.is_available():
        raise RuntimeError("A exportacao TensorRT requer uma GPU NVIDIA com CUDA")
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    source = onnx_path.read_bytes()
    if not parser.parse(source):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Falha ao converter o ONNX para TensorRT:\n{errors}")
    if network.num_inputs != 1 or network.num_outputs != 9:
        raise ValueError("A exportacao espera o SCRFD com uma entrada e nove saidas")
    input_tensor = network.get_input(0)
    shape = (1, 3, size, size)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, shape, shape, shape)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    config.add_optimization_profile(profile)
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("A GPU nao oferece FP16 rapido; use --fp32")
        config.set_flag(trt.BuilderFlag.FP16)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT nao conseguiu construir a engine SCRFD")
    engine_bytes = bytes(serialized)
    metadata = {
        "onnx_sha256": hashlib.sha256(source).hexdigest(),
        "engine_sha256": hashlib.sha256(engine_bytes).hexdigest(),
        "tensorrt_version": trt.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "input_shape": list(shape),
        "output_names": [network.get_output(i).name for i in range(network.num_outputs)],
        "precision": "fp16" if fp16 else "fp32",
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = engine_path.with_suffix(engine_path.suffix + ".tmp")
    temporary.write_bytes(engine_bytes)
    temporary.replace(engine_path)
    engine_path.with_suffix(engine_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


class TensorRTSession:
    """Small session adapter preserving SCRFD's ONNX output order and postprocessing.

    CUDA buffers and a separate stream belong to this instance. The facial worker
    is its only caller, so TensorRT never shares an execution context with YOLO.
    """

    def __init__(self, engine_path: Path, onnx_path: Path,
                 input_size: tuple[int, int] = (640, 640)):
        import torch
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("A engine facial requer CUDA")
        metadata = json.loads(engine_path.with_suffix(engine_path.suffix + ".json").read_text(encoding="utf-8"))
        data = engine_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != metadata["engine_sha256"]:
            raise ValueError("A engine facial nao corresponde aos metadados de exportacao")
        if hashlib.sha256(onnx_path.read_bytes()).hexdigest() != metadata["onnx_sha256"]:
            raise ValueError("A engine facial foi gerada para outro modelo ONNX; exporte novamente")
        if (metadata["tensorrt_version"] != trt.__version__
                or metadata["gpu_name"] != torch.cuda.get_device_name(0)
                or metadata["compute_capability"] != list(torch.cuda.get_device_capability(0))):
            raise ValueError("A engine facial foi gerada em outro ambiente; exporte neste computador")
        shape = (1, 3, input_size[1], input_size[0])
        if tuple(metadata["input_shape"]) != shape:
            raise ValueError(f"A engine facial espera {metadata['input_shape']}, solicitado {shape}")
        self._torch = torch
        self.metadata = metadata
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(data)
        if self._engine is None:
            raise RuntimeError("Falha ao carregar a engine facial")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Falha ao criar o contexto TensorRT facial")
        inputs, outputs = [], []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            (inputs if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or set(outputs) != set(metadata["output_names"]):
            raise ValueError("As entradas/saidas da engine facial nao correspondem aos metadados")
        self._input_name = inputs[0]
        if not self._context.set_input_shape(self._input_name, shape):
            raise ValueError(f"A engine facial nao aceita a entrada {shape}")
        if self._context.infer_shapes():
            raise ValueError("TensorRT nao conseguiu resolver as dimensoes da engine facial")
        self._inputs = [TensorInfo(self._input_name, shape)]
        self._outputs = [TensorInfo(name, tuple(self._context.get_tensor_shape(name)))
                         for name in metadata["output_names"]]
        self._stream = torch.cuda.Stream(device=0)
        self._buffers = {}
        self._host_outputs = {}
        dtype_map = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16}
        with torch.cuda.device(0), torch.cuda.stream(self._stream):
            for item in [*self._inputs, *self._outputs]:
                dtype = dtype_map[np.dtype(trt.nptype(self._engine.get_tensor_dtype(item.name)))]
                self._buffers[item.name] = torch.empty(item.shape, dtype=dtype, device="cuda:0")
                if not self._context.set_tensor_address(item.name, self._buffers[item.name].data_ptr()):
                    raise RuntimeError(f"Falha ao associar o buffer TensorRT: {item.name}")
                if item.name != self._input_name:
                    self._host_outputs[item.name] = torch.empty(item.shape, dtype=dtype, pin_memory=True)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["TensorRTNative"]

    def run(self, output_names, inputs):
        torch = self._torch
        array = np.ascontiguousarray(inputs[self._input_name])
        if array.shape != self._inputs[0].shape:
            raise ValueError(f"Entrada facial {array.shape}; esperado {self._inputs[0].shape}")
        with torch.cuda.device(0), torch.cuda.stream(self._stream):
            self._buffers[self._input_name].copy_(torch.from_numpy(array))
            if not self._context.execute_async_v3(stream_handle=self._stream.cuda_stream):
                raise RuntimeError("Falha na inferencia TensorRT facial")
            for name in output_names:
                self._host_outputs[name].copy_(self._buffers[name], non_blocking=True)
        self._stream.synchronize()
        return [self._host_outputs[name].numpy().copy() for name in output_names]
