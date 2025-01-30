from pathlib import Path

import click
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
)


@click.command()
@click.argument("onnx_path", type=click.Path(exists=True, path_type=Path))
@click.option("--fp16", is_flag=True, help="Enable FP16 precision")
def build_tensorrt_engine(onnx_path: Path, fp16: bool):
    """Build and save a TensorRT engine from an ONNX model."""
    build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_path)), config=CreateConfig(fp16=fp16))
    SaveEngine(build_engine, str(onnx_path.with_suffix(".engine")))
    click.echo(f"Engine saved at {onnx_path.with_suffix('.engine')}")


if __name__ == "__main__":
    build_tensorrt_engine()
