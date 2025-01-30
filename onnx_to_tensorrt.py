from pathlib import Path

import click
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
)


@click.command()
@click.argument("onnx_path", type=click.Path(exists=True, path_type=Path))
@click.option("--fp16", is_flag=True, help="Enable FP16 precision")
def build_tensorrt_engine(onnx_path: Path, fp16: bool):
    """Build and save a TensorRT engine from an ONNX model."""
    # check if file exists
    if not onnx_path.exists():
        click.echo(f"File {onnx_path} does not exist")
        return
    network_loader = NetworkFromOnnxPath(str(onnx_path))
    engine_loader = EngineFromNetwork(network_loader, config=CreateConfig(fp16=fp16))
    engine = engine_loader()
    with open(onnx_path.with_suffix(".engine"), "wb") as f:
        f.write(engine.serialize())
    click.echo(f"Engine saved at {onnx_path.with_suffix('.engine')}")


if __name__ == "__main__":
    build_tensorrt_engine()
