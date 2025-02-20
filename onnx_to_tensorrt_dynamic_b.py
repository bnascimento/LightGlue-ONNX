from pathlib import Path

import click
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    Profile
)


@click.command()
@click.argument("onnx_path", type=click.Path(exists=True, path_type=Path))
@click.option("--fp16", is_flag=True, help="Enable FP16 precision")
@click.option("--output", type=click.Path(path_type=Path), help="Output path for the TensorRT engine file")
@click.option("--min-batch", default=1, help="Minimum batch size to optimize for")
@click.option("--opt-batch", default=8, help="Optimal batch size to optimize for")
@click.option("--max-batch", default=16, help="Maximum batch size to optimize for")
def build_tensorrt_engine(onnx_path: Path, fp16: bool, output: Path, min_batch: int, opt_batch: int, max_batch: int):
    """Build and save a TensorRT engine from an ONNX model."""
    # Check if file exists
    if not onnx_path.exists():
        click.echo(f"File {onnx_path} does not exist")
        return

    network_loader = NetworkFromOnnxPath(str(onnx_path))

    profile = Profile()
    profile.add("input", 
        min=(min_batch, 2, 1024, 1024),  
        opt=(opt_batch, 8, 1024, 1024), 
        max=(max_batch, 16, 1024, 1024)
    )
    engine_loader = EngineFromNetwork(
        network_loader,
        config=CreateConfig(
            fp16=fp16,
            profiles=[profile]
        )
    )
    engine = engine_loader()

    # Determine output path
    output_path = output if output else onnx_path.with_suffix(".engine")

    with open(output_path, "wb") as f:
        f.write(engine.serialize())

    click.echo(f"Engine saved at {output_path}")


if __name__ == "__main__":
    build_tensorrt_engine()
