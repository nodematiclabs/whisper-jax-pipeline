import kfp
from kfp import compiler

import kfp.dsl as dsl

@dsl.container_component
def whisper_jax():
    return dsl.ContainerSpec(
        image='us-central1-docker.pkg.dev/PROJECT/REPOSITORY/IMAGE:TAG',
        args=[
            "--audio_file",
            "gs://BUCKET/example.flac",
            "--transcript_file",
            "gs://BUCKET/example.txt",
        ]
    )

@dsl.pipeline(
    name="whisper-jax"
)
def transcribe_pipeline():
    task = whisper_jax()
    task.set_cpu_request("4")
    task.set_cpu_limit("4")
    task.set_memory_request("32Gi")
    task.set_memory_limit("32Gi")
    task.set_accelerator_type("NVIDIA_TESLA_T4")
    task.set_accelerator_limit(1)

compiler.Compiler().compile(transcribe_pipeline, 'pipeline.json')