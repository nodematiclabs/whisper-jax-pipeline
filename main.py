import argparse
import os
import time

from whisper_jax import FlaxWhisperPipline
from google.cloud import storage

def transcribe_audio(audio_file):
    pipeline = FlaxWhisperPipline("openai/whisper-large-v2")
    return pipeline(audio_file)

def gcs_copy_local(gs_path):
    client = storage.Client()
    bucket_name, blob_path = gs_path.replace("gs://", "").split("/", 1)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    local_path = os.path.basename(blob_path)
    with open(local_path, "wb") as f:
        blob.download_to_file(f)

    return local_path

def gcs_copy_remote(local_file, gs_path):
    client = storage.Client()
    bucket_name, blob_path = gs_path.replace("gs://", "").split("/", 1)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with open(local_file, "rb") as f:
        blob.upload_from_file(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', required=True)
    parser.add_argument('--transcript_file', required=True)
    args = parser.parse_args()

    local_audio_file = gcs_copy_local(args.audio_file)
    start_time = time.time()
    transcript = transcribe_audio(local_audio_file)
    end_time = time.time()
    local_transcript_file = os.path.splitext(local_audio_file)[0] + ".txt"
    with open(local_transcript_file, 'w') as f:
        f.write(transcript["text"])
    gcs_copy_remote(local_transcript_file, args.transcript_file)
    transcription_time = end_time - start_time
    print(f"Completed with transcription time: {transcription_time:.3f} seconds") 