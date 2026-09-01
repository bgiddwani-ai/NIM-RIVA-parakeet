#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Offline NeMo ASR benchmark.

Loads a local .nemo file, replicates one WAV according to the requested
batch size, performs warmup iterations, then measures inference throughput.

RTFX = total audio duration represented by batch / inference wall time

Example:

python benchmark_nemo_asr.py \
    --model /data/stt_en_fastconformer_ctc_large.nemo \
    --audio /data/test.wav \
    --batch-size 32 \
    --warmup-steps 5 \
    --run-steps 20 \
    --device cuda
"""

import argparse
import inspect
import os
import statistics
import time

import soundfile as sf
import torch

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging


def get_audio_duration(audio_path):
    info = sf.info(audio_path)
    return info.frames / float(info.samplerate), info.samplerate, info.channels


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def transcribe_compat(
    model,
    audio_files,
    batch_size,
    num_workers=0,
):
    """
    Compatibility wrapper for older and newer NeMo versions.

    Older NeMo:
        model.transcribe(
            paths2audio_files,
            batch_size=...
        )

    Newer NeMo:
        model.transcribe(
            audio=...,
            override_config=...
        )
    """

    signature = inspect.signature(model.transcribe)
    params = signature.parameters

    #
    # Newer NeMo API
    #
    if "override_config" in params:

        cfg = model.get_transcribe_config()

        if hasattr(cfg, "batch_size"):
            cfg.batch_size = batch_size

        if hasattr(cfg, "num_workers"):
            cfg.num_workers = num_workers

        if "audio" in params:
            return model.transcribe(
                audio=audio_files,
                override_config=cfg,
            )

    #
    # Older NeMo API
    #
    kwargs = {}

    if "batch_size" in params:
        kwargs["batch_size"] = batch_size

    if "num_workers" in params:
        kwargs["num_workers"] = num_workers

    if "verbose" in params:
        kwargs["verbose"] = False

    return model.transcribe(
        audio_files,
        **kwargs,
    )


def get_text(output):
    if hasattr(output, "text"):
        return output.text

    return str(output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to local .nemo model",
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Input WAV",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--run-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use FP16 autocast on CUDA",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model does not exist: {args.model}"
        )

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(
            f"Audio does not exist: {args.audio}"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.run_steps < 1:
        raise ValueError(
            "--run-steps must be >= 1"
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but CUDA is unavailable"
            )

        device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")

    # ---------------------------------------------------------
    # Audio
    # ---------------------------------------------------------

    duration, sample_rate, channels = get_audio_duration(
        args.audio
    )

    audio_batch = [
        args.audio
        for _ in range(args.batch_size)
    ]

    batch_audio_duration = (
        duration * args.batch_size
    )

    print()
    print("=" * 70)
    print("Audio")
    print("=" * 70)

    print(
        f"File                     : {args.audio}"
    )
    print(
        f"Duration                 : {duration:.3f} sec"
    )
    print(
        f"Sample rate              : {sample_rate} Hz"
    )
    print(
        f"Channels                 : {channels}"
    )

    if sample_rate != 16000:
        logging.warning(
            "Input WAV is not 16 kHz."
        )

    # ---------------------------------------------------------
    # Load offline model
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading offline NeMo model")
    print("=" * 70)

    start_load = time.perf_counter()

    model = ASRModel.restore_from(
        restore_path=args.model,
        map_location=device,
    )

    model = model.to(device)
    model.eval()

    cuda_sync(device)

    load_time = time.perf_counter() - start_load

    print(
        f"Model                    : {args.model}"
    )
    print(
        f"Model class              : {type(model).__name__}"
    )
    print(
        f"Device                   : {device}"
    )
    print(
        f"Load time                : {load_time:.3f} sec"
    )

    if device.type == "cuda":
        print(
            f"GPU                      : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print()
    print(
        "transcribe() signature:"
    )
    print(
        inspect.signature(model.transcribe)
    )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Benchmark configuration")
    print("=" * 70)

    print(
        f"Batch size               : {args.batch_size}"
    )
    print(
        f"Audio / item             : {duration:.3f} sec"
    )
    print(
        f"Audio / batch            : "
        f"{batch_audio_duration:.3f} sec"
    )
    print(
        f"Warmup iterations        : {args.warmup_steps}"
    )
    print(
        f"Measured iterations      : {args.run_steps}"
    )
    print(
        f"AMP                      : {args.amp}"
    )

    # ---------------------------------------------------------
    # Precision context
    # ---------------------------------------------------------

    if args.amp and device.type != "cuda":
        logging.warning(
            "--amp ignored on CPU"
        )

    use_amp = (
        args.amp and device.type == "cuda"
    )

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Warmup")
    print("=" * 70)

    with torch.inference_mode():

        for i in range(args.warmup_steps):

            cuda_sync(device)

            start = time.perf_counter()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                _ = transcribe_compat(
                    model=model,
                    audio_files=audio_batch,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

            cuda_sync(device)

            elapsed = (
                time.perf_counter() - start
            )

            rtfx = (
                batch_audio_duration / elapsed
            )

            print(
                f"Warmup {i + 1:02d}: "
                f"{elapsed:.4f} sec | "
                f"RTFX {rtfx:.2f}x"
            )

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Measured iterations")
    print("=" * 70)

    times = []
    rtfx_values = []

    output = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    with torch.inference_mode():

        for i in range(args.run_steps):

            cuda_sync(device)

            start = time.perf_counter()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                output = transcribe_compat(
                    model=model,
                    audio_files=audio_batch,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )

            cuda_sync(device)

            elapsed = (
                time.perf_counter() - start
            )

            rtfx = (
                batch_audio_duration / elapsed
            )

            times.append(elapsed)
            rtfx_values.append(rtfx)

            print(
                f"Iteration {i + 1:02d}: "
                f"{elapsed:.4f} sec | "
                f"RTFX {rtfx:.2f}x"
            )

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------

    total_inference_time = sum(times)

    total_audio_processed = (
        batch_audio_duration
        * args.run_steps
    )

    aggregate_rtfx = (
        total_audio_processed
        / total_inference_time
    )

    mean_time = statistics.mean(times)
    median_time = statistics.median(times)

    mean_rtfx = statistics.mean(
        rtfx_values
    )

    median_rtfx = statistics.median(
        rtfx_values
    )

    std_rtfx = (
        statistics.stdev(rtfx_values)
        if len(rtfx_values) > 1
        else 0.0
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Batch size               : {args.batch_size}"
    )

    print(
        f"Audio / item             : "
        f"{duration:.3f} sec"
    )

    print(
        f"Audio / batch            : "
        f"{batch_audio_duration:.3f} sec"
    )

    print()

    print(
        f"Mean batch latency       : "
        f"{mean_time * 1000:.2f} ms"
    )

    print(
        f"Median batch latency     : "
        f"{median_time * 1000:.2f} ms"
    )

    print()

    print(
        f"Mean RTFX                : "
        f"{mean_rtfx:.2f}x"
    )

    print(
        f"Median RTFX              : "
        f"{median_rtfx:.2f}x"
    )

    print(
        f"RTFX stddev              : "
        f"{std_rtfx:.2f}"
    )

    print(
        f"Min RTFX                 : "
        f"{min(rtfx_values):.2f}x"
    )

    print(
        f"Max RTFX                 : "
        f"{max(rtfx_values):.2f}x"
    )

    print()

    print(
        f"Aggregate RTFX           : "
        f"{aggregate_rtfx:.2f}x"
    )

    print(
        f"Total audio processed    : "
        f"{total_audio_processed:.3f} sec"
    )

    print(
        f"Total inference time     : "
        f"{total_inference_time:.3f} sec"
    )

    # ---------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------

    if device.type == "cuda":

        peak_allocated = (
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved(
                device
            )
            / (1024 ** 3)
        )

        print()

        print(
            f"Peak GPU allocated       : "
            f"{peak_allocated:.3f} GB"
        )

        print(
            f"Peak GPU reserved        : "
            f"{peak_reserved:.3f} GB"
        )

    # ---------------------------------------------------------
    # Transcript
    # ---------------------------------------------------------

    if output:

        print()
        print("=" * 70)
        print("Transcript - first batch item")
        print("=" * 70)

        print(
            get_text(output[0])
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
