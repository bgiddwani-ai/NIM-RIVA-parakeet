# Parakeet CTC 0.6B (FastConformer) on NVIDIA NIM / Riva

Deploy the FastConformer-CTC **parakeet-ctc-0.6b** checkpoint from Hugging Face inside the
`parakeet-1-1b-ctc-en-us` NIM container, then benchmark and run inference against it.

The NIM image ships with its own 1.1B model, but it also contains the full Riva ServiceMaker
toolchain (`nemo2riva`, `riva-build`, `riva-deploy`) and the Riva client binaries. We override the
entrypoint, build the 0.6B model ourselves, and serve that instead.

---

## Contents

- [Prerequisites](#prerequisites)
- [1. Set your NGC API key](#1-set-your-ngc-api-key)
- [2. Launch the container](#2-launch-the-container)
- [3. Download the .nemo checkpoint](#3-download-the-nemo-checkpoint)
- [4. Convert .nemo to .riva](#4-convert-nemo-to-riva)
- [5. Build the RMIR](#5-build-the-rmir)
- [6. Deploy the model repository](#6-deploy-the-model-repository)
- [7. Start the server](#7-start-the-server)
- [8. Prepare LibriSpeech test-clean](#8-prepare-librispeech-test-clean)
- [9. Benchmark the NIM deployment](#9-benchmark-the-nim-deployment)
- [10. NeMo baseline benchmark (optional)](#10-nemo-baseline-benchmark-optional)
- [11. Inference on a single file](#11-inference-on-a-single-file)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU | Volta or newer, ≥16 GB recommended. The example pins GPU index 1. |
| NVIDIA driver + Container Toolkit | `nvidia-smi` must work inside a container. |
| Docker | With `--gpus` support. |
| NGC account + API key | Needed to pull `nvcr.io/nim/...`. |
| Disk | ~30 GB for the image plus artifacts. |

Log in to the registry once on the host:

```bash
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"
```

---

## 1. Set your NGC API key

```bash
export NGC_API_KEY="nvapi-***"
```

Keep this in your shell environment rather than baking it into a script — the `docker run` below
forwards it with `-e NGC_API_KEY` (no value), so it is never written to disk.

---

## 2. Launch the container

```bash
docker run -it --rm --name=parakeet-1-1b-ctc-en-us \
  --gpus '"device=1"' \
  --user=root \
  --entrypoint=/bin/bash \
  --shm-size=8GB \
  -v $PWD:/data \
  -e NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -p 9000:9000 \
  -p 50051:50051 \
  nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:latest
```

What the flags do:

| Flag | Why |
|---|---|
| `--entrypoint=/bin/bash` | Skips the default NIM autostart so we can build a custom model first. |
| `--user=root` | ServiceMaker writes into `/data` and `/opt`; root avoids permission errors. |
| `--gpus '"device=1"'` | Pins to physical GPU 1. Use `'"device=0"'` or `all` as needed. The inner quotes are required. |
| `--shm-size=8GB` | Triton needs shared memory well above Docker's 64 MB default. |
| `-v $PWD:/data` | Your working directory becomes `/data` inside. All artifacts land here and survive `--rm`. |
| `-p 9000:9000` | HTTP / OpenAI-compatible endpoint. |
| `-p 50051:50051` | gRPC endpoint used by the Riva clients. |

Everything from here on runs **inside** the container shell.

---

## 3. Download the .nemo checkpoint

```bash
mkdir -p /data/artifacts

wget -O /data/artifacts/parakeet-ctc-0.6b.nemo \
  "https://huggingface.co/nvidia/parakeet-ctc-0.6b/resolve/main/parakeet-ctc-0.6b.nemo?download=true"
```

This is the FastConformer-CTC Large (~600M parameter) English model, 1024-dim encoder with 8×
depthwise-separable subsampling.

---

## 4. Convert .nemo to .riva

```bash
nemo2riva \
  --out /data/artifacts/parakeet-ctc-0.6b.riva \
  --key nvidia \
  --format onnx \
  --onnx-opset 19 \
  --max-dim 1000 \
  /data/artifacts/parakeet-ctc-0.6b.nemo
```

- `--key nvidia` — encryption key for the `.riva` archive. Whatever you use here must be repeated as
  the `:key` suffix in the next two steps.
- `--format onnx` — export the encoder to ONNX so TensorRT can consume it.
- `--onnx-opset 19` — needed for the FastConformer ops to export cleanly.
- `--max-dim 1000` — maximum time dimension traced during export. At 80 ms per output timestep this
  bounds the longest utterance the graph will accept; raise it if you feed very long audio.

---

## 5. Build the RMIR

```bash
riva-build --config-path=pkg://servicemaker.configs.asr --config-name=offline \
  output_path=/data/artifacts/parakeet-ctc-0.6b.rmir:nvidia \
  'source_path=[/data/artifacts/parakeet-ctc-0.6b.riva:nvidia]' \
  name=parakeet_ctc_600m_offline \
  featurizer.use_utterance_norm_params=False \
  featurizer.precalc_norm_time_steps=0 \
  featurizer.precalc_norm_params=False \
  chunk_size=4.8 \
  left_padding_size=1.6 \
  right_padding_size=1.6 \
  ms_per_timestep=80 \
  nn.max_batch_size=32 \
  featurizer.max_batch_size=512 \
  featurizer.max_execution_batch_size=512 \
  nn.fp16_needs_obey_precision_pass=True \
  decoder=nemo \
  language_code=en-US
```

Key parameters:

| Parameter | Meaning |
|---|---|
| `--config-name=offline` | Offline (batch) recognition pipeline. Use `streaming` for low-latency streaming. |
| `name=` | The Riva model name clients will address. |
| `chunk_size=4.8` | 4.8 s of audio per inference chunk. |
| `left_padding_size` / `right_padding_size` | 1.6 s of context on each side so chunk boundaries don't clip attention context. |
| `ms_per_timestep=80` | FastConformer emits one frame per 80 ms (10 ms hop × 8× subsampling). **Wrong values here silently corrupt timestamps and decoding.** |
| `featurizer.*norm*=False` | Disables utterance-level feature normalization — the NeMo FastConformer checkpoints use per-feature normalization baked into the model. |
| `nn.max_batch_size=32` | Max concurrent utterances through the acoustic model. Match this to your benchmark concurrency. |
| `featurizer.max_batch_size=512` | The featurizer is cheap, so let it batch far more aggressively than the NN. |
| `nn.fp16_needs_obey_precision_pass=True` | Forces TensorRT to respect precision constraints on sensitive layers; prevents FP16 overflow in the Conformer. |
| `decoder=nemo` | Use the NeMo CTC greedy decoder rather than the Kaldi-style Flashlight decoder. |

Note the quoting on `source_path` — the brackets make it a list, and the shell would otherwise try
to glob them.

---

## 6. Deploy the model repository

```bash
riva-deploy -f /data/artifacts/parakeet-ctc-0.6b.rmir:nvidia /data/models
```

This generates the Triton model repository under `/data/models` and builds the TensorRT engine. It
is the slowest step — expect several minutes, and note the engine is specific to this GPU
architecture, so it must be rebuilt if you move to a different GPU generation.

`-f` overwrites an existing repository at that path.

Resulting layout:

```
/data/models/
├── parakeet_ctc_600m_offline            # Riva ensemble
├── parakeet_ctc_600m_offline-ctc-decoder-cpu-streaming-offline
├── parakeet_ctc_600m_offline-feature-extractor-streaming-offline
├── ...
└── riva-trt-parakeet_ctc_600m_offline-am-streaming-offline   # TensorRT engine
```

---

## 7. Start the server

Because we overrode the entrypoint, the server is not running yet. Point Riva at the repository we
just built and start it in the background:

```bash
/opt/riva/bin/riva_server \
  --asr_service=true --nlp_service=false --tts_service=false \
  --model_repository=/data/models &
```

Wait for the log line indicating the server is listening on `0.0.0.0:50051`, then confirm the model
loaded:

```bash
grpc_health_probe -addr=localhost:50051
```

---

## 8. Prepare LibriSpeech test-clean

The benchmark in the next step consumes a NeMo-style JSON manifest. This section builds it.

Run every command from your working directory (the one bind-mounted to `/data`). These helper
scripts are expected to be present there:

- `convert_librispeech_to_nemo.py`
- `filter_manifest.py`
- `sort_manifest.py`

Download and extract the dataset:

```bash
wget https://openslr.trmal.net/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

Convert the LibriSpeech metadata into a NeMo manifest:

```bash
python convert_librispeech_to_nemo.py \
  --librispeech_dir /home/fastconformer/trt_fastconformer/LibriSpeech \
  --output_manifest test_clean_manifest.json \
  --subset test-clean
```

Remove samples longer than 20 seconds:

```bash
python filter_manifest.py \
  --input_manifest test_clean_manifest.json \
  --output_manifest test_clean_manifest_filtered.json \
  --max_duration 20.0
```

Create a duration-ascending version of the filtered manifest. This is the sorted workload used for
more efficient batching:

```bash
python sort_manifest.py \
  --input_manifest test_clean_manifest_filtered.json \
  --output_manifest test_clean_manifest_sorted_filtered.json
```

The two benchmark inputs are:

| Workload | Manifest |
| --- | --- |
| Random | `test_clean_manifest_filtered.json` |
| Sorted | `test_clean_manifest_sorted_filtered.json` |

> **Path note:** `--librispeech_dir` above is an absolute path from the original NeMo-container
> workflow. Point it at wherever you extracted `LibriSpeech/`. The `audio_filepath` entries the
> script writes must also resolve *inside* the NIM container, so prefer paths under `/data`.

---

## 9. Benchmark the NIM deployment

Open a second shell into the **same** container:

```bash
docker exec -it parakeet-1-1b-ctc-en-us /bin/bash
```

Then:

```bash
mkdir -p results

riva_asr_client \
  --automatic_punctuation=false \
  --num_parallel_requests=32 \
  --word_time_offsets=false \
  --print_transcripts=false \
  --num_iterations=1 \
  --audio_file=test_clean_manifest_sorted_filtered.json \
  --output_filename=results/nim_sorted.json
```

| Flag | Purpose |
|---|---|
| `--audio_file` | A NeMo-style JSON manifest (one JSON object per line with `audio_filepath`, `duration`, `text`). Paths inside must resolve within the container. |
| `--num_parallel_requests=32` | Concurrency. Keep this ≤ `nn.max_batch_size` from the build step or you will queue rather than batch. |
| `--automatic_punctuation=false` | Skips the punctuation model so you measure the acoustic pipeline alone. |
| `--word_time_offsets=false` | Skips timestamp computation — measurable overhead at high concurrency. |
| `--print_transcripts=false` | Keeps stdout out of the timing loop. |
| `--num_iterations=1` | Single pass over the manifest. Raise for a warmed-steady-state number. |
| `--output_filename` | Transcripts written here for offline WER scoring. |

The client prints total audio processed, wall time, throughput (RTFX) and latency percentiles. Sort
the manifest by duration (as the filename suggests) to minimize padding waste and get a
best-case throughput figure.

---

## 10. NeMo baseline benchmark (optional)

To compare the deployed NIM/TensorRT pipeline against the unoptimized PyTorch path, run the
FastConformer CTC Large checkpoint directly in NeMo.

> **This runs in a different container.** Exit or leave the NIM container and start a NeMo one —
> the ServiceMaker image does not carry the NeMo training stack:
>
> ```bash
> docker run -it --gpus '"device=0"' --ipc=host -v $PWD:/home nvcr.io/nvidia/nemo:25.11
> pip install pycuda
> ```
>
> `transcribe_speech_file.py` must be present in the working directory.

Download the NeMo model:

```bash
wget https://huggingface.co/nvidia/stt_en_fastconformer_ctc_large/resolve/main/stt_en_fastconformer_ctc_large.nemo
```

Run the baseline on the duration-sorted manifest (check with different batch sizes):

```bash
CUDA_VISIBLE_DEVICES=0 python transcribe_speech_file.py \
  --model ./stt_en_fastconformer_ctc_large.nemo \
  --manifest test_clean_manifest_sorted_filtered.json \
  --batch-size 64 \
  --warmup-steps 5 \
  --run-steps 10 \
  --device cuda
```

Repeat the same command for the random manifest by changing `--manifest` to
`test_clean_manifest_filtered.json`. To get a full comparison, run both manifests with batch sizes
`32`, `64`, and `128`.

> **Caveat on comparing numbers:** this baseline is `stt_en_fastconformer_ctc_large`, while the NIM
> deployment above serves `parakeet-ctc-0.6b`. They are both FastConformer-CTC Large at ~600M
> parameters and are directly comparable in shape, but they are not the same weights — treat the
> RTFX delta as runtime speedup, not an apples-to-apples check of a single checkpoint. Also keep
> `--batch-size` at or below the `nn.max_batch_size=32` used in the RMIR build if you want the two
> sides to see the same batching regime.

---

## 11. Inference on a single file

```bash
python /opt/riva/examples/transcribe_file_offline.py \
  --input-file example.wav \
  --language-code en-US
```

Audio should be mono, 16 kHz, 16-bit PCM WAV. Convert first if needed:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 -c:a pcm_s16le example.wav
```

If the server is on another host, add `--server <host>:50051`.

---

## Troubleshooting

**`Error: model 'parakeet_ctc_600m_offline' is not available`**
The name passed to the client must match `name=` from `riva-build`. Check what actually loaded with
`riva_asr_client --list_models` or inspect the Triton logs.

**`nemo2riva` fails on ONNX export**
Almost always an opset mismatch. FastConformer needs opset 17+; 19 is what's validated here.

**Decryption / key errors during `riva-build` or `riva-deploy`**
The `:nvidia` suffix on every path is the key. Omitting it on any one of the four paths breaks the
chain.

**Out of memory during `riva-deploy`**
TensorRT engine building is memory-hungry. Free the GPU of other work, or lower
`nn.max_batch_size`.

**Garbled or empty transcripts**
Check `ms_per_timestep`. It must be 80 for FastConformer's 8× subsampling — the Conformer default of
40 will produce nonsense.

**Container exits immediately**
Expected if you forget `-it`. The overridden entrypoint is an interactive shell with nothing to do.

**Artifacts disappear after exit**
`--rm` deletes the container, but `/data` is bind-mounted to your host `$PWD`. Anything written
outside `/data` is lost.
