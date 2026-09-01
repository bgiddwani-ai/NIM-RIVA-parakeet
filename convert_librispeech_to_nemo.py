import os
import json
import glob
import soundfile as sf
from pathlib import Path

def get_duration(audio_path):
    """Get duration of audio file in seconds."""
    with sf.SoundFile(audio_path) as f:
        return len(f) / f.samplerate

def convert_librispeech_to_nemo_manifest(
    librispeech_dir: str,
    output_manifest: str,
    subset: str = "dev-clean"
):
    """
    Convert LibriSpeech dataset to NeMo manifest format.

    Args:
        librispeech_dir: Root directory of LibriSpeech dataset.
                         Expects structure: <root>/<subset>/<speaker>/<chapter>/*.flac + *.trans.txt
        output_manifest: Path to output .json manifest file.
        subset: Dataset subset name (e.g. 'dev-clean', 'train-clean-100').
    """
    subset_dir = os.path.join(librispeech_dir, subset)
    assert os.path.isdir(subset_dir), f"Subset directory not found: {subset_dir}"

    # Find all transcript files
    trans_files = glob.glob(os.path.join(subset_dir, "**", "*.trans.txt"), recursive=True)
    assert trans_files, f"No transcript files found under {subset_dir}"

    entries = []

    for trans_file in sorted(trans_files):
        chapter_dir = os.path.dirname(trans_file)

        with open(trans_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: <utterance_id> <transcript>
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    print(f"[WARN] Skipping malformed line: {line}")
                    continue

                utt_id, transcript = parts
                audio_path = os.path.join(chapter_dir, utt_id + ".flac")

                if not os.path.isfile(audio_path):
                    print(f"[WARN] Audio file not found: {audio_path}")
                    continue

                duration = get_duration(audio_path)

                entry = {
                    "audio_filepath": os.path.abspath(audio_path),
                    "duration": round(duration, 4),
                    "text": transcript.lower()  # NeMo convention: lowercase text
                }
                entries.append(entry)

    # Write manifest — one JSON object per line (JSONL format)
    os.makedirs(os.path.dirname(os.path.abspath(output_manifest)), exist_ok=True)
    with open(output_manifest, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Manifest written to: {output_manifest}")
    print(f"   Total utterances: {len(entries)}")
    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"   Total duration:   {total_hours:.2f} hours")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LibriSpeech to NeMo manifest")
    parser.add_argument(
        "--librispeech_dir",
        type=str,
        required=True,
        help="Root LibriSpeech directory (containing dev-clean/, train-clean-100/, etc.)"
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="dev_clean_manifest.json",
        help="Path to output NeMo manifest file"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="dev-clean",
        help="LibriSpeech subset (e.g. dev-clean, train-clean-100, test-clean)"
    )
    args = parser.parse_args()

    convert_librispeech_to_nemo_manifest(
        librispeech_dir=args.librispeech_dir,
        output_manifest=args.output_manifest,
        subset=args.subset
    )
