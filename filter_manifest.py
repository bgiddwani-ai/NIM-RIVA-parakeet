import json
import argparse

def filter_manifest_by_duration(input_manifest: str, output_manifest: str, max_duration: float = 20.0):
    """
    Remove entries exceeding max_duration from a NeMo manifest file.

    Args:
        input_manifest: Path to input NeMo manifest (JSONL).
        output_manifest: Path to write the filtered manifest.
        max_duration: Maximum allowed duration in seconds (default: 20.0).
    """
    entries = []
    with open(input_manifest, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {input_manifest}")

    filtered = [e for e in entries if e["duration"] <= max_duration]
    removed = len(entries) - len(filtered)

    with open(output_manifest, "w") as f:
        for entry in filtered:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Filtered manifest written to: {output_manifest}")
    print(f"   Kept:    {len(filtered)} entries")
    print(f"   Removed: {removed} entries (duration > {max_duration}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter NeMo manifest by max duration")
    parser.add_argument("--input_manifest", type=str, required=True, help="Path to input manifest")
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to output filtered manifest")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Max duration in seconds (default: 20.0)")
    args = parser.parse_args()

    filter_manifest_by_duration(args.input_manifest, args.output_manifest, args.max_duration)
