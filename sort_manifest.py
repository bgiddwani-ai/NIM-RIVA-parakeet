import json
import argparse

def sort_manifest_by_duration(input_manifest: str, output_manifest: str):
    """
    Sort a NeMo manifest file by duration (ascending).

    Args:
        input_manifest: Path to input NeMo manifest (.json JSONL file).
        output_manifest: Path to write the sorted manifest.
    """
    entries = []
    with open(input_manifest, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {input_manifest}")

    entries.sort(key=lambda x: x["duration"])

    with open(output_manifest, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Sorted manifest written to: {output_manifest}")
    print(f"   Shortest: {entries[0]['duration']:.4f}s  → {entries[0]['text'][:60]}")
    print(f"   Longest:  {entries[-1]['duration']:.4f}s → {entries[-1]['text'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort NeMo manifest by duration (low to high)")
    parser.add_argument("--input_manifest", type=str, required=True, help="Path to input manifest")
    parser.add_argument("--output_manifest", type=str, required=True, help="Path to output sorted manifest")
    args = parser.parse_args()

    sort_manifest_by_duration(args.input_manifest, args.output_manifest)
