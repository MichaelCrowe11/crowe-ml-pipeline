import json
import os
import argparse

try:
    from crowechem.core.data_sources import get_dataset
except ImportError:  # Fallback if CroweChem isn't installed
    def get_dataset():
        return [
            {"id": 1, "molecule": "example", "property": 0.0}
        ]

try:
    from crowechem.core.validation import validate_dataset
except ImportError:
    def validate_dataset(data):
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list")

OUTPUT_DIR = "data"
OUTPUT_FILE = "crowechem_dataset.jsonl"


def main(output_dir: str = OUTPUT_DIR, output_file: str = OUTPUT_FILE):
    os.makedirs(output_dir, exist_ok=True)
    dataset = get_dataset()
    validate_dataset(dataset)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CroweChem dataset")
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory to store dataset (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-file",
        default=OUTPUT_FILE,
        help=f"Output filename (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, output_file=args.output_file)
