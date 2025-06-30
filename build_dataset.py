import json
import os

try:
    from crowechem.core.data_sources import get_dataset
except ImportError:  # Fallback if CroweChem isn't installed

    def get_dataset():
        return [{"id": 1, "molecule": "example", "property": 0.0}]


try:
    from crowechem.core.validation import validate_dataset
except ImportError:

    def validate_dataset(data):
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list")


OUTPUT_DIR = "data"
OUTPUT_FILE = "crowechem_dataset.jsonl"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = get_dataset()
    validate_dataset(dataset)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    print(f"Dataset written to {output_path}")


if __name__ == "__main__":
    main()
