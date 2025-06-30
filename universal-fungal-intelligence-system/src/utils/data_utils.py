def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json_file(file_path: str, data: dict) -> None:
    """Save a dictionary as a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def flatten_dict(nested_dict: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary."""
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries, with dict2 values overwriting dict1 values."""
    merged = dict1.copy()
    merged.update(dict2)
    return merged

def extract_keys(data: dict, keys: list) -> dict:
    """Extract specific keys from a dictionary."""
    return {key: data[key] for key in keys if key in data}