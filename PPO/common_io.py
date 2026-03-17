import csv
import json
from pathlib import Path
from typing import Dict, List, Any

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_rows_csv(path: str, rows: List[Dict[str, Any]]):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def append_row_csv(path, row: Dict[str, Any]):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
