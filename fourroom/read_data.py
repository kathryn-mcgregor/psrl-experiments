"""
Utility to read collected participant data from msgpack files.

Usage:
    python read_data.py data/user_data_<seed>.msgpack
"""
import sys
import json
from nicewebrl.utils import read_all_records_sync


def load_participant(filepath: str) -> list[dict]:
    return read_all_records_sync(filepath)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_data.py <msgpack_file>")
        sys.exit(1)

    records = load_participant(sys.argv[1])
    print(f"Total steps recorded: {len(records)}")
    for r in records:
        mark = ""
        if r.get("reached_true"):
            mark = " *** TRUE GOAL REACHED ***"
        elif r.get("visited_goal") is not None:
            mark = f" (visited candidate goal #{r['visited_goal'] + 1})"
        print(
            f"  step={r['step']:4d}  action={r['action']:12s}  "
            f"{r['prev_pos']} -> {r['new_pos']}{mark}"
        )
