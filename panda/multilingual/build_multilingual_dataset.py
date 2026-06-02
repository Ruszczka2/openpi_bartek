"""Build fast_3_tasks_multilingual_v1 from fast_pick_and_place_multitask_merged_v4.

Tasks (9 entries, 3 per original task × 3 languages):
  0: "place the green cube in the yellow area"           (EN, task1)
  1: "połóż zielony sześcian w żółtym obszarze"         (PL, task1)
  2: "place le cube vert dans la zone jaune"             (FR, task1)
  3: "hold the green cube above the yellow area"         (EN, task2)
  4: "trzymaj zielony sześcian nad żółtym obszarem"      (PL, task2)
  5: "tiens le cube vert au-dessus de la zone jaune"     (FR, task2)
  6: "return the green cube to the start position"       (EN, task3)
  7: "zwróć zielony sześcian na pozycję startową"        (PL, task3)
  8: "ramène le cube vert à la position de départ"       (FR, task3)

Episode distribution (~1/3 per language per task):
  Task1 (eps 0-39, 40 eps):   EN=0-12 (13), PL=13-25 (13), FR=26-39 (14)
  Task2 (eps 40-80, 41 eps):  EN=40-53 (14), PL=54-67 (14), FR=68-80 (13)
  Task3 (eps 81-120, 40 eps): EN=81-93 (13), PL=94-106 (13), FR=107-120 (14)
"""

import json
import shutil
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

SRC = "/home/ruszczka/projekty/vla_ft/external/openpi_bartek/panda/outputs/fast_pick_and_place_multitask_merged_v4"
DST = "/home/ruszczka/projekty/vla_ft/external/openpi_bartek/panda/outputs/fast_3_tasks_multilingual_v1"

TASKS = [
    "place the green cube in the yellow area",
    "połóż zielony sześcian w żółtym obszarze",
    "place le cube vert dans la zone jaune",
    "hold the green cube above the yellow area",
    "trzymaj zielony sześcian nad żółtym obszarem",
    "tiens le cube vert au-dessus de la zone jaune",
    "return the green cube to the start position",
    "zwróć zielony sześcian na pozycję startową",
    "ramène le cube vert à la position de départ",
]

# episode -> new task_index
def build_episode_task_map():
    m = {}
    # task1: EN=0-12, PL=13-25, FR=26-39
    for e in range(0, 13):   m[e] = 0
    for e in range(13, 26):  m[e] = 1
    for e in range(26, 40):  m[e] = 2
    # task2: EN=40-53, PL=54-67, FR=68-80
    for e in range(40, 54):  m[e] = 3
    for e in range(54, 68):  m[e] = 4
    for e in range(68, 81):  m[e] = 5
    # task3: EN=81-93, PL=94-106, FR=107-120
    for e in range(81, 94):  m[e] = 6
    for e in range(94, 107): m[e] = 7
    for e in range(107, 121): m[e] = 8
    return m


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- 1. Clean copy ---
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"Copied {SRC} -> {DST}")

    ep_task = build_episode_task_map()

    # --- 2. tasks.jsonl ---
    tasks_path = os.path.join(DST, "meta", "tasks.jsonl")
    with open(tasks_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(TASKS):
            f.write(json.dumps({"task_index": i, "task": task}, ensure_ascii=False) + "\n")
    print("Written tasks.jsonl (9 tasks)")

    # --- 3. parquet: update task_index column (one file per episode) ---
    chunk_dir = os.path.join(DST, "data", "chunk-000")
    total_rows = 0
    task_counts = {i: 0 for i in range(len(TASKS))}
    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(chunk_dir, fname)
        df = pq.read_table(fpath).to_pandas()
        ep_idx = int(df["episode_index"].iloc[0])
        new_task_idx = ep_task[ep_idx]
        df["task_index"] = new_task_idx
        # Preserve original schema column order
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), fpath)
        task_counts[new_task_idx] += 1
        total_rows += len(df)
    print(f"Updated {total_rows} rows across 121 parquet files.")
    print("Episodes per task_index:", {k: v for k, v in task_counts.items() if v > 0})

    # --- 4. episodes.jsonl: update tasks field ---
    ep_jsonl_path = os.path.join(DST, "meta", "episodes.jsonl")
    lines = []
    with open(ep_jsonl_path, encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            ep["tasks"] = [TASKS[ep_task[ep["episode_index"]]]]
            lines.append(json.dumps(ep, ensure_ascii=False))
    with open(ep_jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Updated episodes.jsonl")

    # --- 5. info.json: update total_tasks ---
    info_path = os.path.join(DST, "meta", "info.json")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    info["total_tasks"] = len(TASKS)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    print(f"Updated info.json: total_tasks={info['total_tasks']}")

    print("\nDone! Dataset at:", os.path.abspath(DST))


if __name__ == "__main__":
    main()
