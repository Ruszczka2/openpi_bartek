from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os

os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_home"))

local_folder_path = "/home/ruszczka/projekty/vla_ft/external/openpi_bartek/panda/outputs/fast_3_tasks_multilingual_v1"
target_repo_id = "Ruszczka/fast_3_tasks_multilingual_v1"

print(f"Loading local dataset from {local_folder_path}...")
dataset = LeRobotDataset(
    target_repo_id,
    root=local_folder_path,
    download_videos=False,
)
print(f"Loaded: {dataset.num_episodes} episodes, {dataset.meta.total_tasks} tasks")

dataset.repo_id = target_repo_id

print(f"Pushing to HF as '{dataset.repo_id}'...")
dataset.push_to_hub(
    private=False,
    tags=["lerobot", "panda", "multilingual"],
)
print("Upload complete!")
