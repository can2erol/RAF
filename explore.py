from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd

# Download and get the local path. No video decoding for now
path = Path(snapshot_download(repo_id="lerobot/pusht", repo_type="dataset"))
print("Dataset cached at:", path)

# Trajectory data (state and action) lives in parquet files, check
parquet_files = sorted(path.glob("data/**/*.parquet"))
print(f"Found {len(parquet_files)} parquet file(s)")

# Read the first parquet file straight into a table
df = pd.read_parquet(parquet_files[0])

print("\nColumns:", df.columns.tolist())
print("Rows in this file:", len(df))
print("\n----- First row -----")
print(df.iloc[0])

# How mant distinct episodes (trajectories) are in this file?
if "episode_index" in df.columns:
    n_eps = df["episode_index"].nunique()
    print(f"\nDistinct episodes in this file: {n_eps}")

# Pull out a trajectory
ep0 = df[df["episode_index"] == 0].sort_values("frame_index") # data mask, pandas technique

print(f"\nEpisode 0 has {len(ep0)} frames")
print("First 3 frames of epsiode 0:")
print(ep0[["frame_index", "observation.state", "action", "next.reward"]].head(3))

print("Last 3 frames of episode 0:")
print(ep0[["frame_index", "observation.state", "action", "next.reward"]].tail(3))