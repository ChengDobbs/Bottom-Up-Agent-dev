import os
import shutil
from huggingface_hub import snapshot_download

def download_and_copy_subdirs(
    repo_id="microsoft/OmniParser-v2.0",
    subdirs=("icon_caption", "icon_detect"),
    exclude_files=("",),  # relative paths to exclude, e.g., "icon_caption/LICENSE"
    target_root="weights",
    cache_dir=None,
):
    """
    Download specified subdirectories from the HuggingFace repo, excluding some files,
    and copy them into target_root/<subdir>.
    """
    allow_patterns = [f"{sd}/**" for sd in subdirs]
    # Build ignore_patterns list, filtering out empty strings
    ignore_patterns = [p for p in exclude_files if p]

    print(f"Downloading {subdirs} from {repo_id} with filters...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    print(f"Download completed. Snapshot root: {snapshot_path}")

    os.makedirs(target_root, exist_ok=True)

    for sd in subdirs:
        src = os.path.join(snapshot_path, sd)
        if not os.path.isdir(src):
            print(f"Warning: expected directory not found (skipping): {src}")
            continue
        
        # Rename icon_caption to icon_caption_florence
        if sd == "icon_caption":
            dest_folder_name = "icon_caption_florence"
        else:
            dest_folder_name = sd
            
        dest = os.path.join(target_root, dest_folder_name)
        if os.path.exists(dest):
            print(f"Removing existing target folder: {dest}")
            shutil.rmtree(dest)
        print(f"Copying {src} -> {dest}")
        shutil.copytree(src, dest)

    print(f"Finished. Available under '{target_root}/'.")

if __name__ == "__main__":
    download_and_copy_subdirs(
        exclude_files=("icon_caption/LICENSE", "icon_detect/LICENSE"),
    )
