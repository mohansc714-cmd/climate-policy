"""
Deploy Climate Policy RAG Assistant to HuggingFace Spaces.
Space: https://huggingface.co/spaces/Keerthi-sc/climate_RAN
"""
from huggingface_hub import HfApi
import os

REPO_ID   = "Keerthi-sc/climate_RAN"
REPO_TYPE = "space"

# Files to upload
FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "climate_change_dataset.csv", # Updated data file
]

def deploy():
    token = os.getenv("HF_TOKEN")
    if not token:
        # Fallback to local auth if environment variable is not set
        api = HfApi()
        print("Using local HF credentials...")
    else:
        api = HfApi(token=token)
        print("Using HF_TOKEN from environment...")

    # 1. Create the Space if it doesn't exist
    print(f"Ensuring Space exists: {REPO_ID} ...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk="gradio",
        exist_ok=True,
    )
    print("Space ready.")

    # 2. Upload each file
    for fname in FILES:
        if not os.path.exists(fname):
            print(f"  SKIP (not found): {fname}")
            continue
        print(f"  Uploading: {fname} ...")
        api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  Done: {fname}")

    print("\nDeployment complete!")
    print(f"View your Space: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    deploy()
