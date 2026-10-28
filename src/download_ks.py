import os
from huggingface_hub import snapshot_download


def download_knowledge_store():
    data_dir = "/Volumes/Seagate Backup Plus Drive/AVeriTeC_knowledge_store"

    os.makedirs(data_dir, exist_ok=True)

    print(f"Starting download to: {data_dir}")
    print("Note: Files are large (>100GB). Keep the drive connected. Resumable if interrupted.")

    snapshot_download(
        repo_id="chenxwh/AVeriTeC",
        allow_patterns=["data_store/knowledge_store/*"],
        local_dir=data_dir,
        token="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"
    )

    print(f"Download complete! Files saved to: {data_dir}")


if __name__ == "__main__":
    download_knowledge_store()
