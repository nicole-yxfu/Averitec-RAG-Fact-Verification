import os
import urllib.request


def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    files_to_download = {
        "train.json": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json",
        "dev.json": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"
    }

    for filename, url in files_to_download.items():
        save_path = os.path.join(data_dir, filename)
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f" {filename} is saved in {save_path}")


if __name__ == "__main__":
    main()
