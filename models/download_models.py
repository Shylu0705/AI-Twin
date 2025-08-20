# download_models.py
import os
import requests
from tqdm import tqdm

# Define models and their download links
MODELS = {
    "llama3.1-8b-q4_K_S.gguf": "https://huggingface.co/cha9ro/Meta-Llama-3.1-8B-Instruct-Q4_K_S-GGUF/resolve/main/meta-llama-3.1-8b-instruct-q4_k_s.gguf",
    "shisa-v2-mistral-nemo-12b.Q4_K_M.gguf": "https://huggingface.co/mradermacher/shisa-v2-mistral-nemo-12b-GGUF/resolve/main/shisa-v2-mistral-nemo-12b.Q4_K_M.gguf",
    "shisa-v2-mistral-nemo-12b.Q5_K_M.gguf": "https://huggingface.co/mradermacher/shisa-v2-mistral-nemo-12b-GGUF/resolve/main/shisa-v2-mistral-nemo-12b.Q5_K_M.gguf",
    "shisa-v2-mistral-nemo-12b.Q8_0.gguf": "https://huggingface.co/mradermacher/shisa-v2-mistral-nemo-12b-GGUF/resolve/main/shisa-v2-mistral-nemo-12b.Q8_0.gguf",
}

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = file.write(data)
            bar.update(size)

def main():
    os.makedirs("models", exist_ok=True)

    for name, url in MODELS.items():
        dest = os.path.join("models", name)
        if not os.path.exists(dest):
            print(f"Downloading {name}...")
            download_file(url, dest)
        else:
            print(f"{name} already exists, skipping.")

if __name__ == "__main__":
    main()
