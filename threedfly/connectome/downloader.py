"""Download MaleCNS v1.0 connectome data from Google Cloud Storage."""

import hashlib
import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"

FILES = {
    "annotations": {
        "filename": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "size_mb": 14,
    },
    "weights": {
        "filename": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "size_mb": 1100,
    },
    "neurotransmitters": {
        "filename": "body-neurotransmitters-male-cns-v1.0-minconf-0.5.feather",
        "size_mb": 5,
    },
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Download a file with progress bar and resume capability.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return
    
    # Download with resume support
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    headers = {}
    initial_pos = 0
    if temp_path.exists():
        initial_pos = temp_path.stat().st_size
        headers["Range"] = f"bytes={initial_pos}-"
        print(f"Resuming download from byte {initial_pos}")
    
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    
    # Handle resume
    if response.status_code == 416:  # Range not satisfiable
        print("File already fully downloaded")
        temp_path.rename(dest_path)
        return
    
    total_size = initial_pos
    if response.status_code == 206:  # Partial content
        content_range = response.headers.get("Content-Range", "")
        if content_range:
            total_size = int(content_range.split("/")[-1])
    else:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
    
    mode = "ab" if initial_pos > 0 else "wb"
    
    with open(temp_path, mode) as f, tqdm(
        desc=desc,
        initial=initial_pos,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Move to final location
    temp_path.rename(dest_path)
    print(f"✓ Downloaded: {dest_path}")


def download_connectome_data(
    output_dir: Optional[str] = None,
    files_to_download: Optional[list[str]] = None
) -> Path:
    """
    Download MaleCNS v1.0 connectome data files.
    
    Args:
        output_dir: Directory to save files (default: data/connectome-raw)
        files_to_download: List of file keys to download (default: all)
    
    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "connectome-raw"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if files_to_download is None:
        files_to_download = list(FILES.keys())
    
    print("=" * 70)
    print("MaleCNS v1.0 Connectome Data Download")
    print("=" * 70)
    print(f"Source: {BASE_URL}")
    print(f"Destination: {output_dir.absolute()}")
    print(f"License: CC-BY (cite male-cns.janelia.org)")
    print("=" * 70)
    
    for file_key in files_to_download:
        if file_key not in FILES:
            print(f"Warning: Unknown file key '{file_key}', skipping")
            continue
        
        file_info = FILES[file_key]
        filename = file_info["filename"]
        url = BASE_URL + filename
        dest_path = output_dir / filename
        
        print(f"\n[{file_key}] {filename} (~{file_info['size_mb']}MB)")
        download_file(url, dest_path, desc=f"Downloading {file_key}")
    
    print("\n" + "=" * 70)
    print("✓ Download complete!")
    print("=" * 70)
    
    return output_dir
