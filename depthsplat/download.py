""" This script is used to download the DL3DV benchmark from the huggingface repo.

    The benchmark is composed of 140 different scenes covering different scene complexities (reflection, transparency, indoor/outdoor, etc.) 

    The whole benchmark is very large: 2.1 TB. So we provide this script to download the subset of the dataset based on common needs. 


        - [x] Full benchmark downloading
            Full download can directly be done by git clone (w. lfs installed).

        - [x] scene downloading based on scene hash code  

        Option: 
        - [x] images_4 (960 x 540 resolution) level dataset (approx 50G)

"""

import os 
from os.path import join
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi 
import argparse
import traceback
import pickle
import shutil
from pathlib import Path
import json
from PIL import Image

HF_ENDPOINT = os.getenv('HF_ENDPOINT')
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')


def get_api() -> HfApi:
    return HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)


repo_root = 'DL3DV/DL3DV-10K-Benchmark'


def hf_download_path(repo_path: str, odir: str, max_try: int = 5):
    """ hf api is not reliable, retry when failed with max tries

    :param repo_path: The path of the repo to download
    :param odir: output path 
    """	
    rel_path = os.path.relpath(repo_path, repo_root)

    counter = 0
    while True:
        if counter >= max_try:
            print("ERROR: Download {} failed.".format(repo_path))
            return False

        try:
            api = get_api()
            api.hf_hub_download(
                repo_id=repo_root,
                filename=rel_path,
                repo_type='dataset',
                local_dir=odir,
                cache_dir=join(odir, '.cache'),
                token=HF_TOKEN,
            )
            return True

        except BaseException as e:
            traceback.print_exc()
            counter += 1
            print(f'Retry {counter}')
    

def clean_huggingface_cache(cache_dir: str):
    """ Huggingface cache may take too much space, we clean the cache to save space if necessary

    :param cache_dir: the current cache directory 
    """    
    # Current huggingface hub does not provide good practice to clean the space.  
    # We mannually clean the cache directory if necessary. 
    target = join(cache_dir, 'datasets--DL3DV--DL3DV-10K-Benchmark')
    if os.path.exists(target):
        shutil.rmtree(target)


def normalize_scene_layout(scene_root: Path) -> None:
    """Flatten benchmark nerfstudio scene layout to the minimal format expected by
    downstream conversion scripts:

        <scene_root>/transforms.json
        <scene_root>/images_4/
    """
    nerfstudio_dir = scene_root / "nerfstudio"
    if not nerfstudio_dir.is_dir():
        return

    transforms_src = nerfstudio_dir / "transforms.json"
    images_src = nerfstudio_dir / "images_4"
    transforms_dst = scene_root / "transforms.json"
    images_dst = scene_root / "images_4"

    if transforms_src.is_file() and not transforms_dst.exists():
        shutil.move(str(transforms_src), str(transforms_dst))
    if images_src.is_dir():
        if images_dst.exists():
            shutil.rmtree(images_dst)
        shutil.move(str(images_src), str(images_dst))

    if nerfstudio_dir.exists() and not any(nerfstudio_dir.iterdir()):
        nerfstudio_dir.rmdir()


def rescale_transforms_for_images4(scene_root: Path) -> None:
    """Scale camera intrinsics and stored resolution in transforms.json to match
    the downloaded images_4 resolution.
    """
    transforms_path = scene_root / "transforms.json"
    images_dir = scene_root / "images_4"
    if not transforms_path.is_file() or not images_dir.is_dir():
        return

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        return

    with Image.open(image_files[0]) as image:
        image_w, image_h = image.size

    with transforms_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    src_w = float(data.get("w", image_w))
    src_h = float(data.get("h", image_h))
    if src_w == 0 or src_h == 0:
        return

    scale_x = image_w / src_w
    scale_y = image_h / src_h

    data["w"] = image_w
    data["h"] = image_h
    if "fl_x" in data:
        data["fl_x"] = float(data["fl_x"]) * scale_x
    if "fl_y" in data:
        data["fl_y"] = float(data["fl_y"]) * scale_y
    if "cx" in data:
        data["cx"] = float(data["cx"]) * scale_x
    if "cy" in data:
        data["cy"] = float(data["cy"]) * scale_y

    with transforms_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def prune_scene_to_minimal_layout(scene_root: Path) -> None:
    """Keep only transforms.json and images_4 under the scene root."""
    keep_names = {"transforms.json", "images_4"}
    for child in scene_root.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def get_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def download_by_hash(filepath_dict: dict, odir: str, hash: str, only_level4: bool):
    """ Given a hash, download the relevant data from the huggingface repo 

    :param filepath_dict: the cache dict that stores all the file relative paths 
    :param odir: the download directory 
    :param hash: the hash code for the scene 
    :param only_level4: the images_4 resolution level, if true, only the images_4 resolution level will be downloaded 
    """	
    all_files = filepath_dict[hash]
    download_files = [join(repo_root, f) for f in all_files] 

    if only_level4: # only download images_4 level data
        download_files = []
        for f in all_files:
            normalized = f.replace("\\", "/")
            if normalized == f"{hash}/nerfstudio/transforms.json":
                download_files.append(join(repo_root, f))
                continue
            if normalized.startswith(f"{hash}/nerfstudio/images_4/"):
                download_files.append(join(repo_root, f))

    for f in download_files:
        if hf_download_path(f, odir) == False:
            return False

    if only_level4:
        scene_root = Path(odir) / hash
        normalize_scene_layout(scene_root)
        rescale_transforms_for_images4(scene_root)
        prune_scene_to_minimal_layout(scene_root)
        size_mb = get_size_bytes(scene_root) / (1024 * 1024)
        print(f"[INFO] Final scene size for {hash}: {size_mb:.2f} MB")

    return True
    

def download_benchmark(args):
    """ Download the benchmark based on the user inputs.

        1. download the benchmark-meta.csv
        2. based on the args, download the specific subset 
            a. full benchmark 
            b. full benchmark in images_4 resolution level 
            c. full benchmark only with nerfstudio colmaps (w.o. gaussian splatting colmaps) 
            d. specific scene based on the index in [0, 140)

    :param args: argparse args. Used to decide the subset.
    :return: download success or not
    """	
    output_dir = args.odir
    subset_opt = args.subset
    level4_opt = args.only_level4
    hash_name  = args.hash
    is_clean_cache = args.clean_cache

    # import pdb; pdb.set_trace()
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1: download the benchmark-meta.csv and .cache/filelist.bin
    meta_repo_path = join(repo_root, 'benchmark-meta.csv')
    cache_file_path = join(repo_root, '.cache/filelist.bin')
    if hf_download_path(meta_repo_path, output_dir) == False:
        print('ERROR: Download benchmark-meta.csv failed.')
        return False

    if hf_download_path(cache_file_path, output_dir) == False:
        print('ERROR: Download .cache/filelist.bin failed.')
        return False


    # STEP 2: download the specific subset
    df = pd.read_csv(join(output_dir, 'benchmark-meta.csv'))
    filepath_dict = pickle.load(open(join(output_dir, '.cache/filelist.bin'), 'rb'))
    hashlist = df['hash'].tolist()
    download_list = hashlist

    # sanity check 
    if subset_opt == 'hash':  
        if hash_name not in hashlist: 
            print(f'ERROR: hash {hash_name} not in the benchmark-meta.csv')
            return False

        # if subset is hash, only download the specific hash
        download_list = [hash_name]

    
    # download the dataset 
    for cur_hash in tqdm(download_list):
        if download_by_hash(filepath_dict, output_dir, cur_hash, level4_opt) == False:
            return False

        if is_clean_cache:
            clean_huggingface_cache(join(output_dir, '.cache'))

    return True 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, help='output directory', default='DL3DV-10K-Benchmark')
    parser.add_argument('--subset', choices=['full', 'hash'], help='The subset of the benchmark to download', required=True)
    parser.add_argument('--only_level4', action='store_true', help='If set, only the images_4 resolution level will be downloaded to save space')
    parser.add_argument('--clean_cache', action='store_true', help='If set, will clean the huggingface cache to save space')
    parser.add_argument('--hash', type=str, help='If set subset=hash, this is the hash code of the scene to download', default='')
    parser.add_argument('--hf_endpoint', type=str, default=HF_ENDPOINT, help='Hugging Face endpoint, e.g. https://huggingface.co')
    parser.add_argument('--hf_token', type=str, default=HF_TOKEN, help='Hugging Face access token for gated repos')
    params = parser.parse_args()

    HF_ENDPOINT = params.hf_endpoint
    HF_TOKEN = params.hf_token


    if download_benchmark(params):
        print('Download Done. Refer to', params.odir)
    else:
        print(f'Download to {params.odir} Failed. See error messsage.')
