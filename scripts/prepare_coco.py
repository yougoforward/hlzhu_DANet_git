"""Prepare MS COCO datasets"""
import os
import shutil
import argparse
import zipfile
from encoding.utils import download, mkdir

_TARGET_DIR = os.path.expanduser('~/.encoding/data')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize MS COCO dataset.',
        epilog='Example: python mscoco.py --download-dir ~/mscoco',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_coco(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://images.cocodataset.org/zips/train2017.zip',
         '10ad623668ab00c62c096f0ed636d6aff41faca5'),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         '8551ee4bb5860311e79dace7e79cb91e432e78b3'),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '4950dc9d00dbe1c933ee0170f5797584351d2a41'),
        ('http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',
         'e7aa0f7515c07e23873a9f71d9095b06bcea3e12'),
        ('http://images.cocodataset.org/zips/test2017.zip',
         '99813c02442f3c112d491ea6f30cecf421d0e6b3'),
    ]
    mkdir(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(path=path)


def install_coco_api():
    repo_url = "https://github.com/cocodataset/cocoapi"
    os.system("git clone " + repo_url)
    os.system("cd cocoapi/PythonAPI/ && python setup.py install")
    shutil.rmtree('cocoapi')
    try:
        import pycocotools
    except Exception:
        print("Installing COCO API failed, please install it manually %s"%(repo_url))


if __name__ == '__main__':
    args = parse_args()
    mkdir(os.path.expanduser('~/.encoding/data'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_coco(_TARGET_DIR, overwrite=False)
    install_coco_api()
