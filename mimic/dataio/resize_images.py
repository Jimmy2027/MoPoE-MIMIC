import os
from glob import glob

from PIL import Image
from tqdm import tqdm

SIZE_NEW = 128, 128


def fast_scandir(dirname):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


def center_crop_and_resize(img):
    width, height = img.size  # Get dimensions
    new_width = min(width, height)
    new_height = min(width, height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    # Crop the center of the image
    img_crop = img.crop((left, top, right, bottom))
    img_resized = img_crop.resize(SIZE_NEW, Image.ANTIALIAS)
    return img_resized


def walk_dirs(dir_src_base, dir_dst_base, total_folders):
    for dirName, subdirList, fileList in tqdm(os.walk(dir_src_base), total=total_folders):
        # print(dirName)
        f_imgs = glob(os.path.join(dirName, '*.jpg'))
        if len(f_imgs) > 0:
            d_new = os.path.join(dir_dst_base, '/'.join(dirName.split('/')[-3:]))
            if not os.path.exists(d_new):
                os.makedirs(d_new)
            for f in f_imgs:
                n_img = f.split('/')[-1]
                f_new = os.path.join(d_new, n_img)
                if not os.path.exists(f_new):
                    print(f_new)
                    try:
                        img_src = Image.open(f)
                        img_new = center_crop_and_resize(img_src)
                        img_new.save(f_new, "JPEG")
                        img_src.close()
                        img_new.close()
                    except OSError:
                        print('file could not be opened...')


if __name__ == '__main__':
    dir_base_orig = '/cluster/work/vogtlab/Projects/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
    dir_base_resize = f'/cluster/work/vogtlab/Projects/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0/files_small_{SIZE_NEW[0]}'
    print('computing total folders to iterate over')
    total_folders = len(list(os.walk(dir_base_orig)))
    print('starting to resize images')
    walk_dirs(dir_base_orig, dir_base_resize, total_folders)
