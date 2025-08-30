import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm


def percentile_clip(input_tensor, p_min=0.5, p_max=99.5, strictlyPositive=True):
    v_min, v_max = np.percentile(input_tensor, [p_min, p_max])
    if v_min < 0 and strictlyPositive:
        v_min = 0
    output_tensor = np.clip(input_tensor, v_min, v_max)
    return (output_tensor - v_min) / (v_max - v_min + 1e-8)


def nii2png(img_id, img_path, save_paths):
    modalities = {
        "t1n": "-t1n.nii.gz",
        "t2w": "-t2w.nii.gz",
        "t2f": "-t2f.nii.gz",
        "t1c": "-t1c.nii.gz",
        "seg": "-seg.nii.gz"
    }

    images = {}
    for key, suffix in modalities.items():
        file_path = os.path.join(img_path, img_id, img_id + suffix)
        if not os.path.exists(file_path):
            print(f"Missing Image: {file_path}")
            return
        images[key] = sitk.GetArrayFromImage(sitk.ReadImage(file_path))

    for key in ["t1n", "t2w", "t2f", "t1c"]:
        images[key] = percentile_clip(images[key]) * 255
        images[key] = images[key].astype(np.uint8)

    seg = images["seg"]
    seg[seg != 2] = 0
    seg[seg == 2] = 1
    seg *= 255

    for i in range(seg.shape[0]):
        if np.sum(seg[i]) == 0:
            continue

        for key in ["t1n", "t2w", "t2f", "t1c"]:
            img_pil = Image.fromarray(images[key][i])
            img_pil.save(os.path.join(save_paths[key], f"{img_id}_{i:05d}.png"))

        seg_pil = Image.fromarray(seg[i].astype(np.uint8))
        seg_pil.save(os.path.join(save_paths["seg"], f"{img_id}_{i:05d}.png"))


if __name__ == "__main__":
    img_path = r'./dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    # img_path = r'G:\BraTS2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

    save_paths = {
        "t1n": r'./dataset/BraTS2023-TrainingData_png_t1n',
        "t2w": r'./dataset/BraTS2023-TrainingData_png_t2w',
        "t2f": r'./dataset/BraTS2023-TrainingData_png_t2f',
        "t1c": r'./dataset/BraTS2023-TrainingData_png_t1c',
        "seg": r'./dataset/BraTS2023-TrainingData_png_seg',
    }

    for path in save_paths.values():
        os.makedirs(path, exist_ok=True)

    img_list = sorted(os.listdir(img_path))

    for img_id in tqdm(img_list, desc="Processing cases"):
        nii2png(img_id, img_path, save_paths)
