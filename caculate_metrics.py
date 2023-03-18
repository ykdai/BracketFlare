import os
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips
import math


def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor=ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor, img2_tensor)
    return output_lpips.detach().numpy()[0,0,0,0]


def compare_masked_psnr(img1, img2, mask):
    unique, counts = np.unique(mask, return_counts=True)
    mask_size = dict(zip(unique, counts))[1]
    original_size = (img1.shape[0] * img1.shape[1]) * 3
    f = math.log((mask_size),10) / math.log((original_size),10)
    img1 = img1 * mask
    img2 = img2 * mask
    return f  * compare_psnr(img1, img2, data_range=255)


def caculate_metrics(gt_folder, mask_folder, input_folder):
    loss_fn_alex = lpips.LPIPS(net='alex')

    gt_folder = os.path.join(gt_folder, '*')
    mask_folder = os.path.join(mask_folder, '*')
    input_folder = os.path.join(input_folder, '*')
    gt_list = sorted(glob(gt_folder))
    mask_list = sorted(glob(mask_folder))
    input_list = sorted(glob(input_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val, masked_psnr = 0, 0, 0, 0
    for i in range(n):
        img_gt = io.imread(gt_list[i])
        img_mask = io.imread(mask_list[i]) / 255
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)
        masked_psnr += compare_masked_psnr(img_gt, img_input, img_mask)

    ssim /= n
    psnr /= n
    lpips_val /= n
    masked_psnr /= n

    print(f"ssim: {ssim}, psnr: {psnr}, lpips: {lpips_val}, masked_psnr: {masked_psnr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt', type=str, help='GT image folder.')
    parser.add_argument('--mask', type=str, help='Mask image folder.')
    parser.add_argument('--input', type=str, help='Input image folder.')

    args = parser.parse_args()

    caculate_metrics(args.gt, args.mask, args.input)
    