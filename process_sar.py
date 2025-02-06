import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# 将WHU中高清长方形图片处理为编号对应的正方形高清图和低清图

def add_speckle_noise(image, scale=0.2):
    """基于Rayleigh分布的散斑噪声"""
    noise = np.random.rayleigh(scale, size=image.shape)
    noisy_img = image * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def add_motion_blur(image, kernel_size=10, angle=30):
    """SAR运动模糊"""
    kernel = np.zeros((kernel_size, kernel_size))
    x_center, y_center = kernel_size // 2, kernel_size // 2

    for i in range(kernel_size):
        x = int(x_center + (i - x_center) * np.cos(np.deg2rad(angle)))
        y = int(y_center + (i - x_center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1

    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)

def downsample_image(image, sigma=1.5, scale_factor=4):
    """对单张图片进行下采样"""
    # 低通滤波（去掉部分高频信息）
    blurred_image = gaussian_filter(image, sigma=sigma)
    
    # 下采样（降低分辨率）
    low_res_image = cv2.resize(blurred_image, (image.shape[1] // scale_factor, image.shape[0] // scale_factor), interpolation=cv2.INTER_NEAREST)
    
    return low_res_image


def process_single(input_path, out_ds_folder,out_hr_folder, tag,counter):
    size = 2560
    with rasterio.open(input_path) as src:
        if tag == 'tl':
            info = (0, 0, size, size)
        elif tag == 'tr':
            info = (src.width - size, 0, size, size)
        elif tag == 'bl':
            info = (0, src.height - size, size, size)
        elif tag == 'br':
            info = (src.width - size, src.height - size, size, size)
        else:
            raise ValueError("Invalid tag. Use 'tl', 'tr', 'bl', or 'br'.")

        window = Window(*info)
        cropped_image = src.read(window=window)
        meta = src.meta.copy()
        meta.update({
            "width": size,
            "height": size,
            "transform": src.window_transform(window)
        })
        
        # 将裁剪后的图像转换为 PNG 格式并保存
        out_hr_path = os.path.join(out_hr_folder, f"{counter:04d}.png")
        cropped_image = np.moveaxis(cropped_image, 0, -1)  # 将波段轴移到最后
        if cropped_image.shape[2] == 1:  # 如果是单波段图像，转换为三通道
            cropped_image = np.repeat(cropped_image, 3, axis=2)
        cropped_image = cropped_image.astype(np.uint8)  # 确保数据类型为 uint8
        Image.fromarray(cropped_image).save(out_hr_path)

        # 下采样并保存为 PNG 格式
        downsampled_image = downsample_image(cropped_image)
        downsampled_image = add_speckle_noise(downsampled_image, scale=0.2)
        downsampled_image = add_motion_blur(downsampled_image, kernel_size=5, angle=15)
        out_ds_path = os.path.join(out_ds_folder, f"{counter:04d}.png")
        Image.fromarray(downsampled_image).save(out_ds_path)

        # print(f"已处理并保存: {out_ds_path} 和 {out_hr_path}")
        
def process_all(input_folder, out_ds_folder, out_hr_folder):
    counter = 1
    tags = ['tl', 'tr', 'bl', 'br']
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            for tag in tags:
                process_single(input_path, out_ds_folder, out_hr_folder, tag, counter)
                counter += 1

if __name__ == "__main__":
    input_folder = "dataset/WHU"
    out_ds_folder = "dataset/WHU_cropped/lr_926"
    out_hr_folder = "dataset/WHU_cropped/hr_2560"
    process_all(input_folder, out_ds_folder, out_hr_folder)      
