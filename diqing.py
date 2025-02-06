import os
from osgeo import gdal
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def read_sar_image(file_path):
    """使用 GDAL 读取 SAR .tif 图像"""
    dataset = gdal.Open(file_path)
    
    if dataset is None:
        print(f"错误：无法打开文件 {file_path}，请检查路径！")
        return None
    
    band = dataset.GetRasterBand(1)  # 读取第一波段（SAR数据通常是单波段）
    sar_image = band.ReadAsArray().astype(np.float32)  # 读取为 NumPy 数组

    # 归一化到 0-255 范围（适用于可视化）
    sar_image = (255 * (sar_image - np.min(sar_image)) / (np.max(sar_image) - np.min(sar_image))).astype(np.uint8)
    
    return sar_image

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

def process_sar_image(input_folder, output_folder):
    """对 input_folder 中的所有 SAR .tif 文件进行批量处理，并保存到 output_folder"""
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历所有 .tif 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)  # 输出文件名保持不变

            print(f"处理文件: {input_path}")

            # 读取 SAR 图像
            sar_img = read_sar_image(input_path)
            if sar_img is None:
                print(f"跳过 {filename}，因为读取失败！")
                continue

            # 低通滤波（去掉部分高频信息）
            sar_blur = gaussian_filter(sar_img, sigma=1.5)

            # 下采样（降低分辨率）
            low_res_img = cv2.resize(sar_blur, (sar_img.shape[1] // 4, sar_img.shape[0] // 4), interpolation=cv2.INTER_NEAREST)

            # 添加SAR特性噪声
            noisy_img = add_speckle_noise(low_res_img, scale=0.2)
            motion_blurred_img = add_motion_blur(noisy_img, kernel_size=5, angle=15)

            # 保存最终模拟低清SAR图像
            cv2.imwrite(output_path, motion_blurred_img)
            print(f"已处理并保存: {output_path}")

    print("所有SAR图像已处理完成！")


# **使用批量处理函数**
input_folder = "WHU"# 原始 SAR 图片文件夹
output_folder = "WHU_lr"# 处理后输出文件夹

process_sar_image(input_folder, output_folder)
