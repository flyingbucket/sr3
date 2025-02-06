from osgeo import gdal

gdal.UseExceptions()
# 读取 SAR 图像
dataset = gdal.Open("./whu/test.tif")
band = dataset.GetRasterBand(1)
sar_image = band.ReadAsArray()

# 低通滤波 + 降采样
import cv2
import numpy as np

sar_blur = cv2.GaussianBlur(sar_image, (5, 5), 0)
sar_low_res = cv2.resize(
    sar_blur,
    (sar_image.shape[1] // 4, sar_image.shape[0] // 4),
    interpolation=cv2.INTER_NEAREST,
)

# 保存结果
cv2.imwrite("./whu/test_res.png", sar_low_res)
