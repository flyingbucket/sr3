import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def process_single(input_path, out_folder, row, col, counter, size, step):
    with rasterio.open(input_path) as src:
        x_offset = col * step
        y_offset = row * step

        if x_offset + size > src.width or y_offset + size > src.height:
            return  # Skip this tile if it exceeds the image boundaries

        window = Window(x_offset, y_offset, size, size)
        cropped_image = src.read(window=window)
        meta = src.meta.copy()
        meta.update({
            "width": size,
            "height": size,
            "transform": src.window_transform(window)
        })
        
        out_path = os.path.join(out_folder, f"{counter:04d}.png")
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(cropped_image)

        
        # 将裁剪后的图像转换为 PNG 格式并保存
        # cropped_image = np.moveaxis(cropped_image, 0, -1)  # 将波段轴移到最后
        # if cropped_image.shape[2] == 1:  # 如果是单波段图像，转换为三通道
        #     cropped_image = np.repeat(cropped_image, 3, axis=2)
      
        # cropped_image = cropped_image.astype(np.uint8)  # 确保数据类型为 uint8
        # Image.fromarray(cropped_image).save(out_path)

def process_all(input_folder, out_folder):
    counter = 1
    size = 512
    overlap = 256
    step = size - overlap
    tasks = []
    with ProcessPoolExecutor() as executor:
        for filename in tqdm(os.listdir(input_folder)):
            if filename.endswith(".tif"):
                input_path = os.path.join(input_folder, filename)
                with rasterio.open(input_path) as src:
                    rows = (src.height - overlap) // step + 1
                    cols = (src.width - overlap) // step + 1
                    for row in range(rows):
                        for col in range(cols):
                            tasks.append(executor.submit(process_single, input_path, out_folder, row, col, counter, size, step))
                            counter += 1
        for task in tqdm(tasks):
            task.result()  # 等待所有任务完成

if __name__ == "__main__":
    input_folder = "../datastore/WHU_mountain"
    out_folder = "dataset/WHU_512_mountain"
    os.makedirs(out_folder, exist_ok=True)
    process_all(input_folder, out_folder)
    print("All done!")