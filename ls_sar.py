import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image

def crop_tif_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            with rasterio.open(input_path) as src:
                width, height = src.width, src.height
                square_size = 2560  # 2560x2560
                
                # 左上角裁剪
                top_left_crop = src.read(window=Window(0, 0, square_size, square_size))
                top_left_meta = src.meta.copy()
                top_left_meta.update({"width": square_size, "height": square_size, "transform": src.window_transform(Window(0, 0, square_size, square_size))})
                top_left_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_top_left.tif")
                with rasterio.open(top_left_output_path, "w", **top_left_meta) as dst:
                    dst.write(top_left_crop)
                
                # 右上角裁剪
                top_right_crop = src.read(window=Window(width - square_size, 0, square_size, square_size))
                top_right_meta = src.meta.copy()
                top_right_meta.update({"width": square_size, "height": square_size, "transform": src.window_transform(Window(width - square_size, 0, square_size, square_size))})
                top_right_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_top_right.tif")
                with rasterio.open(top_right_output_path, "w", **top_right_meta) as dst:
                    dst.write(top_right_crop)
                
                # 左下角裁剪
                bottom_left_crop = src.read(window=Window(0, height - square_size, square_size, square_size))
                bottom_left_meta = src.meta.copy()
                bottom_left_meta.update({"width": square_size, "height": square_size, "transform": src.window_transform(Window(0, height - square_size, square_size, square_size))})
                bottom_left_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_bottom_left.tif")
                with rasterio.open(bottom_left_output_path, "w", **bottom_left_meta) as dst:
                    dst.write(bottom_left_crop)
                
                # 右下角裁剪
                bottom_right_crop = src.read(window=Window(width - square_size, height - square_size, square_size, square_size))
                bottom_right_meta = src.meta.copy()
                bottom_right_meta.update({"width": square_size, "height": square_size, "transform": src.window_transform(Window(width - square_size, height - square_size, square_size, square_size))})
                bottom_right_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_bottom_right.tif")
                with rasterio.open(bottom_right_output_path, "w", **bottom_right_meta) as dst:
                    dst.write(bottom_right_crop)

def convert_tif_to_png(directory):
    assert os.path.isdir(directory), f'{directory} is not a valid directory'
    
    for dirpath, _, filenames in tqdm(os.walk(directory), desc="Converting TIF to PNG"):
        for filename in tqdm(filenames, desc=f"Processing {dirpath}", leave=False):
            if filename.lower().endswith('.tif'):
                tif_path = os.path.join(dirpath, filename)
                png_path = os.path.splitext(tif_path)[0] + '.png'
                
                with Image.open(tif_path) as img:
                    img.save(png_path, 'PNG')
                os.remove(tif_path)  # 删除原始的 .tif 文件
                print(f'Converted {tif_path} to {png_path} and removed the original .tif file')

def rename_as_number(directory):
    assert os.path.isdir(directory), f'{directory} is not a valid directory'
    
    for dirpath, _, filenames in tqdm(os.walk(directory), desc="Renaming PNG files"):
        png_files = [f for f in filenames if f.lower().endswith('.png')]
        for i, filename in enumerate(tqdm(png_files, desc=f"Processing {dirpath}", leave=False)):
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, f'{i:04}.png')
            os.rename(old_path, new_path)
            print(f'Renamed {old_path} to {new_path}')



if __name__ == "__main__":
    input_dir = "WHU_lr"
    output_dir = "dataset/WHU_cropped/lr_926"
    crop_tif_images(input_dir, output_dir)
    convert_tif_to_png(output_dir)
    rename_as_number(output_dir)
    print(f"Processed images saved in {output_dir}")