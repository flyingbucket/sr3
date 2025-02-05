# 通过迭代优化实现图像超分辨率

[论文](https://arxiv.org/pdf/2104.07636.pdf) | [项目](https://iterative-refinement.github.io/)

## 简介

这是 **PyTorch** 实现的 **Image Super-Resolution via Iterative Refinement（SR3）** 的非官方版本。

由于论文描述可能存在细节缺失，我们的实现与 `SR3` 的实际结构可能有所不同，主要调整包括：

- 采用类似 `DDPM` 的 ResNet 块和通道拼接方式。
- 在低分辨率特征（$16 \times 16$）上使用 `DDPM` 方式的注意力机制。
- 采用 `WaveGrad` 中 `FilM` 结构对 $γ$ 进行编码，并在嵌入时不进行仿射变换。
- 将后验方差定义为 $\dfrac{1-\gamma_{t-1}}{1-\gamma_{t}} \beta_t$，而非 $β_t$，结果与论文接近。

**如果你只想使用预训练模型将 $(64 \times 64)\text{px} \rightarrow (512 \times 512)\text{px}$，请参考[这个 Google Colab 脚本](https://colab.research.google.com/drive/1G1txPI1GKueKH0cSi_DgQFKwfyJOXlhY?usp=sharing)。**

## 状态

**★★★ 新进展：[Palette-Image-to-Image-Diffusion-Models](https://arxiv.org/abs/2111.05826) 现已发布，详见[此处](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) ★★★**

### 条件生成（超分辨率）
- [x] 16×16 -> 128×128（FFHQ-CelebaHQ）
- [x] 64×64 -> 512×512（FFHQ-CelebaHQ）

### 无条件生成
- [x] 128×128 人脸生成（FFHQ）
- [ ] ~~1024×1024 人脸生成（由 3 个级联模型组成）~~

### 训练进度
- [x] 日志 / 记录器
- [x] 评估指标
- [x] 多 GPU 支持
- [x] 训练恢复 / 预训练模型
- [x] 独立验证脚本
- [x] [Weights and Biases 记录](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/pull/44) 🌟 **NEW**

## 结果

*注意*：我们将最大逆向步数设为 $2000$，并限制模型参数适配 `Nvidia 1080Ti`，高分辨率图像可能出现**噪声和色调偏差**，导致评分较低，仍有优化空间。**欢迎贡献更广泛的实验与代码改进！**

| 任务 / 指标 | SSIM（+） | PSNR（+） | FID（-） | IS（+） |
| -------- | ------- | ------- | ---- | ---- |
| 16×16 -> 128×128 | 0.675 | 23.26 | - | - |
| 64×64 -> 512×512 | 0.445 | 19.87 | - | - |

更多结果：[16×16 -> 128×128](https://drive.google.com/drive/folders/1Vk1lpHzbDf03nME5fV9a-lWzSh3kMK14?usp=sharing) | [64×64 -> 512×512](https://drive.google.com/drive/folders/1yp_4xChPSZUeVIgxbZM-e3ZSsSgnaR9Z?usp=sharing)

## 使用方法

### 环境安装
```python
pip install -r requirement.txt
```

### 预训练模型
本项目基于 "Denoising Diffusion Probabilistic Models"，实现了 DDPM/SR3 结构，分别使用时间步长和 gamma 作为模型输入。在实验中，SR3 模型在相同逆向步数和学习率下表现更佳。

| 任务 | 预训练模型 |
| ---- | ---- |
| 16×16 -> 128×128 | [Google Drive](https://drive.google.com/drive/folders/12jh0K8XoM1FqpeByXvugHHAF3oAZ8KRu?usp=sharing) |  
| 64×64 -> 512×512 | [Google Drive](https://drive.google.com/drive/folders/1mCiWhFqHyjt5zE4IdA41fjFwCYdqDzSF?usp=sharing) |
| 128×128 人脸生成 | [Google Drive](https://drive.google.com/drive/folders/1ldukMgLKAxE7qiKdFJlu-qubGlnW-982?usp=sharing) |

下载预训练模型后，修改 `sr|sample_[ddpm|sr3]_[resolution option].json`：
```json
"resume_state": [你的预训练模型路径]
```

### 训练 / 评估
```python
# 训练超分辨率任务
python sr.py -p train -c config/sr_sr3.json

# 评估模型
python sr.py -p val -c config/sr_sr3.json
python eval.py -p [结果路径]
```

### 生成推理
```python
python infer.py -c [配置文件]
```

## Weights and Biases 🎉
支持实验跟踪、模型检查点和可视化。安装 `wandb` 并登录：
```shell
pip install wandb
wandb login
```
使用 `-enable_wandb` 进行日志记录：
- `-log_wandb_ckpt` 保存模型检查点。
- `-log_eval` 记录评估结果。
- `-log_infer` 记录推理结果。

更多信息请查看[此处](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/pull/44)。🚀

## 致谢
本项目基于以下研究：
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

受以下项目启发：
- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hejingwenhejingwen/AdaFM

