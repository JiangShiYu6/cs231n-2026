# Assignment 3 本地学习环境

在工作区“深度学习”目录双击 `start-assignment3.cmd`，会使用现有 `.venv-cs231n` 环境启动 JupyterLab，并打开 `Transformer_Captioning.ipynb`。也可以直接用 VS Code 打开 Notebook，选择 `Python (CS231n)` 内核，或选择工作区下的 `.venv-cs231n/Scripts/python.exe`。

建议按以下顺序学习，每份 Notebook 先运行第一个本地配置单元格：

1. `Transformer_Captioning.ipynb`
2. `Self_Supervised_Learning.ipynb`
3. `CLIP_DINO.ipynb`
4. `DDPM.ipynb`

四项算法练习已实现。Q3 DDPM 和 Q4 CLIP/DINO 已按单元格顺序完整运行，Q4 五道问答已填写，Notebook 中保留实际图像和数值输出。DDPM 使用课程预训练权重；没有执行可选的从头训练。

## 完成情况与复现

| 项目 | 验证结果 |
| --- | --- |
| ViT | 2 个 epoch，测试准确率 46.72% |
| SimCLR | 下游分类基线 15.24%，预训练后 82.28% |
| DDPM | 正向/反向过程、UNet、两种预测目标与 CFG 已实现；预训练权重成功加载，两组 emoji 生成已保存 |
| CLIP | 相似度相对误差 8.32e-6；零样本分类与文本检索已运行 |
| DINO | 首帧 IoU 0.464，末帧 0.532，全视频 0.625，三项超过题目要求 |

从本目录运行独立测试：

```powershell
python -m unittest discover -s tests -v
```

7 项测试不需要下载数据或预训练权重。DINO 实验使用 DAVIS validation 索引 7（`soapbox`，99 帧），以第 40 帧的 3600 个 patch 训练线性分类器 500 步。Adam 学习率为 0.01、weight decay 为 1.0；交叉熵类别权重仅根据训练帧的标签计数计算。训练帧也包含在全视频均值内。

DDPM 的两项严格数值检查仍未通过。UNet 和 CFG 最大相对误差分别为 1.47e-5、1.18e-4，超过课程要求的 1e-6；绝对误差分别为 1.39e-6、6.45e-6。2026-10-03 下载的课程官方模板包含相同参考数组，不能把它们直接称为过时或错误的参考值。

Linux 与 Windows 的测试图片和文本输入逐项相同，但初始化权重有 8377 个元素不同，最大差值为 2.98e-8。Linux PyTorch 2.3/2.6、macOS ARM 和 CPU 后端对照仍未同时达到两项阈值，详见 `ddpm_cross_platform_checks.json` 中的实测数据及 GitHub Actions 链接。环境差异已有证据；参考值的原始运行环境及全部误差来源尚未确定。独立回归测试通过不代表这两项检查通过。

从本目录运行严格检查：

```powershell
python tests/check_ddpm_references.py
```

脚本直接执行 Notebook 中未修改的两段测试；任何一项超过 `1e-6` 都返回非零退出码。`--diagnostic` 只收集不同线程数和后端的数据，明确打印失败项数量。GitHub 的手动 `DDPM reference diagnostics` 工作流也只收集诊断，其任务执行成功不代表数值达标。原参考数组、误差公式及阈值均未放宽；当前仍需解决严格复现问题。

另外对照了用户提供的 [congyuxiaoyoudao/cs231n](https://github.com/congyuxiaoyoudao/cs231n/tree/8d3fbd0ad61ded86440c72a33628c742865495cb/assignment3)。这是 2025 版作业，其 UNet 测试在训练模式运行，参考数组与当前 2026 版不同：保存的 UNet 相对误差为 6.07e-6，符合当时的 1e-5；CFG 相对误差为 6.04e-5，仍超过其 1e-6 要求。该仓库没有锁定 PyTorch 版本。把它的 UNet 实现直接用于本地未修改的 2026 测试后，两份实现的权重及输出逐元素相同，最大输出差为 0；两项误差仍分别为 1.47e-5、1.18e-4。因此，这份参考实现没有提供能消除当前误差的代码差异。实测记录见 `ddpm_reference_comparison.json`，没有将其代码或旧参考数组替换到本地实现中。

SimCLR 的数据增强检查在单进程加载下复现 worker 0 的随机种子，使原参考误差为 0。CLIP 的失效 Flickr 图片可能被可用的图文对替换，因此分类文字和检索示例不保证与原说明逐项相同。

## 环境

使用 Python 3.12、NumPy 1.26.4、PyTorch 2.6.0 + CUDA 12.4、torchvision 0.21.0。已验证本机 RTX 4060 Laptop GPU 可用。新增 CLIP、einops、thop、pandas、scikit-learn、OpenCV 和 TensorFlow Datasets 等依赖；TensorFlow 仅用于 DAVIS 数据读取，模型计算仍使用 PyTorch。

`requirements-local.txt` 记录主要依赖，`requirements-local.lock.txt` 记录完整环境。课程原始 `requirements.txt` 是旧环境清单，不要在当前 Python 3.12 环境中安装它。若要在新环境中恢复配置，可执行：

```powershell
python -m pip install -r requirements-local.txt
```

安装方式参考 [CLIP 官方说明](https://github.com/openai/CLIP) 和 [DINO 官方仓库](https://github.com/facebookresearch/dino)。CLIP 固定到本次验证的源码提交。

## 数据和模型

- CIFAR-10：复用 Assignment 1 数据，位于 `data/cifar-10-batches-py/`。
- COCO：复用 Assignment 2 特征与描述，位于 `cs231n/datasets/coco_captioning/`。
- SimCLR：`pretrained_model/pretrained_simclr_model.pth`。
- Emoji：`cs231n/datasets/emoji_data.npz` 和 `text_embeddings.pt`。
- DDPM：`cs231n/exp/pretrained/model-70000.pt`。
- CLIP 与 DINO：`.cache/clip/` 和 `.cache/torch/`。
- DAVIS：`cs231n/datasets/tensorflow_datasets/`；下载和解压缓存使用工作区 `tmp/a3-downloads/`，避免 Windows 路径过长。

上述数据、模型和缓存已加入 `.gitignore`，不会随代码提交。CIFAR-10、COCO 在同一磁盘上优先使用硬链接，避免重复占用空间。CLIP 示例会保持图片和描述配对，并替换失效的 Flickr 链接；成功下载的图片保存在 `.cache/images/`。

本地适配还包括移除 Colab/Google Drive 路径、替换 Bash 下载单元格、将 Notebook 中的 DataLoader 工作进程数设为 0，以及修正 OpenCV 掩码缩放的插值参数。DINO 的 DAVIS 验证集及索引 7 保持课程原有设置。

## 验证与备份

原始 Notebook 备份位于工作区 `tmp/assignment3-original-notebooks/`；`tmp/verified-a3-*.ipynb` 是早期环境检查输出。当前完成结果以本目录的 Notebook 为准。Q3/Q4 的全部代码单元格已执行，DINO 可视化视频保存在本地 `dino_res.mp4`，不上传 GitHub。代码与问答整理使用 Codex 辅助。

`collect_submission.ipynb` 是原课程的 Colab 提交工具，保留原样；日常本地学习无需运行它。
