# CS231n 2026

本仓库保存 CS231n 2026 的作业文件。

Assignment 1 已完成。Assignment 2 的五份 Notebook 编程练习、实验及问答已完成；学生声明由本人填写。

- `assignment1/`：Assignment 1 的代码、Notebook、数据集和保存的模型。
- `assignment2/`：Assignment 2 的实现、实验输出、问答和本地环境配置。
- `assignment3/`：Transformer、SimCLR、DDPM、CLIP/DINO 的实现与实验；Q3/Q4 已完整执行并保存输出，见 [本地学习说明](assignment3/README-local.md)。

## Assignment 2

已完成 Batch Normalization / Layer Normalization、Dropout、Convolutional Networks、PyTorch 和 RNN Image Captioning。当前 RNN Notebook 只要求 vanilla RNN；模板保留的 LSTM 函数不在本次练习范围内。

| 检查项 | 结果 |
| --- | --- |
| Module API 三层卷积网络 | 一轮训练后验证准确率 47.70%，超过 45% 要求 |
| Sequential API 三层卷积网络 | 一轮训练后验证准确率 58.60%，超过 55% 要求 |
| CIFAR-10 开放挑战 | 最佳验证准确率 88.60%；已保存的单次测试准确率 88.06% |
| RNN captioning 前向检查 | 损失差值约 2.61e-12 |
| RNN 小数据过拟合 | 最终损失约 0.01337 |

补齐练习时仅使用训练集和验证集，未重新评估开放挑战的测试集。Notebook 保留运行输出；本地配置和运行顺序见 [assignment2/README-local.md](assignment2/README-local.md)。

本次 Assignment 2 提交不包含数据集、模型权重、运行日志或本地缓存。数据需另行准备，数据下载脚本保留。学生声明未代填。

## Assignment 3

Q3 已实现正向加噪、噪声与原图互相恢复、UNet 跳跃连接、加权去噪损失、反向采样和 classifier-free guidance。使用课程提供的 70000 步预训练权重，完成普通条件生成和引导生成；未从头训练 DDPM。引导采样不会修改调用方的参数字典，因此每一步都保留 guidance scale。

Q4 已实现 CLIP 图文相似度、零样本分类、缓存图片特征的文本检索，以及 DINO patch 特征上的线性分割器。五道问答已填写，注意力图、PCA 图和分割预览保存在 Notebook 中。

| 实验 | 实测结果 |
| --- | --- |
| ViT，2 个 epoch | CIFAR-10 测试准确率 46.72% |
| SimCLR 下游分类 | 无预训练 15.24%；自监督预训练 82.28% |
| CLIP 相似度 | 最大相对误差 8.32e-6，低于 1e-5 |
| DINO 首帧 / 末帧 | mean IoU 0.464 / 0.532，超过 0.45 / 0.50 |
| DINO 全视频 | mean IoU 0.625，超过 0.55 |
| 独立回归测试 | 7 项通过，覆盖扩散互逆、末步采样、梯度、CFG 参数保持、CLIP 检索缓存和分割接口 |

DINO 使用 DAVIS validation 索引 7 的 `soapbox` 视频，只用第 40 帧标签训练 500 步。分割器采用带 L2 正则化的 Adam 和基于训练帧类别频率的加权交叉熵；全视频指标包含训练帧，属于课程的单视频实验，并非独立视频泛化结果。

DDPM 的 UNet 与 CFG 原始数值检查仍未通过 `1e-6`，不能视为全部验证完成。Windows / PyTorch 2.6 的最大相对误差分别为 1.47e-5、1.18e-4。进一步对照发现，Linux 与 Windows 的测试输入完全相同，但随机初始化权重有 8377 个元素不同，最大差值为 2.98e-8；Linux、macOS 和不同 CPU 后端也未同时复现两项参考阈值。这证实环境会影响结果，但尚未确定参考值的生成环境，也未证明这是偏差的唯一原因。课程最新模板与本地参考检查相同，参考数组及阈值均未修改。

独立公式、梯度、CFG 恒等式检查及预训练模型生成已验证。原始数值检查现在有[独立脚本](assignment3/tests/check_ddpm_references.py)，未达标时返回失败状态，避免只打印误差却被当作测试通过。详见[本地数值](assignment3/q3_numeric_checks.json)及[跨平台对照](assignment3/ddpm_cross_platform_checks.json)。

对照用户提供的 `congyuxiaoyoudao/cs231n` 后，确认两份 UNet 实现在本地 2026 测试中的输出逐元素完全一致。该仓库的 2025 版 Notebook 也保存了未达到 1e-6 的 CFG 结果，不能作为两项检查均已通过的依据。版本差异和实测结果见[参考实现对照](assignment3/ddpm_reference_comparison.json)。

代码实现、调试和问答整理使用了 Codex 辅助。Notebook 保存实际运行结果；数据集、模型权重、缓存及生成视频不提交到 Git。课程学生声明仍由本人填写。

## 获取仓库

仓库已有的 Assignment 1 数据及部分图片使用 Git LFS 管理。只获取代码、暂不下载历史 LFS 文件时，可以先设置 `GIT_LFS_SKIP_SMUDGE=1` 再克隆。例如在 PowerShell 中：

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/JiangShiYu6/cs231n-2026.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

如需获取仓库原有的 LFS 文件：

```bash
git lfs install
cd cs231n-2026
git lfs pull
```
