# CS231n 2026

本仓库保存 CS231n 2026 的作业文件。

Assignment 1 已完成。Assignment 2 的五份 Notebook 编程练习、实验及问答已完成；学生声明由本人填写。

- `assignment1/`：Assignment 1 的代码、Notebook、数据集和保存的模型。
- `assignment2/`：Assignment 2 的实现、实验输出、问答和本地环境配置。
- `assignment3/`：Assignment 3 作业文件。

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
