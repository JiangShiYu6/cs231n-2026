在“深度学习”目录双击 `start-assignment2.cmd`，会打开 JupyterLab 和 `BatchNormalization.ipynb`。也可以在 VS Code 中打开作业，选择 `Python (CS231n)` 内核，或手动选择 `.venv-cs231n/Scripts/python.exe`。

建议按下面的顺序做，每个 Notebook 从第一个代码单元格开始运行：

1. `BatchNormalization.ipynb`
2. `Dropout.ipynb`
3. `ConvolutionalNetworks.ipynb`
4. `PyTorch.ipynb`
5. `RNN_Captioning_pytorch.ipynb`

环境使用 Python 3.12、NumPy 1.26.4、PyTorch 2.6.0 + CUDA 12.4 和 torchvision 0.21.0。PyTorch Notebook 已有自动选择 GPU 的代码，初始化后应显示 `using device: cuda`。版本配对依据 [PyTorch 官方安装说明](https://docs.pytorch.org/get-started/previous-versions/)。

CIFAR-10 和 COCO 位于 `cs231n/datasets/`。卷积的 Cython 扩展已经用本机 MinGW 编译；对应单元格调用 `build_extensions()`，通常会直接显示已就绪。COCO 的示例原图仍需联网读取；预览会自动替换失效图片，并限制超时和尝试次数。训练使用本地特征，不依赖这些原图链接。数据集和编译产物不随本次提交上传。

五份 Notebook 的编程练习、实验和问答已完成，学生声明保留未填。Module API 和 Sequential API 卷积练习补齐后均完成一轮训练，验证准确率分别为 47.70% 和 58.60%。开放挑战已有验证准确率 88.60%、测试准确率 88.06% 的运行结果，本次未重复运行测试集。当前 RNN 作业只要求 vanilla RNN，模板中的 LSTM 函数仍保留待实现。

本机原始 Notebook 备份位于工作区 `tmp/assignment2-original-notebooks/`。模型权重、数据集、运行日志和本地缓存不上传；如需模型权重，可重新运行对应训练单元格生成。

`requirements-local.txt` 记录主要依赖，`requirements-local.lock.txt` 记录配置完成时的完整包版本。此环境与 Assignment 1 共用。

`collect_submission.ipynb` 是课程原有的 Colab 提交工具，仍需在 Colab 运行；本地启动脚本用于完成作业，不调用这个提交工具。
