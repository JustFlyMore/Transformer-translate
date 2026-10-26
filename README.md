## 本项目是基于Transformer的中译英翻译模型
#### 一、具体实现步骤
###### 1.环境准备
我使用的环境是python3.11.17，windows11,cuda-version:13.0
###### 2.语料准备
我用的是opus上面的中英平行语料UNPC, 地址在https://opus.nlpl.eu/datasets/UNPC?pair=en&zh

### 训练代码的资源优化

打开 `src/model/Transformer_translate.ipynb`，使用安装了 PyTorch 和 SentencePiece 的环境。更新代码后重启 Notebook 内核，再从头执行，避免旧模型、数据集和中间张量继续占用内存。支持从项目根目录或 Notebook 所在目录启动。

- 原文逐行读取；Token 使用连续的 int64 存储，组批时直接读取 Tensor 视图，避免每轮反复把 Python 整数列表转换成 Tensor。
- 训练样本先随机分池，再按句长组批并打乱批次，每轮通过 `set_epoch` 改变顺序。所有样本都保留，包括不足一个批次的尾部。验证按长度组批。
- 每个批次在前向计算之前清理梯度，批次结束后释放 logits；验证前也释放最后一批的梯度。
- Padding mask 在计算设备上生成，因果 mask 每轮创建一次并切片复用。损失和有效 Token 数在设备上累计，只在日志输出及轮末读取。
- 保持 FP32、批次大小 32、模型结构、截断长度、Adam 学习率及有效 Token 平均损失不变。未启用混合精度或增加数据加载进程。分桶改变样本顺序，训练曲线不保证逐步相同。
- 正式训练前不再执行额外的演示参数更新。每轮输出有效 Token/s 和 CUDA 峰值已分配显存；后者不包含全部驱动开销，也不同于显存保留量或任务管理器读数。

仍然只在验证损失改善时保存 `checkpoints/best_transformer.pt`。执行正式训练会更新这个文件，如需保留上一轮最佳模型，请先另存。小批次测试不会写入训练 checkpoint。

在项目根目录运行回归验证：

```powershell
& 'D:\Anaconda\envs\Transformer-translate\python.exe' -m unittest discover -s tests -v
```

测试覆盖 Token/标签与旧实现的一致性、空数据和错行检查、分桶样本覆盖，以及 CPU/CUDA 上的损失和参数更新等价性。没有 CUDA 时跳过 GPU 检查。

本机抽样检查（前 4096 条训练语料、最大长度 128）：Token 存储估算由 7,450,880 字节降至 2,072,360 字节；相同样本的源序列加解码器输入的批次位置总数由 794,400 降至 284,352。前者不代表整个进程内存，后者包含有效 Token 与 Padding，不能直接当作实际提速比例。完整训练的速度和显存应以每轮日志为准。
