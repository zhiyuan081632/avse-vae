# NTCD-TIMIT 训练快速指南

## 环境配置

- **工程目录**: `/mnt/d/project/prjANS/src/AVSE/avse-vae`
- **数据目录**: `/mnt/d/project/prjANS/src/data/NTCD-TIMIT/TCDTIMIT`

## 快速开始

### 方法 1: 快速测试（推荐首次使用）⭐

**只需1分钟快速验证训练流程！**

```bash
cd /mnt/d/project/prjANS/src/AVSE/avse-vae

# 添加执行权限
chmod +x run.sh

# 快速测试：少量数据+少轮训练
./run.sh --quick
```

**快速测试配置**：
- 📊 数据：2个训练说话人，1个验证说话人，每人5个文件（约10-15个音频）
- 🏃 训练：batch_size=32, epochs=5
- ⏱️ 时间：约1-3分钟完成
- 🎯 目的：快速验证环境和训练流程是否正常

### 方法 2: 一键完整训练

```bash
cd /mnt/d/project/prjANS/src/AVSE/avse-vae

# 准备全部数据并训练
./run.sh --all
```

### 方法 3: 分步执行

```bash
cd /mnt/d/project/prjANS/src/AVSE/avse-vae

# 步骤1: 准备数据
python3 prepare_ntcd_timit_data.py              # 完整数据
python3 prepare_ntcd_timit_data.py --quick_test # 快速测试数据

# 步骤2: 训练模型
python3 train_with_ntcd_timit.py              # 完整训练
python3 train_with_ntcd_timit.py --quick_test # 快速测试
```

## run.sh 常用命令

```bash
./run.sh --help              # 查看帮助
./run.sh --quick             # 快速测试（推荐首次使用）⭐
./run.sh --check             # 检查环境
./run.sh --prepare           # 仅准备数据
./run.sh --train             # 仅训练
./run.sh --all               # 准备+训练（完整数据）

# 使用不同噪声
./run.sh --all --noise Babble --snr 0
./run.sh --quick --noise Babble --snr 0  # 快速测试带噪声数据
```

## 快速测试 vs 完整训练对比

| 项目 | 快速测试 | 完整训练 |
|------|---------|----------|
| 数据量 | 10-15个文件 | 数百个文件 |
| 训练轮数 | 5 epochs | 50+ epochs |
| 批次大小 | 32 | 128 |
| 训练时间 | 1-3分钟 | 数小时 |
| 用途 | 验证流程 | 实际训练 |

## 训练结果

**快速测试**模型保存在 `saved_model/` 目录:
- `ntcd_timit_quicktest_checkpoint.pt` - 快速测试检查点
- `final_model_ntcd_timit_quicktest.pt` - 快速测试最终模型

**完整训练**模型保存在 `saved_model/` 目录:
- `ntcd_timit_checkpoint.pt` - 最佳检查点
- `final_model_ntcd_timit.pt` - 最终模型

## 测试模型

### 快速测试模型

```bash
# 使用默认参数测试快速训练的模型
python3 test_model.py

# 或指定参数
python3 test_model.py \
    --model ./saved_model/final_model_ntcd_timit_quicktest.pt \
    --audio ./validation_speech/validation/03F/sa1.wav \
    --output ./test_results
```

**测试会做什么**:
1. ✅ 加载训练好的模型
2. ✅ 读取测试音频
3. ✅ 提取 STFT 特征
4. ✅ 通过 VAE 编码-解码
5. ✅ 重建音频信号
6. ✅ 保存原始和重建音频
7. ✅ 计算简单的评估指标（MSE、相关系数）

**输出文件** (`test_results/`):
- `original.wav` - 原始音频
- `reconstructed.wav` - 模型重建的音频

### 测试完整模型

```bash
# 测试完整训练的模型
python3 test_model.py \
    --model ./saved_model/final_model_ntcd_timit.pt \
    --audio ./validation_speech/validation/03F/sa2.wav
```

## 常见问题

**Q: 权限错误？**
```bash
chmod +x run.sh
```

**Q: 找不到模块？**
```bash
pip3 install torch librosa numpy scipy soundfile
```

**Q: 内存不足？**
修改 `train_with_ntcd_timit.py` 第 158 行：
```python
batch_size = 64  # 从 128 减小
```

**Q: 维度不匹配错误？**
如果看到 `mat1 and mat2 shapes cannot be multiplied`，说明 TCD_TIMIT.py 已自动修复。
在纯音频模式下，会自动创建 4489 维的零向量作为视频输入占位符。

完成！
