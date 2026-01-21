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

# 模型对比
## 总述：
AV‑CVAE (含视觉先验) ≳ AV‑CVAE‑VI ≥ AV‑VAE ≫ A‑VAE / A‑VAE‑VI即：带视频的条件 AV‑CVAE 系列最好，普通 AV‑VAE 次之，纯音频 A‑VAE 最差。

## How this comes from the repo & papers
README 引用的 3 篇文章大致对应几类模型：
[1] Audio‑visual Speech Enhancement Using Conditional VAE → AV‑CVAE 类（条件 VAE，视觉作为条件/先验）
[2] Robust Unsupervised Audio‑visual SE Using Mixture of VAEs → 多 VAE 混合，提升鲁棒性
[3] Mixture of Inference Networks for VAE‑based AV SE → 改进推断网络/推理算法的版本

## 工程对比
工程里的主要几种模型/推理组合是（在 speech_enhance_VAE.py / test_avse.py 里）：
A‑VAE：音频 VAE，完全不利用视频；
AV‑VAE：标准 “音频 + 视频” VAE，z 的先验是标准高斯；
AV‑CVAE：条件 VAE，z 的先验由视频估计（zprior），理论上更好地利用视觉信息；
A‑VAE‑VI / AV‑CVAE‑VI：在上面基础上，用 VI‑MCEM 这种变分版本的推理算法，通常会让估计更稳一些。

根据原论文和典型实验结论：
在 视频信息正常、没有严重失配 的前提下：
AV‑CVAE 系列（特别是带视觉先验的）一般优于普通 AV‑VAE；
AV‐VAE 明显优于 A‑VAE（只音频）；
在 只有音频、不用视频 的场景，A‑VAE/A‑VAE‑VI 是唯一选择，VI 版通常会略好一点或更稳定。


## 对比信息

graph LR

  %% Papers
  P1["[1] AV-CVAE (TASLP 2020)"]
  P2["[2] Mixture of VAEs (ICASSP 2020)"]
  P3["[3] Mixture of Inference Nets (TSP 2021)"]

  %% Model families (architectures)
  M_A["A-VAE (audio-only)\nmyVAE with blockVenc=1, blockVdec=1"]
  M_AV["AV-VAE (audio+video)\nmyVAE with blockVenc=0, blockVdec=0"]
  M_CVAE["AV-CVAE (conditional)\nCVAE / CVAER / CVAERTied2"]

  %% Training scripts
  S_train_vae["train_VAE.py\n(original AV-TCD-TIMIT)"]
  S_train_ntcd["train_with_ntcd_timit.py\n(NTCD-TIMIT AV-VAE)"]
  S_train_cvae["train_CVAE.py\n(AV-CVAE, CVAERTied2)"]

  %% Checkpoints / model files
  C_A["A_VAE_checkpoint.pt\n(audio-only baseline)"]
  C_AV["AV_VAE_checkpoint.pt\n(AV-VAE baseline / NTCD version)"]
  C_AV_cvae["AV_CVAE_checkpoint.pt\n(AV-CVAE main model)"]
  C_ntcd["ntcd_timit*_checkpoint.pt\n(NTCD AV-VAE best ckpt)"]

  %% Inference algorithms
  I_mcem["MCEM_algo\n(for VAE / AV-VAE)"]
  I_mcem_cvae["MCEM_algo_cvae\n(for AV-CVAE)"]
  I_vi["VI_MCEM_algo\n(variational MCEM)"]

  %% Inference / evaluation scripts
  R_se["speech_enhance_VAE.py\nrun: AV-VAE, AV-CVAE, A-VAE, A-VAE-VI, AV-CVAE-VI"]
  R_test["test_avse.py / speech_avse.py\nrun selected mode by --mode"]

  %% Paper → model / algorithm
  P1 --> M_CVAE
  P1 --> I_mcem_cvae
  P1 --> I_vi

  P2 --> M_AV
  P2 --> M_CVAE

  P3 --> M_AV
  P3 --> M_CVAE
  P3 --> I_vi

  %% Model arch → training scripts
  M_A --> S_train_vae
  M_AV --> S_train_vae
  M_AV --> S_train_ntcd
  M_CVAE --> S_train_cvae

  %% Training scripts → checkpoints
  S_train_vae --> C_A
  S_train_vae --> C_AV
  S_train_cvae --> C_AV_cvae
  S_train_ntcd --> C_ntcd
  S_train_ntcd --> C_AV

  %% Checkpoints → inference scripts
  C_A --> R_se
  C_AV --> R_se
  C_AV_cvae --> R_se
  C_ntcd --> R_se

  C_A --> R_test
  C_AV --> R_test
  C_AV_cvae --> R_test