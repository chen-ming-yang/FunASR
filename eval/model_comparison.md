# ASR 模型对比分析报告

> Fun-ASR-Nano vs Paraformer-large vs Qwen3-ASR  
> 评测时间: 2026-05-18  
> 评测环境: CUDA GPU, 200 条音频, 单条推理 (batch_size=1)

---

## 一、模型基本信息

| 维度 | Fun-ASR-Nano | Paraformer-large | Qwen3-ASR-1.7B |
|------|-------------|-----------------|----------------|
| 架构 | Audio Encoder + LLM (AR) | CIF + NAR Decoder | Audio Encoder + LLM (AR) |
| 模型大小 | 2,029 MB | ~900 MB | ~3,400 MB (BF16) |
| 解码方式 | 自回归逐 token 生成 | 非自回归一次输出 | 自回归逐 token 生成 |
| 参数量 | ~0.6B | ~220M | 1.7B |

---

## 二、推理性能对比

| 指标 | Fun-ASR-Nano | Paraformer-large | Qwen3-ASR-1.7B |
|------|-------------|-----------------|----------------|
| 整体 RTF | 0.6804 | **0.1841** | **0.0409** |
| 实时加速比 | 1.5x | 5.4x | **24x** |
| E2E 延迟 P50 | 1,729 ms | 472 ms | **92 ms** |
| E2E 延迟 P90 | 1,827 ms | 496 ms | 169 ms |
| E2E 延迟 P99 | 1,980 ms | 521 ms | 291 ms |
| E2E 延迟 Max | 2,027 ms | 572 ms | 330 ms |

---

## 三、按音频时长分段 RTF

| 时长段 | Fun-ASR-Nano | Paraformer-large | Qwen3-ASR |
|--------|-------------|-----------------|-----------|
| 短音频 < 3s (n=142) | 1.0547 ❌ | 0.2865 | **0.0499** |
| 中音频 3-5s (n=42) | 0.4502 | 0.1201 | **0.0339** |
| 长音频 5-10s (n=15) | 0.3112 | 0.0827 | **0.0306** |
| 超长音频 10-20s (n=1) | 0.1983 | 0.0560 | **0.0323** |

---

## 四、资源占用对比

| 指标 | Fun-ASR-Nano | Paraformer-large | Qwen3-ASR-1.7B |
|------|-------------|-----------------|----------------|
| GPU 显存峰值 | 3,404 MB | **1,041 MB** | 3,978 MB |
| CPU RSS 峰值 | 5,252 MB | — | ~2,039 MB |
| 模型文件大小 | 2,029 MB | ~900 MB | ~3,400 MB |

---

## 五、识别质量 (CER)

| 模型 | CER | 备注 |
|------|-----|------|
| Fun-ASR-Nano | 48.02% | 未微调预训练模型 |
| Paraformer-large | 51.35% | 未微调，通用中文模型 |
| Qwen3-ASR | **25.04%** | 已在目标数据上微调 |

> CER 对比不完全公平：Fun-ASR-Nano 和 Paraformer-large 均为未微调模型在该测试集上的表现，Qwen3-ASR 已经过微调。微调后三者 CER 差距会缩小。

---

## 六、延迟特征分析

| 特征 | Fun-ASR-Nano | Paraformer-large | Qwen3-ASR |
|------|-------------|-----------------|-----------|
| 延迟随音频长度变化 | 几乎不变 (~1.7s 固定) | 线性增长 | 线性增长 |
| 短音频 (<3s) 实时可行 | ❌ RTF > 1.0 | ✅ | ✅ |
| 长音频 (>5s) 实时可行 | ✅ RTF = 0.2~0.3 | ✅ | ✅ |
| 延迟瓶颈 | LLM generate 固定开销 | Encoder 前向传播 | 极低，优化充分 |

**关键发现**:

- Fun-ASR-Nano 每条音频无论长短都需要 ~1.7s，瓶颈在 LLM 自回归 decode 阶段
- Paraformer-large 延迟从 464ms 线性增长到 572ms，符合 NAR 架构特征
- Qwen3-ASR 延迟从 92ms 增长到 330ms，虽然也是 AR 架构但推理优化极其充分

---

## 七、部署场景推荐

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 云端实时 ASR (高并发) | Qwen3-ASR | RTF=0.04，单卡可服务 24 路实时流 |
| 云端低成本部署 | Paraformer-large | 显存仅 1GB，单卡可部署 10+ 实例 |
| 端侧/边缘部署 | Paraformer-large | 模型小、显存低、NAR 无需逐步生成 |
| 短音频实时 (<3s) | Paraformer-large / Qwen3-ASR | Fun-ASR-Nano 短音频 RTF>1，不可用 |
| 多任务/多语言/可定制 | Fun-ASR-Nano / Qwen3-ASR | LLM 架构支持 prompt 定制、热词、ITN |
| 低延迟要求 (<200ms) | Qwen3-ASR | P50 仅 92ms |

---

## 八、总结

**性能排序**: Qwen3-ASR (RTF=0.04) >> Paraformer-large (RTF=0.18) >> Fun-ASR-Nano (RTF=0.68)

| 结论 | 说明 |
|------|------|
| Qwen3-ASR 综合最优 | 推理最快、延迟最低，但显存占用最大 (4GB)，需微调 |
| Paraformer-large 性价比最高 | 显存仅 1GB，RTF=0.18，适合大规模低成本部署和端侧场景 |
| Fun-ASR-Nano 当前版本有瓶颈 | 固定 1.7s 延迟限制了短音频场景，需推理优化或等待新版本 |

**Fun-ASR-Nano 的优势场景**: 需要 LLM 能力 (prompt 定制、热词、多语言切换) 且音频较长 (>5s) 的场景下，Fun-ASR-Nano 的 RTF (0.2~0.3) 仍然可用，且具备 Paraformer 不具备的灵活性。
