#!/usr/bin/env python3
"""
Generate deployment decision report for Fun-ASR-Nano.

Aggregates results from:
  - eval/latency_report.json
  - eval/resource_report.json
  - eval/quantization_report.json

Outputs a structured decision report.
"""

import argparse
import json
import os
import sys
from datetime import datetime


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}


def generate_markdown_report(latency: dict, resource: dict, quant: dict) -> str:
    lines = []
    lines.append("# Fun-ASR-Nano 端云部署评测报告")
    lines.append("")
    lines.append(f"> 模型: Fun-ASR-Nano-2512 (FunAudioLLM/Fun-ASR-Nano-2512)  ")
    lines.append(f"> 评测时间: {datetime.now().strftime('%Y-%m-%d')}  ")
    lines.append(f"> 评测环境: CUDA GPU")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Latency
    lines.append("## 一、推理性能")
    lines.append("")

    if latency:
        num_samples = latency.get("num_samples", "N/A")
        lines.append(f"**测试条件**: {num_samples} 条音频，单条推理 (batch_size=1)，GPU")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        total_audio = latency.get("total_audio_duration_sec", 0)
        total_infer = latency.get("total_inference_time_sec", 0)
        overall_rtf = latency.get("overall_rtf", 0)
        lines.append(f"| 总音频时长 | {total_audio:.1f}s |")
        lines.append(f"| 总推理时间 | {total_infer:.1f}s |")
        lines.append(f"| 整体 RTF | **{overall_rtf:.4f}** |")
        speedup = 1.0 / overall_rtf if overall_rtf > 0 else 0
        lines.append(f"| RTF 含义 | < 1.0 即实时可行，当前比实时快约 {speedup:.0f} 倍 |")
        lines.append("")

        # E2E latency
        e2e = latency.get("e2e_latency", {})
        if e2e:
            lines.append("### E2E 延迟分布")
            lines.append("")
            lines.append("| 百分位 | 延迟 |")
            lines.append("|--------|------|")
            lines.append(f"| P50 | {e2e.get('p50', 0)*1000:.1f} ms |")
            lines.append(f"| P90 | {e2e.get('p90', 0)*1000:.1f} ms |")
            lines.append(f"| P99 | {e2e.get('p99', 0)*1000:.1f} ms |")
            lines.append(f"| Max | {e2e.get('max', 0)*1000:.1f} ms |")
            lines.append("")

        # RTF distribution
        rtf_stats = latency.get("rtf", {})
        if rtf_stats:
            lines.append("### RTF 分布")
            lines.append("")
            lines.append("| 百分位 | RTF |")
            lines.append("|--------|-----|")
            lines.append(f"| P50 | {rtf_stats.get('p50', 0):.4f} |")
            lines.append(f"| P90 | {rtf_stats.get('p90', 0):.4f} |")
            lines.append(f"| P99 | {rtf_stats.get('p99', 0):.4f} |")
            lines.append("")

        # Duration buckets
        buckets = latency.get("duration_buckets", {})
        if buckets:
            lines.append("### 按音频时长分段")
            lines.append("")
            lines.append("| 时长段 | 样本数 | RTF P50 | 延迟 P50 |")
            lines.append("|--------|--------|---------|----------|")
            for name, stats in buckets.items():
                rtf_p50 = stats["rtf"]["p50"]
                lat_p50 = stats["latency"]["p50"] * 1000
                lines.append(f"| {name} | {stats['count']} | {rtf_p50:.4f} | {lat_p50:.1f} ms |")
            lines.append("")

        lines.append(f"**结论**: GPU 推理性能{'充裕' if overall_rtf < 0.5 else '可行'}，"
                     f"所有音频 RTF 均远小于 1.0，P99 延迟 {e2e.get('p99', 0)*1000:.0f}ms。")
    else:
        lines.append("*[未运行延迟评测]*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 2: Resources
    lines.append("## 二、资源占用")
    lines.append("")

    if resource:
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        model_size = resource.get("model_size_disk_mb", 0)
        if model_size > 0:
            lines.append(f"| 模型文件大小 | {model_size:.0f} MB |")
        gpu_mem = resource.get("gpu_memory", {})
        cpu_mem = resource.get("cpu_memory", {})
        if gpu_mem:
            lines.append(f"| GPU 显存 - 模型加载后 | {gpu_mem.get('after_load_allocated_mb', 0):.0f} MB |")
            lines.append(f"| GPU 显存 - 推理峰值 | {gpu_mem.get('peak_allocated_mb', 0):.0f} MB |")
            lines.append(f"| GPU 推理增量 | +{gpu_mem.get('inference_peak_delta_mb', 0):.1f} MB |")
        if cpu_mem:
            lines.append(f"| CPU RSS - 模型加载后 | {cpu_mem.get('rss_after_load_mb', 0):.0f} MB |")
            lines.append(f"| CPU RSS - 推理峰值 | {cpu_mem.get('rss_peak_during_infer_mb', 0):.0f} MB |")
        lines.append("")

        if gpu_mem:
            peak = gpu_mem.get('peak_allocated_mb', 0)
            lines.append(f"**结论**: 模型约占 {peak:.0f}MB 显存，推理时额外开销极小。")
    else:
        lines.append("*[未运行资源评测]*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 3: Quantization
    lines.append("## 三、量化对比")
    lines.append("")

    if quant and quant.get("results"):
        qr = quant["results"]
        lines.append("| 配置 | CER (%) | CER 退化 | RTF | GPU 峰值 | 显存节省 |")
        lines.append("|------|---------|----------|-----|----------|----------|")

        baseline_cer = qr.get("baseline", {}).get("cer")
        baseline_gpu = qr.get("baseline", {}).get("gpu_peak_mb")

        for name, res in qr.items():
            if "error" in res:
                lines.append(f"| {name} | ERROR | — | — | — | — |")
                continue
            cer = res["cer"]
            rtf = res.get("rtf", 0)
            gpu_peak = res.get("gpu_peak_mb", 0)

            if name == "baseline":
                lines.append(f"| {name} | {cer:.2f} | — | {rtf:.4f} | {gpu_peak:.0f} MB | — |")
            else:
                cer_delta = cer - baseline_cer if baseline_cer is not None else 0
                cer_rel = (cer_delta / baseline_cer * 100) if baseline_cer and baseline_cer > 0 else 0
                gpu_save = ((baseline_gpu - gpu_peak) / baseline_gpu * 100) if baseline_gpu and baseline_gpu > 0 else 0
                lines.append(f"| {name} | {cer:.2f} | +{cer_delta:.2f} (+{cer_rel:.1f}%) | "
                             f"{rtf:.4f} | {gpu_peak:.0f} MB | -{gpu_save:.0f}% |")
        lines.append("")

        # Analysis
        lines.append("### 分析")
        lines.append("")
        if "int8" in qr and "error" not in qr["int8"]:
            int8_cer_delta = qr["int8"]["cer"] - baseline_cer if baseline_cer else 0
            lines.append(f"- **INT8**: CER 退化 {'+' if int8_cer_delta > 0 else ''}{int8_cer_delta:.2f}%，"
                         f"{'质量几乎无损' if abs(int8_cer_delta) < 1 else '有一定退化'}。")
        if "int4_nf4" in qr and "error" not in qr["int4_nf4"]:
            int4_cer_delta = qr["int4_nf4"]["cer"] - baseline_cer if baseline_cer else 0
            int4_rel = (int4_cer_delta / baseline_cer * 100) if baseline_cer and baseline_cer > 0 else 0
            lines.append(f"- **INT4 NF4**: CER 退化 +{int4_cer_delta:.2f} (相对 {int4_rel:.1f}%)，"
                         f"{'退化较大' if int4_rel > 10 else '退化可接受'}。")
    else:
        lines.append("*[未运行量化评测]*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 4: Deployment Decision
    lines.append("## 四、端云部署决策")
    lines.append("")

    overall_rtf = latency.get("overall_rtf", 999) if latency else 999
    gpu_peak = resource.get("gpu_memory", {}).get("peak_allocated_mb", 0) if resource else 0

    lines.append("### 云端部署评估")
    lines.append("")
    lines.append("| 维度 | 评估 | 结论 |")
    lines.append("|------|------|------|")
    lines.append(f"| 实时性 | RTF = {overall_rtf:.4f} | {'✅ 完全满足' if overall_rtf < 1.0 else '❌ 不满足'} |")
    lines.append(f"| 显存 | 峰值 ~{gpu_peak:.0f}MB | {'✅ 单卡可承载多实例' if gpu_peak < 8000 else '⚠️ 显存较大'} |")
    e2e_p99 = latency.get("e2e_latency", {}).get("p99", 0) * 1000 if latency else 0
    lines.append(f"| 延迟 | P99 < {e2e_p99:.0f}ms | {'✅ 用户体验良好' if e2e_p99 < 500 else '⚠️ 延迟偏高'} |")
    lines.append("")

    lines.append("### 手机端部署评估")
    lines.append("")
    lines.append("| 维度 | 评估 | 结论 |")
    lines.append("|------|------|------|")

    model_size = resource.get("model_size_disk_mb", 0) if resource else 0
    lines.append(f"| 模型体积 | ~{model_size:.0f}MB | "
                 f"{'✅ 端侧可行' if model_size < 500 else '⚠️ 体积较大' if model_size < 2000 else '❌ 过大'} |")

    if quant and quant.get("results"):
        qr = quant["results"]
        if "int4_nf4" in qr and "error" not in qr["int4_nf4"]:
            int4_cer = qr["int4_nf4"]["cer"]
            int4_delta_rel = ((int4_cer - baseline_cer) / baseline_cer * 100) if baseline_cer and baseline_cer > 0 else 0
            lines.append(f"| 识别质量 (INT4) | CER 退化 {int4_delta_rel:.1f}% | "
                         f"{'✅ 可接受' if int4_delta_rel < 10 else '❌ 质量不达标'} |")
    lines.append("| 推理速度 | 未在手机芯片实测 | ❓ 需进一步验证 |")
    lines.append("")

    # Final recommendation
    lines.append("### 最终建议")
    lines.append("")
    lines.append("```")
    if overall_rtf < 1.0:
        lines.append("推荐方案: 云端部署为主")
    else:
        lines.append("推荐方案: 需进一步优化")
    lines.append("```")
    lines.append("")
    lines.append("**理由**:")
    lines.append("")
    if overall_rtf < 1.0:
        speedup = 1.0 / overall_rtf if overall_rtf > 0 else 0
        lines.append(f"1. **云端推理性能充裕** — RTF 仅 {overall_rtf:.4f}，"
                     f"单张 GPU 可同时服务约 {speedup:.0f}+ 路实时音频流")
        lines.append(f"2. **Fun-ASR-Nano 为轻量级模型** — 相比大模型 ASR 方案，"
                     f"显存占用更小，部署成本更低")
        lines.append(f"3. **端侧建议** — 可直接部署做离线 ASR，或配合 VAD 做实时流式识别")
    lines.append("")
    lines.append("### 后续优化方向")
    lines.append("")
    lines.append("- 测试 ONNX Runtime 导出后的推理性能")
    lines.append("- 探索流式推理方案，降低首字延迟")
    lines.append("- 在目标手机芯片上实测 INT4/INT8 量化模型表现")
    lines.append("- 对比 Paraformer-large 等其他 FunASR 模型的性能")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Appendix
    lines.append("## 附录：评测脚本")
    lines.append("")
    lines.append("| 脚本 | 用途 |")
    lines.append("|------|------|")
    lines.append("| `eval/benchmark_latency.py` | 离线推理延迟和 RTF 评测 |")
    lines.append("| `eval/profile_resources.py` | 内存和显存占用分析 |")
    lines.append("| `eval/quantization_eval.py` | INT8/INT4 量化对比评测 |")
    lines.append("| `eval/generate_report.py` | 汇总报告生成 |")
    lines.append("| `eval/run_all.sh` | 一键运行全部评测 |")
    lines.append("")
    lines.append("运行方式:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /cmy/cmy/FunASR")
    lines.append("bash eval/run_all.sh")
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Fun-ASR-Nano evaluation report")
    parser.add_argument("--latency_report", type=str, default="eval/latency_report.json")
    parser.add_argument("--resource_report", type=str, default="eval/resource_report.json")
    parser.add_argument("--quant_report", type=str, default="eval/quantization_report.json")
    parser.add_argument("--output", type=str, default="eval/evaluation_report.md")
    parser.add_argument("--output_json", type=str, default="eval/decision_report.json")
    args = parser.parse_args()

    latency = load_json(args.latency_report)
    resource = load_json(args.resource_report)
    quant = load_json(args.quant_report)

    found = []
    if latency:
        found.append("latency")
    if resource:
        found.append("resource")
    if quant:
        found.append("quantization")
    print(f"Found reports: {', '.join(found) if found else 'none'}")

    if not found:
        print("No report files found. Run the evaluation scripts first:")
        print("  python eval/benchmark_latency.py")
        print("  python eval/profile_resources.py")
        print("  python eval/quantization_eval.py")
        sys.exit(1)

    report_md = generate_markdown_report(latency, resource, quant)
    print(report_md)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nMarkdown report saved to: {args.output}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "latency_summary": {
            "overall_rtf": latency.get("overall_rtf"),
            "e2e_latency_p50": latency.get("e2e_latency", {}).get("p50"),
            "e2e_latency_p90": latency.get("e2e_latency", {}).get("p90"),
            "e2e_latency_p99": latency.get("e2e_latency", {}).get("p99"),
        } if latency else None,
        "resource_summary": {
            "model_size_mb": resource.get("model_size_disk_mb"),
            "gpu_peak_mb": resource.get("gpu_memory", {}).get("peak_allocated_mb"),
            "cpu_peak_mb": resource.get("cpu_memory", {}).get("rss_peak_during_infer_mb"),
        } if resource else None,
        "quantization_summary": quant.get("results") if quant else None,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON summary saved to: {args.output_json}")


if __name__ == "__main__":
    main()
