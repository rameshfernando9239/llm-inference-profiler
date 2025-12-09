# LLM Inference Profiler

A comprehensive profiling framework for evaluating inference latency distribution across Large Language Models (LLMs) for edge case scenarios.

## Supported Models

| Model | Attention Type | FP16 VRAM | 4-bit VRAM |
|-------|---------------|-----------|------------|
| OPT-1.3B | MHA | ~2.6 GB | ~1.5 GB |
| Phi-2B | MHA | ~4.0 GB | ~2.0 GB |
| Llama-2-7B | GQA | ~14 GB | ~4.5 GB |
| Llama-2-13B | GQA | ~26 GB | ~8.5 GB |
| Llama-3-8B | GQA | ~16 GB | ~5.5 GB |

## Features

- **Component-level Profiling**: Separate timing for embeddings, attention, FFN, non-linear activations, layer norms
- **Attention Mechanism Support**: Multi-Head Attention (MHA) and Grouped Query Attention (GQA)
- **Task Types**: Discriminative (MCQ) and Generative (250:250 input:output tokens)
- **Quantization Support**: AWQ and BitsAndBytes 4-bit quantization
- **GPU Profiling**: CUDA event-based timing for accurate measurements
- **Memory Tracking**: Reports allocated and peak GPU memory usage

## Installation

```bash
# Create virtual environment
python -m venv llm_profiler
source llm_profiler/bin/activate  # Linux/Mac
# or: llm_profiler\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Optional: Install optimized AWQ kernels
pip install autoawq-kernels
```

## Usage

### Basic Profiling (All Models)

```bash
python llm_profiler_awq.py
```

### Profile Specific Models

```python
from llm_profiler_awq import AWQProfiler

profiler = AWQProfiler(
    warmup_iterations=3,
    profile_iterations=5,
    prefer_awq=True,
)

# Profile specific models
results = profiler.profile_all(
    models=["OPT-1.3B", "Llama-2-7B"],
    tasks=["discriminative", "generative"],
)

profiler.print_summary()
profiler.save_results()
```

### Compare Quantization Methods

```python
profiler.compare_quantization_methods("Llama-2-7B")
```

### Detailed Attention Profiling

```bash
python detailed_attention_profiler.py
```

### Visualize Results

```bash
python visualize_profiling_results.py
```

## Output

The profiler generates:
- Console summary table with timing breakdowns
- JSON file with detailed results (`awq_profiling_results.json`)
- Visualization plots (when using visualization script)

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM (with 4-bit quantization)
- **Recommended**: NVIDIA RTX 3080/3090 or better
- **Optimal**: NVIDIA A100/H100 for FP16 profiling of larger models

## License

MIT License