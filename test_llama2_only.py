"""
================================================================================
Quick Test Script - Llama-2-7B Profiling Only
================================================================================
Author: Ramesh Fernando
Generated with: Claude Opus 4.5
Description: A minimal test script to evaluate the profiler using only
             the Llama-2-7B model before running the full benchmark suite.
================================================================================
"""

from llm_profiler_awq import AWQProfiler, QuantizationType


def test_llama2_quick():
    """Quick test with reduced iterations"""
    print("\n" + "=" * 60)
    print("QUICK TEST: Llama-2-7B Profiling")
    print("=" * 60)
    
    profiler = AWQProfiler(
        warmup_iterations=2,      # Reduced for faster testing
        profile_iterations=3,     # Reduced for faster testing
        prefer_awq=True,
    )
    
    # Profile only Llama-2-7B with both tasks
    results = profiler.profile_all(
        models=["Llama-2-7B"],
        tasks=["discriminative", "generative"],
    )
    
    profiler.print_summary()
    profiler.save_results("llama2_7b_quick_results.json")
    
    return results


def test_llama2_generative_only():
    """Test only generative task (250:250)"""
    print("\n" + "=" * 60)
    print("TEST: Llama-2-7B Generative Task (250:250)")
    print("=" * 60)
    
    profiler = AWQProfiler(
        warmup_iterations=2,
        profile_iterations=3,
        prefer_awq=True,
    )
    
    # Profile single model with specific settings
    result = profiler.profile_model(
        model_name="Llama-2-7B",
        task_type="generative",
        input_tokens=250,
        output_tokens=250,
        quantization=QuantizationType.AWQ,
    )
    
    if result:
        print("\n" + "-" * 60)
        print("RESULTS: Llama-2-7B Generative (250:250)")
        print("-" * 60)
        print(f"  Quantization:     {result.quantization}")
        print(f"  Total time:       {result.total_time_ms:.2f} ms")
        print(f"  Prefill time:     {result.prefill_time_ms:.2f} ms")
        print(f"  Decode time:      {result.decode_time_ms:.2f} ms")
        print(f"  Avg token latency:{result.avg_token_latency_ms:.2f} ms")
        print(f"  First token:      {result.first_token_latency_ms:.2f} ms")
        print(f"  Tokens/sec:       {result.tokens_per_second:.1f}")
        print(f"  VRAM peak:        {result.memory_peak_mb:.0f} MB")
        print("-" * 60)
        print("\nComponent Breakdown:")
        print(f"  Embedding:        {result.embedding_time_ms:.2f} ms")
        print(f"  Attention ({result.attention_type}):  {result.attention_time_ms:.2f} ms")
        print(f"  FFN:              {result.ffn_time_ms:.2f} ms")
        print(f"  Layer Norm:       {result.layer_norm_time_ms:.2f} ms")
        print(f"  Non-linear:       {result.nonlinear_time_ms:.2f} ms")
        print("-" * 60)
    
    return result


def test_llama2_discriminative_only():
    """Test only discriminative task (MCQ)"""
    print("\n" + "=" * 60)
    print("TEST: Llama-2-7B Discriminative Task (MCQ)")
    print("=" * 60)
    
    profiler = AWQProfiler(
        warmup_iterations=2,
        profile_iterations=3,
        prefer_awq=True,
    )
    
    result = profiler.profile_model(
        model_name="Llama-2-7B",
        task_type="discriminative",
        quantization=QuantizationType.AWQ,
    )
    
    if result:
        print("\n" + "-" * 60)
        print("RESULTS: Llama-2-7B Discriminative (MCQ)")
        print("-" * 60)
        print(f"  Quantization:     {result.quantization}")
        print(f"  Input tokens:     {result.input_tokens}")
        print(f"  Output tokens:    {result.output_tokens}")
        print(f"  Total time:       {result.total_time_ms:.2f} ms")
        print(f"  Prefill time:     {result.prefill_time_ms:.2f} ms")
        print(f"  Decode time:      {result.decode_time_ms:.2f} ms")
        print(f"  Tokens/sec:       {result.tokens_per_second:.1f}")
        print(f"  VRAM peak:        {result.memory_peak_mb:.0f} MB")
        print("-" * 60)
    
    return result


def compare_quantization():
    """Compare AWQ vs BitsAndBytes for Llama-2-7B"""
    print("\n" + "=" * 60)
    print("COMPARISON: AWQ vs BitsAndBytes on Llama-2-7B")
    print("=" * 60)
    
    profiler = AWQProfiler(
        warmup_iterations=2,
        profile_iterations=3,
        prefer_awq=True,
    )
    
    results = profiler.compare_quantization_methods("Llama-2-7B")
    
    return results


def main():
    """Main entry point - choose your test"""
    import sys
    
    print("\n" + "=" * 60)
    print("LLM INFERENCE PROFILER - Llama-2-7B Test Suite")
    print("=" * 60)
    print("\nAvailable tests:")
    print("  1. Quick test (both tasks)")
    print("  2. Generative only (250:250)")
    print("  3. Discriminative only (MCQ)")
    print("  4. Compare quantization methods")
    print("  5. Run all tests")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        test_llama2_quick()
    elif choice == "2":
        test_llama2_generative_only()
    elif choice == "3":
        test_llama2_discriminative_only()
    elif choice == "4":
        compare_quantization()
    elif choice == "5":
        test_llama2_generative_only()
        test_llama2_discriminative_only()
        compare_quantization()
    else:
        print("Invalid choice. Running quick test...")
        test_llama2_quick()


if __name__ == "__main__":
    main()
