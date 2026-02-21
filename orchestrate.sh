#!/bin/bash
# orchestrate.sh - Define your experiment pipeline
# 
# Edit this file to configure your runs, then execute:
#   chmod +x orchestrate.sh && ./orchestrate.sh
set -e  # Exit on error
echo "========================================"
echo "Experiment Orchestration"
echo "========================================"
# ===========================================
# OPTIONAL: Fine-tune models
# ===========================================
# Fine-tune Qwen3-7B with LoRA
# uv run finetune qwen3-7b-lora
# Fine-tune Qwen2.5-7B
# uv run finetune qwen2.5-7b
# Fine-tune Qwen3-14B
# uv run finetune qwen3-14b
# Resume from checkpoint
# uv run finetune qwen3-7b-lora --resume
# ===========================================
# OPTIONAL: Test inference on fine-tuned model
# ===========================================
# uv run inference --model ./outputs/qwen3-7b-lora-finetuned --prompt "What is 2+2?"
# uv run inference --model ./outputs/qwen3-7b-lora-finetuned --interactive
# ===========================================
# Evaluation runs
# ===========================================
# Evaluate base model on tasks
uv run eval --model Qwen/Qwen2.5-7B-Instruct --tasks task_rag,task_bool
# Evaluate another model
uv run eval --model Qwen/Qwen3-7B --tasks task_rag,task_bool
# Evaluate fine-tuned model (uncomment after finetuning)
# uv run eval --model ./outputs/qwen3-7b-lora-finetuned --tasks task_rag,task_bool
# Evaluate with thinking mode enabled
# uv run eval --model Qwen/Qwen3-7B --tasks task_rag,task_bool --thinking
# Evaluate with few-shot
# uv run eval --model Qwen/Qwen2.5-7B-Instruct --tasks task_rag --num-fewshot 2
# ===========================================
# Analysis
# ===========================================
# Analyze the last 2 runs
uv run analyze --last 2
# Or analyze specific runs
# uv run analyze --runs run_20260221_103000,run_20260221_110000
# ===========================================
# Done
# ===========================================
echo "========================================"
echo "Orchestration complete!"
echo "========================================"