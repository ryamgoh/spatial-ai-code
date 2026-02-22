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
# Base model only
# uv run inference --base-model Qwen/Qwen2.5-7B-Instruct --prompt "What is 2+2?"
# Base model + LoRA adapter
# uv run inference --base-model Qwen/Qwen3-7B --adapter ./outputs/qwen3-7b-lora-finetuned --prompt "test"
# Interactive with adapter
# uv run inference --base-model Qwen/Qwen3-7B --adapter ./outputs/adapter --interactive
# With RAG
# uv run inference --base-model Qwen/Qwen2.5-7B-Instruct --prompt "What is X?" --rag --corpus ./datasets/

# ===========================================
# Evaluation runs
# ===========================================
# Evaluate base model on tasks
cd eval
uv run eval.py --base-model Qwen/Qwen2.5-0.5B-Instruct --tasks sciq_shuffled,sciq

# Evaluate another model
# uv run eval --base-model Qwen/Qwen3-7B --tasks task_rag,task_bool

# Evaluate fine-tuned model with LoRA adapter
# uv run eval --base-model Qwen/Qwen3-7B --adapter ./outputs/qwen3-7b-lora-finetuned --tasks task_rag,task_bool

# Evaluate with thinking mode enabled
# uv run eval --base-model Qwen/Qwen3-7B --tasks task_rag,task_bool --thinking

# Evaluate with few-shot
# uv run eval --base-model Qwen/Qwen2.5-7B-Instruct --tasks task_rag --num-fewshot 2

# Evaluate with RAG enabled
# uv run eval --base-model Qwen/Qwen2.5-7B-Instruct --tasks task_rag_augmented --rag-enabled

# Evaluate with RAG and force rebuild index
# uv run eval --base-model Qwen/Qwen2.5-7B-Instruct --tasks task_rag_augmented --rag-enabled --rag-force-rebuild

# ===========================================
# Analysis
# ===========================================
# Analyze the last 2 runs
# cd ..
uv run analyze.py --last 1

# Or analyze specific runs
# uv run analyze --runs run_20260221_103000,run_20260221_110000

# ===========================================
# Done
# ===========================================
echo "========================================"
echo "Orchestration complete!"
echo "========================================"
