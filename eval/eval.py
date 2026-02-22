"""
Evaluate a model using LM Eval Harness with vLLM.
"""
import argparse
import json
from pathlib import Path
from lm_eval.config.evaluate_config import EvaluatorConfig
import lm_eval
from vllm.sampling_params import StructuredOutputsParams
from utils import generate_datetime_id


TASKS_DIR = Path(__file__).parent.parent / "configs" / "tasks"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "--config", required=True, help="Path to the evaluation configuration YAML file"
    )
    return parser.parse_args()


def run_evaluation(config_path: Path) -> dict:
    config = EvaluatorConfig.from_config(config_path)
    task_manager = config.process_tasks()
    output_dir = RESULTS_DIR / generate_datetime_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert structured_outputs dict to StructuredOutputsParams object
    gen_kwargs = config.gen_kwargs.copy() if config.gen_kwargs else {}
    structured_outputs_config = gen_kwargs.pop("structured_outputs", None)
    
    if structured_outputs_config:
        gen_kwargs["structured_outputs"] = StructuredOutputsParams(
            **structured_outputs_config
        )
    
    results = lm_eval.simple_evaluate(
        model=config.model,
        model_args=config.model_args,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        task_manager=task_manager,
        log_samples=config.log_samples,
        gen_kwargs=gen_kwargs,
        apply_chat_template=config.apply_chat_template,
        system_instruction=config.system_instruction,
    )
    
    if results is not None:
        if hasattr(results, "results"):
            results_dict = results.results
        else:
            results_dict = results
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        if hasattr(results, "samples") and results.samples:
            for task_name, samples in results.samples.items():
                with open(output_dir / f"responses_{task_name}.jsonl", "w") as f:
                    for sample in samples:
                        f.write(json.dumps(sample, default=str) + "\n")
        
        print(f"Results saved to: {output_dir}")
    
    return results


def main():
    args = parse_args()
    run_evaluation(Path(args.config))
    print(f"\nDone! Config: {args.config}")


if __name__ == "__main__":
    main()