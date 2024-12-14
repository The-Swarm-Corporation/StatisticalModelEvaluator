from evalops.main import StatisticalModelEvaluator
from typing import List, Any
import loguru


def eval(
    questions: List[str],
    answers: List[str],
    agent: Any,
    samples: int = 5,
    batch_size: int = 32,
    *args,
    **kwargs,
):
    try:
        evaluator = StatisticalModelEvaluator(
            cache_dir="./eval_cache"
        )
        result = evaluator.evaluate_model(
            model=agent,
            questions=questions,
            correct_answers=answers,
            num_samples=samples,
            batch_size=batch_size,
            *args,
            **kwargs,
        )
        loguru.logger.info("Model evaluation successful.")
        return result
    except Exception as e:
        loguru.logger.error(f"Model evaluation failed: {e}")
        return {"error": str(e)}
