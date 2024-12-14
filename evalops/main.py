import json
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from scipy import stats


class ModelInterface(Protocol):
    """Protocol defining the required interface for model classes."""

    def run(self, task: str) -> str:
        """
        Run the model on a given task.

        Args:
            task: The input task/question string

        Returns:
            The model's response as a string
        """
        ...


class EvalResult(BaseModel):
    """
    Pydantic model to store evaluation results for a single model run.

    Attributes:
        mean_score (float): Average score across all questions
        sem (float): Standard error of the mean
        ci_lower (float): Lower bound of 95% confidence interval
        ci_upper (float): Upper bound of 95% confidence interval
        raw_scores (List[float]): Individual question scores
        metadata (Dict): Additional metadata about the evaluation
    """

    mean_score: float
    sem: float
    ci_lower: float
    ci_upper: float
    raw_scores: List[float]
    metadata: Dict[str, Any]


class StatisticalModelEvaluator:
    """
    A statistical approach to model evaluations implementing the methodology
    described in the paper "Adding Error Bars to Evals".

    This class provides tools for:
    - Computing robust statistical metrics for model evaluation
    - Handling clustered questions
    - Implementing variance reduction techniques
    - Performing power analysis
    - Conducting paired difference tests

    Args:
        cache_dir (Optional[str]): Directory to cache evaluation results
        log_level (str): Logging level (default: "INFO")
        random_seed (Optional[int]): Random seed for reproducibility
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        log_level: str = "INFO",
        random_seed: Optional[int] = None,
    ):
        logger.add(
            lambda msg: print(msg),
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        logger.debug("Initializing StatisticalModelEvaluator")

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory set to {self.cache_dir}")
        else:
            self.cache_dir = None
            logger.debug("No cache directory specified")

        if random_seed is not None:
            np.random.seed(random_seed)
            logger.debug(f"Random seed set to {random_seed}")

        logger.info("Initialized StatisticalModelEvaluator")

    def _calculate_score(
        self,
        prediction: str,
        correct_answer: str,
    ) -> float:
        logger.debug(
            f"Calculating score for prediction: '{prediction}' vs answer: '{correct_answer}'"
        )

        prediction = prediction.strip().lower()
        correct_answer = correct_answer.strip().lower()

        # Check if correct answer is in prediction
        if correct_answer in prediction:
            logger.debug("Exact match found")
            return 1.0

        # Fallback to sequence matching
        similarity = SequenceMatcher(
            None,
            prediction,
            correct_answer,
        ).ratio()

        logger.debug(f"Sequence similarity score: {similarity}")
        return similarity if similarity > 0.8 else 0.0

    def evaluate_model(
        self,
        model: ModelInterface,
        questions: List[str],
        correct_answers: List[str],
        cluster_ids: Optional[List[str]] = None,
        num_samples: int = 1,
        batch_size: int = 32,
        cache_key: Optional[str] = None,
    ) -> EvalResult:
        """
        Evaluate a model on a set of questions with statistical analysis.

        Args:
            model: Model instance with a .run(task: str) -> str method
            questions: List of question strings
            correct_answers: List of correct answer strings
            cluster_ids: Optional list of cluster identifiers for questions
            num_samples: Number of times to sample each question
            batch_size: Batch size for parallel processing
            cache_key: Optional key for caching results

        Returns:
            EvalResult object containing statistical metrics

        Example:
            ```python
            class MyModel:
                def run(self, task: str) -> str:
                    return "model response"

            model = MyModel()
            evaluator = StatisticalModelEvaluator()
            result = evaluator.evaluate_model(
                model=model,
                questions=["What is 2+2?"],
                correct_answers=["4"]
            )
            ```
        """
        logger.debug("Starting model evaluation")
        start_time = time.time()

        # Check if cached results exist
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                logger.info(f"Loading cached results for {cache_key}")
                with open(cache_path) as f:
                    cached_data = json.load(f)
                logger.debug("Successfully loaded cached results")
                return EvalResult(**cached_data)

        # Validate inputs
        logger.debug("Validating inputs")
        assert len(questions) == len(
            correct_answers
        ), "Questions and answers must have same length"
        if cluster_ids:
            assert len(cluster_ids) == len(
                questions
            ), "Cluster IDs must match question length"

        logger.info(
            f"Starting evaluation of {len(questions)} questions with {num_samples} samples each"
        )

        # Run model predictions in parallel batches
        all_scores = []
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i : i + batch_size]
                batch_answers = correct_answers[i : i + batch_size]

                logger.debug(
                    f"Processing batch {i//batch_size + 1} with {len(batch_questions)} questions"
                )

                # Create partial function for each question/answer pair
                tasks = [
                    partial(
                        self._evaluate_single_question,
                        model,
                        q,
                        a,
                        num_samples,
                    )
                    for q, a in zip(batch_questions, batch_answers)
                ]

                # Execute batch
                batch_scores = list(
                    executor.map(lambda f: f(), tasks)
                )
                all_scores.extend(batch_scores)
                logger.debug(f"Batch {i//batch_size + 1} complete")

        # Calculate statistics
        logger.debug("Calculating statistics")
        scores_array = np.array(all_scores)
        mean_score = np.mean(scores_array)

        if cluster_ids:
            logger.debug("Calculating clustered standard error")
            sem = self._calculate_clustered_sem(
                scores_array, cluster_ids
            )
        else:
            logger.debug("Calculating regular standard error")
            sem = stats.sem(scores_array)

        # Calculate 95% confidence interval
        ci_lower, ci_upper = stats.norm.interval(
            0.95, loc=mean_score, scale=sem
        )
        logger.debug(
            f"Confidence interval: [{ci_lower:.3f}, {ci_upper:.3f}]"
        )

        # Create result object
        result = EvalResult(
            mean_score=float(mean_score),
            sem=float(sem),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            raw_scores=all_scores,
            metadata={
                "num_questions": len(questions),
                "num_samples": num_samples,
                "has_clusters": cluster_ids is not None,
                "evaluation_time": time.time() - start_time,
            },
        )

        # Cache results if requested
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            logger.debug(f"Caching results to {cache_path}")
            with open(cache_path, "w") as f:
                json.dump(result.__dict__, f)
            logger.info(f"Cached results to {cache_path}")

        logger.info(
            f"Evaluation complete. Mean score: {mean_score:.3f} ± {sem:.3f} (95% CI)"
        )

        # json_output = result.model_dump()
        return result.model_dump_json(indent=4)

    def compare_models(
        self, results_a: EvalResult, results_b: EvalResult
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between two model evaluation results.

        Args:
            results_a: EvalResult for first model
            results_b: EvalResult for second model

        Returns:
            Dictionary containing comparison metrics
        """
        logger.debug("Starting model comparison")

        # Calculate mean difference
        mean_diff = results_a.mean_score - results_b.mean_score
        logger.debug(f"Mean difference: {mean_diff:.3f}")

        # Calculate correlation between scores
        correlation = np.corrcoef(
            results_a.raw_scores, results_b.raw_scores
        )[0, 1]
        logger.debug(f"Score correlation: {correlation:.3f}")

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(
            results_a.raw_scores, results_b.raw_scores
        )
        logger.debug(
            f"T-test results: t={t_stat:.3f}, p={p_value:.3f}"
        )

        return {
            "mean_difference": mean_diff,
            "correlation": correlation,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_difference": p_value < 0.05,
        }

    def calculate_required_samples(
        self,
        effect_size: float,
        baseline_variance: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """
        Calculate required number of samples for desired statistical power.

        Args:
            effect_size: Minimum difference to detect
            baseline_variance: Estimated variance in scores
            power: Desired statistical power (default: 0.8)
            alpha: Significance level (default: 0.05)

        Returns:
            Required number of samples
        """
        logger.debug(
            f"Calculating required samples for effect size {effect_size}"
        )

        # Calculate required sample size using power analysis
        required_n = stats.tt_ind_solve_power(
            effect_size=effect_size / np.sqrt(baseline_variance),
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
        logger.info(
            f"Required number of samples: {int(np.ceil(required_n))}"
        )
        return int(np.ceil(required_n))

    def _evaluate_single_question(
        self,
        model: ModelInterface,
        question: str,
        correct_answer: str,
        num_samples: int,
    ) -> float:
        """
        Evaluate a single question multiple times and return average score.

        Args:
            model: Model instance with .run() method
            question: Question string
            correct_answer: Correct answer string
            num_samples: Number of samples to take

        Returns:
            Average score for the question
        """
        logger.debug(f"Evaluating question: '{question}'")
        scores = []
        for i in range(num_samples):
            try:
                prediction = model.run(question)
                score = self._calculate_score(
                    prediction, correct_answer
                )
                scores.append(score)
                logger.debug(
                    f"Sample {i+1}/{num_samples} score: {score:.3f}"
                )
            except Exception as e:
                logger.error(f"Error evaluating question: {str(e)}")
                scores.append(0.0)

        avg_score = np.mean(scores)
        logger.debug(f"Average score for question: {avg_score:.3f}")
        return avg_score

    def _calculate_clustered_sem(
        self, scores: np.ndarray, cluster_ids: List[str]
    ) -> float:
        """
        Calculate clustered standard error of the mean.

        Args:
            scores: Array of scores
            cluster_ids: List of cluster identifiers

        Returns:
            Clustered standard error
        """
        logger.debug("Calculating clustered standard error")
        df = pd.DataFrame({"score": scores, "cluster": cluster_ids})

        # Calculate cluster means
        cluster_means = df.groupby("cluster")["score"].mean()
        logger.debug(f"Number of clusters: {len(cluster_means)}")

        # Calculate clustered standard error
        n_clusters = len(cluster_means)
        cluster_variance = cluster_means.var()
        sem = np.sqrt(cluster_variance / n_clusters)
        logger.debug(f"Clustered SEM: {sem:.3f}")
        return sem
