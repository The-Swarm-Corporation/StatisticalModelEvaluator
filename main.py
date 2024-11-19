from typing import List, Optional, Tuple, Dict, Any, Protocol
from dataclasses import dataclass
import numpy as np
from scipy import stats
import pandas as pd
from loguru import logger
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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

@dataclass
class EvalResult:
    """
    Data class to store evaluation results for a single model run.
    
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
        random_seed: Optional[int] = None
    ):
        # Initialize logging
        logger.remove()
        logger.add(
            lambda msg: print(msg),
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
            
        if random_seed is not None:
            np.random.seed(random_seed)
            
        logger.info("Initialized StatisticalModelEvaluator")

    def evaluate_model(
        self,
        model: ModelInterface,
        questions: List[str],
        correct_answers: List[str],
        cluster_ids: Optional[List[str]] = None,
        num_samples: int = 1,
        batch_size: int = 32,
        cache_key: Optional[str] = None
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
        start_time = time.time()
        
        # Check if cached results exist
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                logger.info(f"Loading cached results for {cache_key}")
                with open(cache_path) as f:
                    cached_data = json.load(f)
                return EvalResult(**cached_data)
        
        # Validate inputs
        assert len(questions) == len(correct_answers), "Questions and answers must have same length"
        if cluster_ids:
            assert len(cluster_ids) == len(questions), "Cluster IDs must match question length"
            
        logger.info(f"Starting evaluation of {len(questions)} questions with {num_samples} samples each")
        
        # Run model predictions in parallel batches
        all_scores = []
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                batch_answers = correct_answers[i:i + batch_size]
                
                # Create partial function for each question/answer pair
                tasks = [
                    partial(self._evaluate_single_question, model, q, a, num_samples)
                    for q, a in zip(batch_questions, batch_answers)
                ]
                
                # Execute batch
                batch_scores = list(executor.map(lambda f: f(), tasks))
                all_scores.extend(batch_scores)
        
        # Calculate statistics
        scores_array = np.array(all_scores)
        mean_score = np.mean(scores_array)
        
        if cluster_ids:
            # Calculate clustered standard error
            sem = self._calculate_clustered_sem(scores_array, cluster_ids)
        else:
            # Calculate regular standard error
            sem = stats.sem(scores_array)
            
        # Calculate 95% confidence interval
        ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean_score, scale=sem)
        
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
                "evaluation_time": time.time() - start_time
            }
        )
        
        # Cache results if requested
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            with open(cache_path, 'w') as f:
                json.dump(result.__dict__, f)
            logger.info(f"Cached results to {cache_path}")
            
        logger.info(f"Evaluation complete. Mean score: {mean_score:.3f} ± {sem:.3f} (95% CI)")
        return result

    def compare_models(
        self,
        results_a: EvalResult,
        results_b: EvalResult
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between two model evaluation results.
        
        Args:
            results_a: EvalResult for first model
            results_b: EvalResult for second model
            
        Returns:
            Dictionary containing comparison metrics
        """
        # Calculate mean difference
        mean_diff = results_a.mean_score - results_b.mean_score
        
        # Calculate correlation between scores
        correlation = np.corrcoef(results_a.raw_scores, results_b.raw_scores)[0, 1]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(results_a.raw_scores, results_b.raw_scores)
        
        return {
            "mean_difference": mean_diff,
            "correlation": correlation,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_difference": p_value < 0.05
        }

    def calculate_required_samples(
        self,
        effect_size: float,
        baseline_variance: float,
        power: float = 0.8,
        alpha: float = 0.05
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
        # Calculate required sample size using power analysis
        required_n = stats.tt_ind_solve_power(
            effect_size=effect_size / np.sqrt(baseline_variance),
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='two-sided'
        )
        return int(np.ceil(required_n))

    def _evaluate_single_question(
        self,
        model: ModelInterface,
        question: str,
        correct_answer: str,
        num_samples: int
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
        scores = []
        for _ in range(num_samples):
            try:
                prediction = model.run(question)
                score = self._calculate_score(prediction, correct_answer)
                scores.append(score)
            except Exception as e:
                logger.error(f"Error evaluating question: {str(e)}")
                scores.append(0.0)
        return np.mean(scores)

    def _calculate_clustered_sem(
        self,
        scores: np.ndarray,
        cluster_ids: List[str]
    ) -> float:
        """
        Calculate clustered standard error of the mean.
        
        Args:
            scores: Array of scores
            cluster_ids: List of cluster identifiers
            
        Returns:
            Clustered standard error
        """
        df = pd.DataFrame({
            'score': scores,
            'cluster': cluster_ids
        })
        
        # Calculate cluster means
        cluster_means = df.groupby('cluster')['score'].mean()
        
        # Calculate clustered standard error
        n_clusters = len(cluster_means)
        cluster_variance = cluster_means.var()
        return np.sqrt(cluster_variance / n_clusters)

    @staticmethod
    def _calculate_score(prediction: str, correct_answer: str) -> float:
        """
        Calculate score for a single prediction.
        Override this method for custom scoring logic.
        
        Args:
            prediction: Model's prediction
            correct_answer: Correct answer
            
        Returns:
            Score between 0 and 1
        """
        return float(prediction.strip().lower() == correct_answer.strip().lower())