import json
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from loguru import logger
from pydantic import BaseModel
from scipy import stats

from evalops.function_eval import FunctionCallEvaluator
from evalops.main import StatisticalModelEvaluator


class ModelInterface(Protocol):
    """Protocol defining the required interface for model classes."""

    def run(self, task: str, img: str = None) -> str:
        """Run the model on a given task."""
        ...


class EvalResult(BaseModel):
    """Stores evaluation results for a single model run."""

    mean_score: float
    sem: float
    ci_lower: float
    ci_upper: float
    raw_scores: List[float]
    metadata: Dict[str, Any]
    function_call_results: Optional[Dict[str, Any]] = None
    sentiment_score: Optional[float] = None


class FunctionCallResult(BaseModel):
    """Stores the evaluation results for function calling tests."""

    schema_valid: bool
    execution_valid: bool
    schema_errors: List[str]
    execution_errors: List[str]
    matching_score: float
    metadata: Dict[str, Any]


class IntegratedModelEvaluator:
    """
    Enhanced model evaluator that combines statistical evaluation,
    function calling assessment, and sentiment analysis.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        log_level: str = "INFO",
        random_seed: Optional[int] = None,
    ):
        # Initialize base statistical evaluator
        self.statistical_evaluator = StatisticalModelEvaluator(
            cache_dir=cache_dir,
            log_level=log_level,
            random_seed=random_seed,
        )

        # Initialize function call evaluator
        self.function_evaluator = FunctionCallEvaluator(
            self.statistical_evaluator
        )

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.add(
            lambda msg: print(msg),
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

    def _calculate_score(
        self, prediction: str, correct_answer: str
    ) -> float:
        """Calculate similarity score between prediction and correct answer."""
        prediction = prediction.strip().lower()
        correct_answer = correct_answer.strip().lower()

        if correct_answer in prediction:
            return 1.0

        similarity = SequenceMatcher(
            None, prediction, correct_answer
        ).ratio()
        return similarity if similarity > 0.8 else 0.0

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment in text and return score between 0.1 and 1.0.
        Basic implementation - could be enhanced with more sophisticated NLP.
        """
        # List of positive and negative sentiment words
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "helpful",
            "perfect",
            "thank",
            "thanks",
            "appreciated",
            "love",
            "nice",
        }
        negative_words = {
            "bad",
            "poor",
            "terrible",
            "horrible",
            "useless",
            "waste",
            "unhelpful",
            "wrong",
            "fail",
            "failed",
            "confused",
            "disappointing",
        }

        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)

        total_count = pos_count + neg_count
        if total_count == 0:
            return 0.5  # Neutral sentiment

        sentiment = (
            (pos_count / (pos_count + neg_count))
            if total_count > 0
            else 0.5
        )
        # Scale to 0.1-1.0 range
        return max(0.1, min(1.0, 0.1 + sentiment * 0.9))

    def validate_function_schema(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validates if a given function schema follows JSON Schema specification."""
        return self.function_evaluator.validate_function_schema(
            schema
        )

    def evaluate_function_call(
        self,
        function_schema: Dict[str, Any],
        test_cases: List[Dict[str, Any]],
        expected_outputs: List[Any],
    ) -> FunctionCallResult:
        """Evaluates function calling implementation against test cases."""
        return self.function_evaluator.evaluate_function_call(
            function_schema=function_schema,
            test_cases=test_cases,
            expected_outputs=expected_outputs,
        )

    def _compare_outputs(self, actual: Any, expected: Any) -> float:
        """Compares actual output with expected output and returns similarity score."""
        if isinstance(actual, dict) and isinstance(expected, dict):
            actual_keys = set(actual.keys())
            expected_keys = set(expected.keys())

            key_similarity = len(actual_keys & expected_keys) / len(
                expected_keys
            )

            value_scores = []
            for key in actual_keys & expected_keys:
                value_scores.append(
                    self._compare_outputs(actual[key], expected[key])
                )

            value_similarity = (
                sum(value_scores) / len(value_scores)
                if value_scores
                else 0
            )
            return (key_similarity + value_similarity) / 2

        elif isinstance(actual, (list, tuple)) and isinstance(
            expected, (list, tuple)
        ):
            if len(actual) != len(expected):
                return 0.5
            element_scores = [
                self._compare_outputs(a, e)
                for a, e in zip(actual, expected)
            ]
            return sum(element_scores) / len(element_scores)
        else:
            return float(actual == expected)

    def evaluate_model(
        self,
        model: ModelInterface,
        questions: List[str],
        correct_answers: List[str],
        imgs: Optional[List[str]] = None,
        cluster_ids: Optional[List[str]] = None,
        num_samples: int = 1,
        batch_size: int = 32,
        cache_key: Optional[str] = None,
        function_schema: Optional[Dict[str, Any]] = None,
        function_test_cases: Optional[List[Dict[str, Any]]] = None,
        function_expected_outputs: Optional[List[Any]] = None,
        analyze_sentiment: bool = False,
    ) -> EvalResult:
        """
        Enhanced evaluation that includes statistical analysis, function calling,
        and optional sentiment analysis.
        """
        start_time = time.time()

        # Check cache
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    return EvalResult(**json.load(f))

        # Validate inputs
        assert len(questions) == len(
            correct_answers
        ), "Questions and answers must have same length"
        if cluster_ids:
            assert len(cluster_ids) == len(
                questions
            ), "Cluster IDs must match question length"

        # Run model predictions
        all_scores = []
        sentiment_scores = [] if analyze_sentiment else None

        with ThreadPoolExecutor() as executor:
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i : i + batch_size]
                batch_answers = correct_answers[i : i + batch_size]

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

                batch_scores = list(
                    executor.map(lambda f: f(), tasks)
                )
                all_scores.extend(batch_scores)

                if analyze_sentiment:
                    batch_predictions = [
                        model.run(q) for q in batch_questions
                    ]
                    sentiment_scores.extend(
                        [
                            self._analyze_sentiment(p)
                            for p in batch_predictions
                        ]
                    )

        # Calculate statistics
        scores_array = np.array(all_scores)
        mean_score = np.mean(scores_array)

        if cluster_ids:
            sem = self._calculate_clustered_sem(
                scores_array, cluster_ids
            )
        else:
            sem = stats.sem(scores_array)

        ci_lower, ci_upper = stats.norm.interval(
            0.95, loc=mean_score, scale=sem
        )

        # Evaluate function calling if provided
        function_results = None
        if (
            function_schema
            and function_test_cases
            and function_expected_outputs
        ):
            function_results = self.evaluate_function_call(
                function_schema,
                function_test_cases,
                function_expected_outputs,
            ).__dict__

        # Create result
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
            function_call_results=function_results,
            sentiment_score=(
                np.mean(sentiment_scores)
                if sentiment_scores
                else None
            ),
        )

        # Cache results
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.json"
            with open(cache_path, "w") as f:
                json.dump(result.__dict__, f)

        return result

    def _calculate_clustered_sem(
        self, scores: np.ndarray, cluster_ids: List[str]
    ) -> float:
        """Calculate clustered standard error of the mean."""
        import pandas as pd

        df = pd.DataFrame({"score": scores, "cluster": cluster_ids})
        cluster_means = df.groupby("cluster")["score"].mean()
        n_clusters = len(cluster_means)
        cluster_variance = cluster_means.var()
        return np.sqrt(cluster_variance / n_clusters)


def create_test_suite(
    function_schema: Dict[str, Any], num_cases: int = 10
) -> List[Dict[str, Any]]:
    """Creates a test suite for a given function schema."""
    test_cases = []
    properties = function_schema["parameters"]["properties"]

    for _ in range(num_cases):
        test_case = {}
        for prop_name, prop_schema in properties.items():
            test_case[prop_name] = _generate_test_value(prop_schema)
        test_cases.append(test_case)

    return test_cases


def _generate_test_value(property_schema: Dict[str, Any]) -> Any:
    """Helper function to generate test values based on property schema."""
    schema_type = property_schema.get("type", "string")

    if schema_type == "string":
        return "test_string"
    elif schema_type == "number":
        return 42.0
    elif schema_type == "integer":
        return 42
    elif schema_type == "boolean":
        return True
    elif schema_type == "array":
        items_schema = property_schema.get(
            "items", {"type": "string"}
        )
        return [_generate_test_value(items_schema) for _ in range(2)]
    elif schema_type == "object":
        obj = {}
        for prop_name, prop_schema in property_schema.get(
            "properties", {}
        ).items():
            obj[prop_name] = _generate_test_value(prop_schema)
        return obj
    else:
        return None
