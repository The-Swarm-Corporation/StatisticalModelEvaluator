from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from datasets import load_dataset, Dataset
from loguru import logger
import json
import hashlib
from datetime import datetime


class EvalDatasetLoader:
    def __init__(self, cache_dir: Union[str, Path] = "./eval_cache"):
        """
        Initialize the EvalDatasetLoader with improved logging and cache management.

        Args:
            cache_dir: Directory path for caching datasets. Created if it doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure logger with detailed formatting
        logger.add(
            self.cache_dir / "eval_loader.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="10 MB",
            retention="1 month",
            level="INFO",
        )

        self.loaded_datasets: Dict[str, Dataset] = {}
        logger.info(
            f"Initialized EvalDatasetLoader with cache directory: {self.cache_dir}"
        )

    def load_dataset(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "test",
        question_key: str = "question",
        answer_key: str = "answer",
        fallback_question_keys: List[str] = [
            "input",
            "premise",
            "context",
        ],
        fallback_answer_keys: List[str] = [
            "output",
            "hypothesis",
            "label",
            "mc1_targets",
            "mc2_targets",
        ],
    ) -> Tuple[List[str], List[str]]:
        """
        Load a dataset with enhanced error handling and flexible key mapping.

        Args:
            dataset_name: Name of the dataset on Hugging Face
            subset: Specific subset of the dataset
            split: Dataset split to use
            question_key: Primary key for questions
            answer_key: Primary key for answers
            fallback_question_keys: Alternative keys to try for questions
            fallback_answer_keys: Alternative keys to try for answers

        Returns:
            Tuple of (questions, answers) lists

        Raises:
            ValueError: If no valid data could be extracted after trying all keys
        """
        cache_key = self._generate_cache_key(
            dataset_name, subset, split
        )
        cache_path = self._get_cache_path(cache_key)

        try:
            # Try loading from cache first
            if cache_path.exists():
                logger.info(f"Loading cached dataset: {cache_key}")
                questions, answers = self._load_from_cache(cache_path)
                if questions and answers:
                    return questions, answers
                logger.warning(
                    "Cache data invalid or empty, reloading from source"
                )

            # Load from Hugging Face
            logger.info(
                f"Fetching dataset from Hugging Face: {dataset_name}"
            )
            dataset_args = {"path": dataset_name, "split": split}
            if subset:
                dataset_args["name"] = subset

            dataset = load_dataset(**dataset_args)

            if not dataset or len(dataset) == 0:
                raise ValueError(
                    f"Dataset {dataset_name} is empty or failed to load"
                )

            # Log dataset structure for debugging
            sample_item = dataset[0]
            logger.debug(
                f"Dataset structure: {list(sample_item.keys())}"
            )

            # Extract questions and answers
            questions, answers = [], []
            for item in dataset:
                try:
                    # Try to get question using primary and fallback keys
                    question = None
                    for key in [
                        question_key
                    ] + fallback_question_keys:
                        if key in item:
                            question = item[key]
                            break

                    # Try to get answer using primary and fallback keys
                    answer = None
                    for key in [answer_key] + fallback_answer_keys:
                        if key in item:
                            answer = item[key]
                            break

                    if question is None or answer is None:
                        logger.warning(
                            f"Skipping item. Available keys: {list(item.keys())}"
                        )
                        continue

                    # Process answer based on its type
                    if isinstance(answer, list):
                        answer = answer[0] if answer else ""
                    elif isinstance(answer, dict):
                        for possible_key in [
                            "text",
                            "answer",
                            "value",
                            "label",
                        ]:
                            if possible_key in answer:
                                answer = answer[possible_key]
                                break
                        else:
                            answer = str(answer)

                    questions.append(str(question))
                    answers.append(str(answer))

                except Exception as e:
                    logger.warning(f"Error processing item: {str(e)}")
                    continue

            if not questions or not answers:
                raise ValueError(
                    f"No valid question-answer pairs found in dataset {dataset_name}. "
                    f"Tried question keys: {[question_key] + fallback_question_keys}, "
                    f"answer keys: {[answer_key] + fallback_answer_keys}"
                )

            # Cache the processed data
            self._save_to_cache(cache_path, questions, answers)
            logger.success(
                f"Successfully loaded dataset {cache_key} with {len(questions)} examples"
            )

            return questions, answers

        except Exception as e:
            logger.error(
                f"Error loading dataset {dataset_name}: {str(e)}"
            )
            raise

    def _generate_cache_key(
        self, dataset_name: str, subset: Optional[str], split: str
    ) -> str:
        """Generate a unique cache key for the dataset configuration."""
        components = [dataset_name]
        if subset:
            components.append(subset)
        components.append(split)
        return "_".join(components)

    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate a unique cache file path using MD5 hashing."""
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"dataset_{hashed_key}.json"

    def _save_to_cache(
        self,
        cache_path: Path,
        questions: List[str],
        answers: List[str],
    ):
        """Save processed dataset to cache with validation and metadata."""
        if not questions or not answers:
            logger.warning(
                "Attempting to cache empty dataset, skipping"
            )
            return

        data = {
            "questions": questions,
            "answers": answers,
            "metadata": {
                "num_examples": len(questions),
                "cached_at": datetime.now().isoformat(),
                "version": "1.0",
            },
        }

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Cached {len(questions)} examples to {cache_path}"
        )

    def _load_from_cache(
        self, cache_path: Path
    ) -> Tuple[List[str], List[str]]:
        """Load and validate cached dataset."""
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            questions = data.get("questions", [])
            answers = data.get("answers", [])

            # Validate cache data
            if (
                not questions
                or not answers
                or len(questions) != len(answers)
            ):
                logger.warning(f"Invalid cache data in {cache_path}")
                return [], []

            logger.info(
                f"Loaded {len(questions)} examples from cache"
            )
            return questions, answers

        except Exception as e:
            logger.error(
                f"Error loading cache {cache_path}: {str(e)}"
            )
            return [], []

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache files with age filtering option.

        Args:
            older_than_days: If provided, only clear files older than this many days
        """
        try:
            count = 0
            current_time = datetime.now()

            for file in self.cache_dir.glob("dataset_*.json"):
                if older_than_days:
                    file_time = datetime.fromtimestamp(
                        file.stat().st_mtime
                    )
                    age_days = (current_time - file_time).days
                    if age_days <= older_than_days:
                        continue

                file.unlink()
                count += 1

            logger.info(f"Cleared {count} cached files")

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
