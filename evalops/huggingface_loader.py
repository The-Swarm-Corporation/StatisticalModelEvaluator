from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from datasets import load_dataset, Dataset
from loguru import logger
import json
import os
import hashlib

class EvalDatasetLoader:
    """
    A class to load and manage evaluation datasets from Hugging Face and local sources.
    
    This class provides functionality to:
    - Load common evaluation datasets from Hugging Face
    - Cache datasets locally for faster subsequent access
    - Format datasets into Q&A pairs for model evaluation
    - Handle different dataset structures and formats
    
    Attributes:
        cache_dir (Path): Directory for caching downloaded datasets
        logger: Loguru logger instance for tracking operations
        loaded_datasets (Dict): Cache of currently loaded datasets
    """
    
    def __init__(self, cache_dir: Union[str, Path] = "./eval_cache"):
        """
        Initialize the EvalDatasetLoader.
        
        Args:
            cache_dir: Directory path for caching datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.add(
            self.cache_dir / "eval_loader.log",
            rotation="10 MB",
            retention="1 month",
            level="INFO"
        )
        
        self.loaded_datasets: Dict[str, Dataset] = {}
        logger.info(f"Initialized EvalDatasetLoader with cache directory: {self.cache_dir}")

    def load_dataset(
        self,
        dataset_name: str,
        subset: Optional[str] = None,
        split: str = "test",
        question_key: str = "question",
        answer_key: str = "answer"
    ) -> Tuple[List[str], List[str]]:
        """
        Load a dataset from Hugging Face and format it as questions and answers.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face (e.g., "truthful_qa", "hellaswag")
            subset: Specific subset of the dataset if applicable
            split: Dataset split to use (default: "test")
            question_key: Key for questions in the dataset
            answer_key: Key for answers in the dataset
            
        Returns:
            Tuple containing lists of questions and corresponding answers
            
        Raises:
            ValueError: If dataset cannot be loaded or required keys are missing
        """
        cache_key = f"{dataset_name}_{subset}_{split}"
        cache_path = self._get_cache_path(cache_key)

        try:
            # Try loading from cache first
            if cache_path.exists():
                logger.info(f"Loading cached dataset: {cache_key}")
                return self._load_from_cache(cache_path)

            # Load from Hugging Face if not cached
            logger.info(f"Fetching dataset from Hugging Face: {dataset_name}")
            dataset = load_dataset(dataset_name, subset, split=split)
            
            if dataset is None:
                raise ValueError(f"Failed to load dataset: {dataset_name}")

            # Extract questions and answers
            questions: List[str] = []
            answers: List[str] = []

            for item in dataset:
                try:
                    question = item[question_key]
                    answer = item[answer_key]
                    
                    # Handle different answer formats
                    if isinstance(answer, (list, dict)):
                        if isinstance(answer, list):
                            answer = answer[0] if answer else ""
                        elif isinstance(answer, dict):
                            answer = answer.get('text', answer.get('answer', ''))
                    
                    questions.append(str(question))
                    answers.append(str(answer))
                
                except KeyError as e:
                    logger.warning(f"Skipping item due to missing key: {e}")
                    continue

            # Cache the processed data
            self._save_to_cache(cache_path, questions, answers)
            logger.success(f"Successfully loaded and cached dataset: {cache_key}")
            
            return questions, answers

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise

    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate a unique cache file path based on the dataset parameters."""
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"dataset_{hashed_key}.json"

    def _save_to_cache(self, cache_path: Path, questions: List[str], answers: List[str]):
        """Save processed dataset to cache."""
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump({'questions': questions, 'answers': answers}, f)

    def _load_from_cache(self, cache_path: Path) -> Tuple[List[str], List[str]]:
        """Load processed dataset from cache."""
        with cache_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            return data['questions'], data['answers']

    def clear_cache(self):
        """Clear all cached datasets."""
        try:
            for file in self.cache_dir.glob("dataset_*.json"):
                file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
