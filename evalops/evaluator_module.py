import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Union

import spacy
from loguru import logger
from num2words import num2words
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AnswerMatch:
    """Stores the match results for a single evaluation."""

    score: float
    match_type: str
    details: Dict[str, float]
    error_analysis: Optional[str] = None


class AdvancedScorer:
    """
    Comprehensive scoring system that combines multiple evaluation strategies.
    """

    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.8,
        cache_embeddings: bool = True,
    ):
        # Initialize models
        try:
            self.semantic_model = SentenceTransformer(semantic_model)
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.semantic_model = None
            self.nlp = None

        # Default weights for different matching strategies
        self.weights = weights or {
            "exact": 1.0,
            "semantic": 0.8,
            "numeric": 0.9,
            "substring": 0.7,
            "sequence": 0.6,
        }

        self.threshold = threshold
        self.cache = {} if cache_embeddings else None

    def calculate_score(
        self,
        prediction: Union[str, List[str]],
        correct_answer: Union[str, List[str]],
        context: Optional[str] = None,
        require_reasoning: bool = False,
    ) -> AnswerMatch:
        """
        Calculate comprehensive score using multiple matching strategies.

        Args:
            prediction: Model's predicted answer(s)
            correct_answer: Correct answer(s)
            context: Optional context for contextual evaluation
            require_reasoning: Whether to evaluate reasoning quality

        Returns:
            AnswerMatch object with score and analysis
        """
        # Handle list inputs
        if isinstance(prediction, list):
            prediction = " ".join(prediction)
        if isinstance(correct_answer, list):
            # Try each correct answer and take the best score
            scores = [
                self._evaluate_single_answer(
                    prediction, ans, context, require_reasoning
                )
                for ans in correct_answer
            ]
            return max(scores, key=lambda x: x.score)

        return self._evaluate_single_answer(
            prediction, correct_answer, context, require_reasoning
        )

    def _evaluate_single_answer(
        self,
        prediction: str,
        correct_answer: str,
        context: Optional[str],
        require_reasoning: bool,
    ) -> AnswerMatch:
        # Normalize texts
        pred_norm = self._normalize_text(prediction)
        ans_norm = self._normalize_text(correct_answer)

        # Calculate various match scores
        match_scores = {
            "exact": self._exact_match(pred_norm, ans_norm),
            "semantic": self._semantic_match(pred_norm, ans_norm),
            "numeric": self._numeric_match(pred_norm, ans_norm),
            "substring": self._substring_match(pred_norm, ans_norm),
            "sequence": self._sequence_match(pred_norm, ans_norm),
        }

        # Add contextual evaluation if context provided
        if context:
            match_scores["contextual"] = self._contextual_match(
                prediction, correct_answer, context
            )

        # Add reasoning evaluation if required
        if require_reasoning:
            match_scores["reasoning"] = self._evaluate_reasoning(
                prediction
            )

        # Calculate weighted score
        weighted_score = sum(
            score * self.weights.get(match_type, 0.5)
            for match_type, score in match_scores.items()
        ) / sum(self.weights.values())

        # Generate error analysis if score is below threshold
        error_analysis = None
        if weighted_score < self.threshold:
            error_analysis = self._analyze_error(
                prediction, correct_answer, match_scores
            )

        return AnswerMatch(
            score=weighted_score,
            match_type=self._determine_match_type(match_scores),
            details=match_scores,
            error_analysis=error_analysis,
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _exact_match(self, pred: str, ans: str) -> float:
        """Check for exact match."""
        return float(pred == ans)

    def _semantic_match(self, pred: str, ans: str) -> float:
        """Calculate semantic similarity using embeddings."""
        if not self.semantic_model:
            return 0.0

        try:
            # Use cache if available
            if self.cache is not None:
                pred_emb = self.cache.get(
                    pred
                ) or self.semantic_model.encode(pred)
                ans_emb = self.cache.get(
                    ans
                ) or self.semantic_model.encode(ans)
                self.cache[pred] = pred_emb
                self.cache[ans] = ans_emb
            else:
                pred_emb = self.semantic_model.encode(pred)
                ans_emb = self.semantic_model.encode(ans)

            return float(
                cosine_similarity([pred_emb], [ans_emb])[0][0]
            )
        except Exception as e:
            logger.error(f"Semantic matching error: {e}")
            return 0.0

    def _numeric_match(self, pred: str, ans: str) -> float:
        """Check for numerical equivalence."""
        try:
            pred_nums = self._extract_numbers(pred)
            ans_nums = self._extract_numbers(ans)
            if pred_nums and ans_nums:
                return float(
                    any(
                        abs(p - a) < 1e-6
                        for p in pred_nums
                        for a in ans_nums
                    )
                )
        except Exception as e:
            logger.debug(f"Numeric matching error: {e}")
        return 0.0

    def _substring_match(self, pred: str, ans: str) -> float:
        """Check if answer is contained in prediction."""
        return float(ans in pred)

    def _sequence_match(self, pred: str, ans: str) -> float:
        """Calculate sequence similarity ratio."""
        return SequenceMatcher(None, pred, ans).ratio()

    def _contextual_match(
        self, pred: str, ans: str, context: str
    ) -> float:
        """Evaluate answer in context."""
        if not self.nlp:
            return 0.0

        try:
            # Extract key entities and relations
            pred_doc = self.nlp(pred)
            ans_doc = self.nlp(ans)
            context_doc = self.nlp(context)

            # Compare entity overlap
            pred_ents = set(e.text.lower() for e in pred_doc.ents)
            ans_ents = set(e.text.lower() for e in ans_doc.ents)
            context_ents = set(
                e.text.lower() for e in context_doc.ents
            )

            entity_score = len(
                pred_ents & ans_ents & context_ents
            ) / max(len(ans_ents), 1)
            return entity_score
        except Exception as e:
            logger.error(f"Contextual matching error: {e}")
            return 0.0

    def _evaluate_reasoning(self, text: str) -> float:
        """Evaluate quality of reasoning in answer."""
        if not self.nlp:
            return 0.0

        try:
            doc = self.nlp(text)

            # Check for reasoning indicators
            reasoning_markers = [
                "because",
                "therefore",
                "since",
                "as a result",
                "consequently",
            ]
            has_markers = any(
                marker in text.lower() for marker in reasoning_markers
            )

            # Check sentence structure
            sent_count = len(list(doc.sents))
            has_multiple_sentences = sent_count > 1

            return (has_markers + has_multiple_sentences) / 2
        except Exception as e:
            logger.error(f"Reasoning evaluation error: {e}")
            return 0.0

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text, including written numbers."""
        numbers = []

        # Extract numeric values
        numeric_matches = re.findall(r"-?\d*\.?\d+", text)
        numbers.extend(float(n) for n in numeric_matches)

        # Extract written numbers
        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                try:
                    num = num2words(token.text, to="cardinal")
                    numbers.append(float(num))
                except:
                    continue

        return numbers

    def _determine_match_type(self, scores: Dict[str, float]) -> str:
        """Determine the primary type of match based on scores."""
        return max(scores.items(), key=lambda x: x[1])[0]

    def _analyze_error(
        self,
        prediction: str,
        correct_answer: str,
        scores: Dict[str, float],
    ) -> str:
        """Generate detailed error analysis."""
        analysis = []

        if scores["exact"] < 1.0:
            if scores["semantic"] > 0.8:
                analysis.append(
                    "Semantically similar but not exact match"
                )
            if scores["numeric"] > 0.8:
                analysis.append(
                    "Numerically equivalent but different format"
                )
            if scores["substring"] > 0.8:
                analysis.append(
                    "Contains correct answer but with extra information"
                )

        return (
            "; ".join(analysis)
            if analysis
            else "No specific error pattern identified"
        )
