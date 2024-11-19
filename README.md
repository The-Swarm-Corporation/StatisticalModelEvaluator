# Statistical Model Evaluator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust, production-ready framework for statistically rigorous evaluation of language models, implementing the methodology described in "A Statistical Approach to Model Evaluations" (2024).


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


## 🚀 Features

- **Statistical Robustness**: Leverages Central Limit Theorem for reliable metrics
- **Clustered Standard Errors**: Handles non-independent question groups
- **Variance Reduction**: Multiple sampling strategies and parallel processing
- **Paired Difference Analysis**: Sophisticated model comparison tools
- **Power Analysis**: Sample size determination for meaningful comparisons
- **Production Ready**: 
  - Comprehensive logging
  - Type hints throughout
  - Error handling
  - Result caching
  - Parallel processing
  - Modular design

## 📋 Requirements

- Python 3.8+
- Dependencies:
  ```
  numpy>=1.21.0
  pandas>=1.3.0
  scipy>=1.7.0
  loguru>=0.6.0
  ```



## Usage 

```python
from main import StatisticalModelEvaluator

class MyModel:
    def run(self, task: str) -> str:
        # Your model implementation here
        return "model response"

# Initialize model and evaluator
model = MyModel()
evaluator = StatisticalModelEvaluator(cache_dir="./eval_cache")

# Run evaluation
result = evaluator.evaluate_model(
    model=model,
    questions=["What is 2+2?", "Who wrote Hamlet?"],
    correct_answers=["4", "Shakespeare"],
    num_samples=3
)

# Compare two models
model_a = MyModel()
model_b = MyModel()
result_a = evaluator.evaluate_model(
    model=model_a,
    questions=["What is the capital of France?", "What is the square root of 16?"],
    correct_answers=["Paris", "4"],
    num_samples=5
)
result_b = evaluator.evaluate_model(
    model=model_b,
    questions=["What is the capital of France?", "What is the square root of 16?"],
    correct_answers=["Paris", "4"],
    num_samples=5
)
comparison = evaluator.compare_models(result_a, result_b)
```


## 📖 Detailed Usage

### Basic Model Evaluation

```python
class MyLanguageModel:
    def run(self, task: str) -> str:
        # Your model implementation
        return "model response"

evaluator = StatisticalModelEvaluator(
    cache_dir="./eval_cache",
    log_level="INFO",
    random_seed=42
)

# Prepare your evaluation data
questions = ["Question 1", "Question 2", ...]
answers = ["Answer 1", "Answer 2", ...]

# Run evaluation
result = evaluator.evaluate_model(
    model=MyLanguageModel(),
    questions=questions,
    correct_answers=answers,
    num_samples=3,  # Number of times to sample each question
    batch_size=32,  # Batch size for parallel processing
    cache_key="model_v1"  # Optional caching key
)

# Access results
print(f"Mean Score: {result.mean_score:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Handling Clustered Questions

```python
# For questions that are grouped (e.g., multiple questions about the same passage)
cluster_ids = ["passage1", "passage1", "passage2", "passage2", ...]

result = evaluator.evaluate_model(
    model=MyLanguageModel(),
    questions=questions,
    correct_answers=answers,
    cluster_ids=cluster_ids
)
```

### Comparing Models

```python
# Evaluate two models
result_a = evaluator.evaluate_model(model=ModelA(), ...)
result_b = evaluator.evaluate_model(model=ModelB(), ...)

# Compare results
comparison = evaluator.compare_models(result_a, result_b)

print(f"Mean Difference: {comparison['mean_difference']:.3f}")
print(f"P-value: {comparison['p_value']:.4f}")
print(f"Significant Difference: {comparison['significant_difference']}")
```

### Power Analysis

```python
required_samples = evaluator.calculate_required_samples(
    effect_size=0.05,  # Minimum difference to detect
    baseline_variance=0.1,  # Estimated variance in scores
    power=0.8,  # Desired statistical power
    alpha=0.05  # Significance level
)

print(f"Required number of samples: {required_samples}")
```

## 🎛️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cache_dir` | Directory for caching results | `None` |
| `log_level` | Logging verbosity ("DEBUG", "INFO", etc.) | `"INFO"` |
| `random_seed` | Seed for reproducibility | `None` |
| `batch_size` | Batch size for parallel processing | `32` |
| `num_samples` | Samples per question | `1` |

## 📊 Output Formats

### EvalResult Object

```python
@dataclass
class EvalResult:
    mean_score: float      # Average score across questions
    sem: float            # Standard error of the mean
    ci_lower: float       # Lower bound of 95% CI
    ci_upper: float       # Upper bound of 95% CI
    raw_scores: List[float]  # Individual question scores
    metadata: Dict        # Additional evaluation metadata
```

### Comparison Output

```python
{
    "mean_difference": float,    # Difference between means
    "correlation": float,        # Score correlation
    "t_statistic": float,       # T-test statistic
    "p_value": float,           # Statistical significance
    "significant_difference": bool  # True if p < 0.05
}
```

## 🔍 Best Practices

1. **Sample Size**: Use power analysis to determine appropriate sample sizes
2. **Clustering**: Always specify cluster IDs when questions are grouped
3. **Caching**: Enable caching for expensive evaluations
4. **Error Handling**: Monitor logs for evaluation failures
5. **Reproducibility**: Set random seed for consistent results

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this evaluator in your research, please cite:

```bibtex
@article{statistical_model_evaluator,
    title={A Statistical Approach to Model Evaluations},
    year={2024},
    journal={ArXiv}
}
```

## 🙋‍♂️ Support

- 📫 Email: support@statisticalmodelevauator.org
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/statistical-model-evaluator/issues)
- 📖 Documentation: [Full Documentation](https://statistical-model-evaluator.readthedocs.io/)

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by the paper "A Statistical Approach to Model Evaluations" (2024)