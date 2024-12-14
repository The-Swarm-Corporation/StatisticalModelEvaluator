# Statistical Model Evaluator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

A robust, production-ready framework for statistically rigorous evaluation of language models, implementing the methodology described in ["A Statistical Approach to Model Evaluations"](https://www.anthropic.com/research/statistical-approach-to-model-evals) (2024).



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

## Instal


```bash
pip3 install -U evalops
```

## Usage 

```python
import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent

from evalops import StatisticalModelEvaluator

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create instances of the OpenAIChat class with different models
model_gpt4 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o", temperature=0.1
)

model_gpt35 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize a general knowledge agent
agent = Agent(
    agent_name="General-Knowledge-Agent",
    system_prompt="You are a helpful assistant that answers general knowledge questions accurately and concisely.",
    llm=model_gpt4,
    max_loops=1,
    dynamic_temperature_enabled=True,
    saved_state_path="general_agent.json",
    user_name="swarms_corp",
    context_length=200000,
    return_step_meta=False,
    output_type="string",
)

evaluator = StatisticalModelEvaluator(cache_dir="./eval_cache")

# General knowledge test cases
general_questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
]

general_answers = [
    "Paris",
    "William Shakespeare",
    "Jupiter",
    "Au",
    "Leonardo da Vinci",
]

# Evaluate models on general knowledge questions
result_gpt4 = evaluator.evaluate_model(
    model=agent,
    questions=general_questions,
    correct_answers=general_answers,
    num_samples=5,
)

result_gpt35 = evaluator.evaluate_model(
    model=agent,
    questions=general_questions,
    correct_answers=general_answers,
    num_samples=5,
)

# Compare model performance
comparison = evaluator.compare_models(result_gpt4, result_gpt35)

# Print results
print(f"GPT-4 Mean Score: {result_gpt4.mean_score:.3f}")
print(f"GPT-3.5 Mean Score: {result_gpt35.mean_score:.3f}")
print(
    f"Significant Difference: {comparison['significant_difference']}"
)
print(f"P-value: {comparison['p_value']:.3f}")

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


## Loading datasets from huggingface

```python
import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent

from evalops import StatisticalModelEvaluator
from evalops.huggingface_loader import EvalDatasetLoader

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create instance of OpenAIChat
model_gpt4 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o", temperature=0.1
)

# Initialize a general knowledge agent
agent = Agent(
    agent_name="General-Knowledge-Agent",
    system_prompt="You are a helpful assistant that answers general knowledge questions accurately and concisely.",
    llm=model_gpt4,
    max_loops=1,
    dynamic_temperature_enabled=True,
    saved_state_path="general_agent.json",
    user_name="swarms_corp",
    context_length=200000,
    return_step_meta=False,
    output_type="string",
)

evaluator = StatisticalModelEvaluator(cache_dir="./eval_cache")

# Initialize the dataset loader
eval_loader = EvalDatasetLoader(cache_dir="./eval_cache")

# Load a common evaluation dataset
questions, answers = eval_loader.load_dataset(
    dataset_name="truthful_qa",
    subset="multiple_choice",
    split="validation",
    answer_key="best_question",
)

# Use the loaded questions and answers with your evaluator
result_gpt4 = evaluator.evaluate_model(
    model=agent,
    questions=questions,
    correct_answers=answers,
    num_samples=5,
)


# Print results
print(result_gpt4)


```


## Simple Eval
`eval` is a simple function that wraps the evaluator class and makes it easy to use.

```python
import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent

from evalops.wrapper import eval

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Create instance of OpenAIChat
model_gpt4 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o", temperature=0.1
)

# Initialize a general knowledge agent
agent = Agent(
    agent_name="General-Knowledge-Agent",
    system_prompt="You are a helpful assistant that answers general knowledge questions accurately and concisely.",
    llm=model_gpt4,
    max_loops=1,
    dynamic_temperature_enabled=True,
    saved_state_path="general_agent.json",
    user_name="swarms_corp",
    context_length=200000,
    return_step_meta=False,
    output_type="string",
)


# General knowledge test cases
general_questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
]

# Answers
general_answers = [
    "Paris",
    "William Shakespeare",
    "Jupiter",
    "Au",
    "Leonardo da Vinci",
]


print(eval(
    questions = general_questions,
    answers=general_answers,
    agent=agent,
    samples=2,
))

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

## 🙋‍♂️ Support

- 📫 Email: kye@swarms.world
- 💬 Issues: [GitHub Issues](https://github.com/The-Swarm-Corporation/StatisticalModelEvaluator/issues)
- 📖 Documentation: [Full Documentation](https://docs.swarms.world)

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by the paper "A Statistical Approach to Model Evaluations" (2024)
