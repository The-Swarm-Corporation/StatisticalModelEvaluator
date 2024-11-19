# Statistical Model Evaluator


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


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
