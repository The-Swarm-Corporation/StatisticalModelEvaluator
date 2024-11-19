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