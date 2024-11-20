import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent

from main import StatisticalModelEvaluator

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
