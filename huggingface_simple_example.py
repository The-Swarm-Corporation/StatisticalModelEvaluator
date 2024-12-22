import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent

from evalops.huggingface_loader import EvalDatasetLoader
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

# Initialize the dataset loader
eval_loader = EvalDatasetLoader(cache_dir="./eval_cache")


# Load a common evaluation dataset
questions, answers = eval_loader.load_dataset(
    dataset_name="truthful_qa",
    subset="multiple_choice",
    split="validation",
    answer_key="best_question",
)

print(
    eval(
        questions=questions,
        answers=answers,
        agent=agent,
        samples=1,
    )
)
