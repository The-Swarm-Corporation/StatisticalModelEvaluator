import json
from typing import Any, Dict, List

from evalops.function_eval import FunctionCallEvaluator
from evalops.main import StatisticalModelEvaluator


# First, let's create a mock model class that implements function calling
class MockLanguageModel:
    """
    Mock language model that implements function calling capabilities.
    This simulates how a real LLM might handle function calls.
    """

    def run(
        self, task: str, functions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Simulates model inference with function calling"""
        # This is a mock implementation - in reality, this would be an LLM
        if "weather" in task.lower():
            return {
                "function_call": {
                    "name": "get_weather",
                    "arguments": {
                        "location": "New York",
                        "date": "2024-01-02",
                    },
                }
            }
        elif "calculate" in task.lower():
            return {
                "function_call": {
                    "name": "calculate_total",
                    "arguments": {"amount": 100, "tax_rate": 0.1},
                }
            }
        return {"content": "I cannot help with that task."}


# Define some example functions that we want to test
def get_weather(location: str, date: str) -> Dict[str, Any]:
    """Mock weather function"""
    return {
        "temperature": 72,
        "conditions": "sunny",
        "location": location,
        "date": date,
    }


def calculate_total(amount: float, tax_rate: float) -> Dict[str, Any]:
    """Calculate total with tax"""
    total = amount * (1 + tax_rate)
    return {
        "total": total,
        "breakdown": {
            "base_amount": amount,
            "tax_amount": amount * tax_rate,
        },
    }


# Define the function schemas
function_schemas = [
    {
        "name": "get_weather",
        "description": "Get weather information for a location and date",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format",
                },
            },
            "required": ["location", "date"],
        },
    },
    {
        "name": "calculate_total",
        "description": "Calculate total amount including tax",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Base amount",
                },
                "tax_rate": {
                    "type": "number",
                    "description": "Tax rate as decimal",
                },
            },
            "required": ["amount", "tax_rate"],
        },
    },
]


def main():
    # Initialize evaluators
    base_evaluator = StatisticalModelEvaluator()
    func_evaluator = FunctionCallEvaluator(base_evaluator)

    # Initialize model
    model = MockLanguageModel()

    # Test cases for weather function
    weather_test_cases = [
        {
            "task": "What's the weather like in New York today?",
            "expected_schema": function_schemas[0],
            "expected_output": {
                "temperature": 72,
                "conditions": "sunny",
                "location": "New York",
                "date": "2024-01-02",
            },
        },
        {
            "task": "Tell me the weather forecast for New York",
            "expected_schema": function_schemas[0],
            "expected_output": {
                "temperature": 72,
                "conditions": "sunny",
                "location": "New York",
                "date": "2024-01-02",
            },
        },
    ]

    # Test cases for calculation function
    calculation_test_cases = [
        {
            "task": "Calculate the total for $100 with 10% tax",
            "expected_schema": function_schemas[1],
            "expected_output": {
                "total": 110.0,
                "breakdown": {
                    "base_amount": 100.0,
                    "tax_amount": 10.0,
                },
            },
        }
    ]

    # Evaluate each function schema
    for schema in function_schemas:
        print(f"\nEvaluating schema for {schema['name']}:")
        schema_result = func_evaluator.validate_function_schema(
            schema
        )
        print(
            f"Schema validation result: {json.dumps(schema_result, indent=2)}"
        )

    # Run test cases for weather function
    print("\nTesting weather function:")
    weather_results = []
    for test_case in weather_test_cases:
        # Get model response
        model_response = model.run(
            test_case["task"], functions=function_schemas
        )

        if "function_call" in model_response:
            # Execute the function with model's arguments
            func_name = model_response["function_call"]["name"]
            func_args = model_response["function_call"]["arguments"]

            if func_name == "get_weather":
                actual_output = get_weather(**func_args)
            else:
                actual_output = {"error": "Wrong function called"}

            # Evaluate the execution
            result = func_evaluator.evaluate_function_call(
                function_schema=test_case["expected_schema"],
                test_cases=[func_args],
                expected_outputs=[test_case["expected_output"]],
            )
            weather_results.append(result)

            print(f"\nTest case: {test_case['task']}")
            print(f"Model called: {func_name}")
            print(f"Arguments: {json.dumps(func_args, indent=2)}")
            print(f"Output: {json.dumps(actual_output, indent=2)}")
            print(
                f"Evaluation result: {json.dumps(result.model_dump(), indent=2)}"
            )

    # Run test cases for calculation function
    print("\nTesting calculation function:")
    calc_results = []
    for test_case in calculation_test_cases:
        model_response = model.run(
            test_case["task"], functions=function_schemas
        )

        if "function_call" in model_response:
            func_name = model_response["function_call"]["name"]
            func_args = model_response["function_call"]["arguments"]

            if func_name == "calculate_total":
                actual_output = calculate_total(**func_args)
            else:
                actual_output = {"error": "Wrong function called"}

            result = func_evaluator.evaluate_function_call(
                function_schema=test_case["expected_schema"],
                test_cases=[func_args],
                expected_outputs=[test_case["expected_output"]],
            )
            calc_results.append(result)

            print(f"\nTest case: {test_case['task']}")
            print(f"Model called: {func_name}")
            print(f"Arguments: {json.dumps(func_args, indent=2)}")
            print(f"Output: {json.dumps(actual_output, indent=2)}")
            print(
                f"Evaluation result: {json.dumps(result.model_dump(), indent=2)}"
            )

    # Calculate aggregate statistics
    all_results = weather_results + calc_results
    avg_matching_score = sum(
        r.matching_score for r in all_results
    ) / len(all_results)
    schema_validity = all(r.schema_valid for r in all_results)
    execution_validity = all(r.execution_valid for r in all_results)

    print("\nAggregate Results:")
    print(f"Average matching score: {avg_matching_score:.2f}")
    print(f"All schemas valid: {schema_validity}")
    print(f"All executions valid: {execution_validity}")


if __name__ == "__main__":
    main()
