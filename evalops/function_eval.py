from typing import Any, Dict, List
from pydantic import BaseModel, ValidationError
import jsonschema

from evalops.main import StatisticalModelEvaluator


class FunctionCallResult(BaseModel):
    """
    Stores the evaluation results for function calling tests.

    Attributes:
        schema_valid: Whether the function schema is valid JSON Schema
        execution_valid: Whether the function execution was successful
        schema_errors: List of schema validation errors if any
        execution_errors: List of execution errors if any
        matching_score: Score for how well the execution matched expected output
        metadata: Additional metadata about the evaluation
    """

    schema_valid: bool
    execution_valid: bool
    schema_errors: List[str]
    execution_errors: List[str]
    matching_score: float
    metadata: Dict[str, Any]


class FunctionCallEvaluator:
    """
    Evaluator for testing function calling capabilities and schema correctness.

    This evaluator extends the base StatisticalModelEvaluator to add specific
    function calling evaluation capabilities.
    """

    def __init__(self, base_evaluator: StatisticalModelEvaluator):
        self.base_evaluator = base_evaluator

    def validate_function_schema(
        self, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates if a given function schema is correct and follows JSON Schema specification.

        Args:
            schema: The function schema to validate

        Returns:
            Dictionary containing validation results and any errors found
        """
        errors = []
        try:
            # Validate basic JSON Schema structure
            jsonschema.Draft7Validator.check_schema(schema)

            # Check for required function schema elements
            required_fields = ["name", "description", "parameters"]
            missing_fields = [
                field
                for field in required_fields
                if field not in schema
            ]
            if missing_fields:
                errors.append(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )

            # Validate parameters object
            if "parameters" in schema:
                if not isinstance(schema["parameters"], dict):
                    errors.append("Parameters must be an object")
                else:
                    if "properties" not in schema["parameters"]:
                        errors.append(
                            "Parameters object must contain 'properties'"
                        )
                    if "required" not in schema["parameters"]:
                        errors.append(
                            "Parameters object must specify 'required' fields"
                        )

            schema_valid = len(errors) == 0

        except jsonschema.exceptions.SchemaError as e:
            schema_valid = False
            errors.append(f"Schema validation error: {str(e)}")

        return {"valid": schema_valid, "errors": errors}

    def evaluate_function_call(
        self,
        function_schema: Dict[str, Any],
        test_cases: List[Dict[str, Any]],
        expected_outputs: List[Any],
    ) -> FunctionCallResult:
        """
        Evaluates function calling implementation against test cases.

        Args:
            function_schema: The function schema to test
            test_cases: List of input test cases
            expected_outputs: List of expected outputs for each test case

        Returns:
            FunctionCallResult containing evaluation metrics
        """
        # First validate the schema
        schema_validation = self.validate_function_schema(
            function_schema
        )

        execution_errors = []
        execution_scores = []

        # If schema is valid, test execution
        if schema_validation["valid"]:
            for test_case, expected in zip(
                test_cases, expected_outputs
            ):
                try:
                    # Validate test case against schema
                    jsonschema.validate(
                        test_case, function_schema["parameters"]
                    )

                    # Compare output structure
                    execution_score = self._compare_outputs(
                        test_case, expected
                    )
                    execution_scores.append(execution_score)

                except ValidationError as e:
                    execution_errors.append(
                        f"Test case validation error: {str(e)}"
                    )
                except Exception as e:
                    execution_errors.append(
                        f"Execution error: {str(e)}"
                    )

        # Calculate average execution score
        avg_execution_score = (
            sum(execution_scores) / len(execution_scores)
            if execution_scores
            else 0.0
        )

        return FunctionCallResult(
            schema_valid=schema_validation["valid"],
            execution_valid=len(execution_errors) == 0,
            schema_errors=schema_validation["errors"],
            execution_errors=execution_errors,
            matching_score=avg_execution_score,
            metadata={
                "num_test_cases": len(test_cases),
                "test_coverage": len(execution_scores)
                / len(test_cases),
            },
        )

    def _compare_outputs(self, actual: Any, expected: Any) -> float:
        """
        Compares actual output with expected output and returns a similarity score.

        Args:
            actual: The actual output
            expected: The expected output

        Returns:
            Float between 0 and 1 indicating similarity
        """
        if isinstance(actual, dict) and isinstance(expected, dict):
            # Compare dictionary structures
            actual_keys = set(actual.keys())
            expected_keys = set(expected.keys())

            # Calculate key overlap
            key_similarity = len(actual_keys & expected_keys) / len(
                expected_keys
            )

            # Calculate value similarity for overlapping keys
            value_scores = []
            for key in actual_keys & expected_keys:
                value_scores.append(
                    self._compare_outputs(actual[key], expected[key])
                )

            value_similarity = (
                sum(value_scores) / len(value_scores)
                if value_scores
                else 0
            )

            return (key_similarity + value_similarity) / 2

        elif isinstance(actual, (list, tuple)) and isinstance(
            expected, (list, tuple)
        ):
            # Compare sequence structures
            if len(actual) != len(expected):
                return 0.5  # Partial match for different lengths

            element_scores = [
                self._compare_outputs(a, e)
                for a, e in zip(actual, expected)
            ]
            return sum(element_scores) / len(element_scores)

        else:
            # Direct comparison for primitive types
            return float(actual == expected)


def create_test_suite(
    function_schema: Dict[str, Any], num_cases: int = 10
) -> List[Dict[str, Any]]:
    """
    Creates a test suite for a given function schema.

    Args:
        function_schema: The function schema to create tests for
        num_cases: Number of test cases to generate

    Returns:
        List of test cases
    """
    test_cases = []
    properties = function_schema["parameters"]["properties"]

    for _ in range(num_cases):
        test_case = {}
        for prop_name, prop_schema in properties.items():
            test_case[prop_name] = _generate_test_value(prop_schema)
        test_cases.append(test_case)

    return test_cases


def _generate_test_value(property_schema: Dict[str, Any]) -> Any:
    """Helper function to generate test values based on property schema"""
    schema_type = property_schema.get("type", "string")

    if schema_type == "string":
        return "test_string"
    elif schema_type == "number":
        return 42.0
    elif schema_type == "integer":
        return 42
    elif schema_type == "boolean":
        return True
    elif schema_type == "array":
        items_schema = property_schema.get(
            "items", {"type": "string"}
        )
        return [_generate_test_value(items_schema) for _ in range(2)]
    elif schema_type == "object":
        obj = {}
        for prop_name, prop_schema in property_schema.get(
            "properties", {}
        ).items():
            obj[prop_name] = _generate_test_value(prop_schema)
        return obj
    else:
        return None


# # Create base evaluator
# base_evaluator = StatisticalModelEvaluator()

# # Create function call evaluator
# func_evaluator = FunctionCallEvaluator(base_evaluator)

# # Example function schema
# schema = {
#     "name": "calculate_total",
#     "description": "Calculates total with tax",
#     "parameters": {
#         "properties": {
#             "amount": {"type": "number"},
#             "tax_rate": {"type": "number"}
#         },
#         "required": ["amount", "tax_rate"]
#     }
# }

# # Create test cases
# test_cases = create_test_suite(schema)

# # Expected outputs for test cases
# expected_outputs = [
#     {"total": 110.0},
#     {"total": 220.0},
#     # ... more expected outputs
# ]

# # Evaluate the function
# result = func_evaluator.evaluate_function_call(
#     function_schema=schema,
#     test_cases=test_cases,
#     expected_outputs=expected_outputs
# )

# print(f"Schema valid: {result.schema_valid}")
# print(f"Execution valid: {result.execution_valid}")
# print(f"Matching score: {result.matching_score}")
