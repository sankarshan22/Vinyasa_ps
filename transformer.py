import pandas as pd
import random
import os

import google.generativeai as genai


#GOOGLE_API_KEY = "AIzaSyAjCd35y22JJRu7C3wA0j0r6j7FU2GUEEs"
GOOGLE_API_KEY = "AIzaSyA_YMbw8MXOC6wtkzdUV_JQ7BiPv9DhfY0"
GOOGLE_MODEL_NAME = "gemini-1.5-flash-latest"

gemini_model = None
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(GOOGLE_MODEL_NAME)
    print(
        f"Successfully configured Google Generative AI with model: {GOOGLE_MODEL_NAME}"
    )
except Exception as e:
    print(f"Error configuring Google Generative AI or initializing model: {e}")


def classify_transformation_type_llm(source_target_pairs):
    """
    Classifies the transformation type based on source-target examples using an LLM (Google Gemini).
    """
    if not source_target_pairs or not isinstance(source_target_pairs, list):
        return "Prompt Error"
    if gemini_model is None:
        return "API Configuration Error"

    examples_str = [f"Source: '{s}' Target: '{t}'" for s, t in source_target_pairs]
    examples_str_joined = "\n".join(examples_str)
    classification_prompt = f"""Analyze the following source-target transformation examples and classify the transformation into one of these categories: String-based, Numerical, Algorithmic, General. Examples:\n{examples_str_joined}\nOutput only one word. If unsure, respond with "General".\nCategory:"""

    try:
        response = gemini_model.generate_content(
            classification_prompt.strip(),
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,  # Low temperature for consistent classification
                max_output_tokens=30,
            ),
        )
        # Ensure response has parts before accessing text
        if not response.parts or not response.text:
            print(
                "LLM response has no content for classification or response.text is empty."
            )
            return "Malformed Response"

        raw_classification = response.text.strip().lower()

        if "string" in raw_classification:
            return "String-based"
        if "numerical" in raw_classification:
            return "Numerical"
        if "algorithmic" in raw_classification:
            return "Algorithmic"
        return "General"
    except Exception as e:
        print(f"API Error (classify_transformation_type_llm): {e}")
        return "API Error"


def get_llm_transformation_function(source_target_pairs, classified_type=None):
    """
    Generates a Python transformation function based on source-target examples using an LLM (Google Gemini).
    """
    if not source_target_pairs:
        return None
    if gemini_model is None:  # Check if API was configured successfully
        return "API Configuration Error"

    examples_str = [f"Source: '{s}' Target: '{t}'" for s, t in source_target_pairs]
    examples_str_joined = "\n".join(examples_str)
    type_hint = (
        f"The user has classified this transformation as {classified_type}. "
        if classified_type
        else ""
    )
    prompt_template = f"""You are an expert data transformation engine. {type_hint}Your task is to analyze the provided source-target examples and generate a complete, executable Python function named `transform(value)`. Constraints: - The function must be named `transform`. It must take one argument: `value`. - It should return the transformed value. - Do NOT use any external libraries (e.g., re, datetime, math). - Handle invalid inputs by returning the original value or None. - Output must only be a code block between ```python and ```. Examples:\n{examples_str_joined}\nGenerate the function:\n```python"""

    try:
        response = gemini_model.generate_content(
            prompt_template.strip(),
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Higher temperature for creativity in code generation
                max_output_tokens=1024,  # Sufficient tokens for a function
            ),
        )
        if not response.parts or not response.text:
            print(
                "LLM response has no content for function generation or response.text is empty."
            )
            return None

        llm_response_content = response.text.strip()

        if "```python" in llm_response_content:
            start = llm_response_content.find("```python") + len("```python")
            end = llm_response_content.find("```", start)
            if end != -1:
                return llm_response_content[start:end].strip()
            else:
                print(
                    "Warning: Closing ``` not found in LLM response for transform function code."
                )
                return llm_response_content[start:].strip()
        return llm_response_content
    except Exception as e:
        print(f"API Error (get_llm_transformation_function): {e}")
        return None


def get_llm_transformation_function_from_prompt(user_prompt, sample_source_values):
    """
    Generates a Python transformation function based on a user's text prompt and source data samples.
    """
    if not user_prompt or not sample_source_values:
        return None
    if gemini_model is None:  # Check if API was configured successfully
        return "API Configuration Error"

    samples_str = "\n".join([f"- '{s}'" for s in sample_source_values])
    prompt_template = f"""You are an expert data transformation engine. A user wants to transform data based on the following instruction:
"{user_prompt}"

Here are some examples of the source data to help you understand the format:
{samples_str}

Based on the user's instruction and the data samples, generate a complete, executable Python function named `transform(value)`.
Constraints:
- The function must be named `transform`. It must take one argument: `value`.
- It should return the transformed value.
- Do NOT use any external libraries (e.g., re, datetime, math).
- Handle edge cases and invalid inputs gracefully by returning the original value or None.
- Your output must ONLY be a Python code block enclosed in ```python ... ```.

Generate the function:
```python
"""

    try:
        response = gemini_model.generate_content(
            prompt_template.strip(),
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, max_output_tokens=1024
            ),
        )
        if not response.parts or not response.text:
            print(
                "LLM response has no content for function generation from prompt or response.text is empty."
            )
            return None

        llm_response_content = response.text.strip()

        if "```python" in llm_response_content:
            start = llm_response_content.find("```python") + len("```python")
            end = llm_response_content.find("```", start)
            if end != -1:
                return llm_response_content[start:end].strip()
            else:
                print(
                    "Warning: Closing ``` not found in LLM response for prompt-based transform."
                )
                return llm_response_content[start:].strip()
        return llm_response_content
    except Exception as e:
        print(f"API Error (get_llm_transformation_function_from_prompt): {e}")
        return None


def create_and_execute_transform_function(function_code):
    """
    Dynamically creates and returns a callable Python function from code string.
    Executes in a restricted environment for safety.
    """
    if not function_code or not function_code.strip().startswith("def transform("):
        raise ValueError(
            "Invalid function code: Must define a function named 'transform'."
        )

    local_scope = {}
    try:
        # Define a safe subset of built-in functions and types for the `exec` environment.
        restricted_builtins = {
            "float": float,
            "int": int,
            "str": str,
            "len": len,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "None": None,
            "isinstance": isinstance,
            "list": list,
            "range": range,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "True": True,
            "False": False,
            "str.strip": str.strip,  # Added to allow basic string methods if used by LLM
            "str.replace": str.replace,
            "str.startswith": str.startswith,
            "str.endswith": str.endswith,
            "str.lower": str.lower,
            "str.upper": str.upper,
            "str.split": str.split,
            "str.join": str.join,
            "str.isdigit": str.isdigit,
        }
        restricted_globals = {"__builtins__": restricted_builtins}

        exec(function_code, restricted_globals, local_scope)

        if "transform" not in local_scope:
            raise ValueError(
                "`transform` function not defined within the provided code."
            )

        return local_scope["transform"]
    except Exception as e:
        print(f"Execution error creating/validating transform function: {e}")
        return None


def apply_transformation_to_column(dataframe, column_name, transform_function):
    """
    Applies a given transformation function to a specified column of a DataFrame.
    Handles missing columns, NaN values, and errors during transformation.
    """
    if column_name not in dataframe.columns:
        print(
            f"Warning: Column '{column_name}' not found in DataFrame. Returning a Series of NaNs."
        )
        return pd.Series(
            [pd.NA] * len(dataframe), index=dataframe.index, name=column_name
        )

    if transform_function is None:
        print(
            "Warning: No transform function provided. Returning a copy of the original column."
        )
        return dataframe[column_name].copy()

    def safe_transform(value):
        """
        Wrapper function to safely apply the transformation, handling NaNs and exceptions.
        """
        if pd.isna(value):
            return value
        try:
            # Ensure value is string before passing to transform, as LLM expects string
            return transform_function(str(value))
        except Exception:
            return value

    return dataframe[column_name].map(safe_transform)


def generate_health_report(original_series, transformed_series):
    """
    Generates a health report by comparing the original and transformed columns.
    Provides insights into transformation success, changes, data types, and nulls.
    """
    report = {}
    total_rows = len(original_series)

    if total_rows == 0:
        return {
            "success_rate": "N/A",
            "value_change_rate": "N/A",
            "new_data_type": "N/A",
            "null_values": "N/A",
            "samples": [],
        }

    successful_transforms = transformed_series.notna().sum()
    report["success_rate"] = (
        f"{(successful_transforms / total_rows) * 100:.2f}%"
        if total_rows > 0
        else "N/A"
    )

    original_str = original_series.astype(str).fillna("NaN_placeholder")
    transformed_str = transformed_series.astype(str).fillna("NaN_placeholder")

    changed_mask = original_str != transformed_str
    changed_count = changed_mask.sum()
    report["value_change_rate"] = (
        f"{(changed_count / total_rows) * 100:.2f}% ({changed_count} of {total_rows} rows)"
        if total_rows > 0
        else "N/A"
    )

    report["new_data_type"] = str(transformed_series.dtype)

    nulls_before = original_series.isnull().sum()
    nulls_after = transformed_series.isnull().sum()
    report["null_values"] = f"{nulls_before} before -> {nulls_after} after"

    # Sample selection for report
    sample_indices = []
    max_samples = 5

    # Get indices of changed and unchanged rows
    changed_indices = original_series[changed_mask].index.tolist()
    unchanged_indices = original_series[~changed_mask].index.tolist()

    # Prioritize sampling changed items
    num_changed_to_sample = min(len(changed_indices), max_samples // 2 + 1)
    sample_indices.extend(random.sample(changed_indices, num_changed_to_sample))

    # Fill remaining spots with unchanged items
    remaining_needed = max_samples - len(sample_indices)
    if remaining_needed > 0:
        # Filter out any unchanged indices that might have been picked if changed_indices was very small
        available_unchanged = [
            idx for idx in unchanged_indices if idx not in sample_indices
        ]
        sample_indices.extend(
            random.sample(
                available_unchanged, min(len(available_unchanged), remaining_needed)
            )
        )

    # Fallback if total rows < max_samples, just take all available indices
    if total_rows < max_samples and total_rows > 0:
        sample_indices = list(range(total_rows))
    elif total_rows == 0:
        sample_indices = []  # No samples for empty data

    before_after_samples = []
    for idx in sample_indices:
        before_val = original_series.loc[idx]
        after_val = transformed_series.loc[idx]
        before_after_samples.append(
            {
                "before": str(before_val) if pd.notna(before_val) else "NaN",
                "after": str(after_val) if pd.notna(after_val) else "NaN",
            }
        )
    report["samples"] = before_after_samples

    return report


if __name__ == "__main__":
    print("\n--- Testing LLM Transformation Service ---")

    if gemini_model is None:
        print("\n!!! API configuration failed. Please check GOOGLE_API_KEY. !!!")
    else:
        print("\n--- Example 1: String Capitalization (by example) ---")
        string_pairs = [("hello", "Hello"), ("world", "World")]
        function_code_1 = get_llm_transformation_function(string_pairs)
        if function_code_1 and "API Error" not in function_code_1:
            print("\nGenerated Function Code:")
            print(function_code_1)
        else:
            print(f"Failed to generate code: {function_code_1}")

        print("\n--- Example 2: Numerical Multiplication (by example) ---")
        numerical_pairs = [("5", "10"), ("1.5", "3.0")]
        function_code_2 = get_llm_transformation_function(numerical_pairs)
        if function_code_2 and "API Error" not in function_code_2:
            print("\nGenerated Function Code:")
            print(function_code_2)
        else:
            print(f"Failed to generate code: {function_code_2}")

        print("\n--- Example 3: String Transformation (by prompt) ---")
        user_prompt = "Remove the prefix 'ID-' and convert the rest to an integer."
        sample_values = ["ID-123", "ID-456", "ID-789"]
        function_code_3 = get_llm_transformation_function_from_prompt(
            user_prompt, sample_values
        )
        if function_code_3 and "API Error" not in function_code_3:
            print("\nGenerated Function Code:")
            print(function_code_3)

            try:
                transform_func = create_and_execute_transform_function(function_code_3)
                if transform_func:
                    data = {"id_column": ["ID-123", "ID-456", "bad-data", None]}
                    df = pd.DataFrame(data)
                    transformed_series = apply_transformation_to_column(
                        df, "id_column", transform_func
                    )
                    df["transformed"] = transformed_series
                    print("\nTransformed DataFrame:")
                    print(df)
                else:
                    print("Failed to create transform function.")
            except Exception as e:
                print(f"Error during function execution test: {e}")
        else:
            print(
                f"Failed to generate transformation function code. Reason: {function_code_3}"
            )

    print("\n--- End of Test ---")
