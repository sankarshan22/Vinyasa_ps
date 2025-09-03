# Vinayasa: Tabular Transformation Service

**Vinayasa** (from the Sanskrit root *nyāsa*, meaning placement or arrangement, with the prefix *vi-* denoting "special" or "careful") can be interpreted as "careful arrangement" or "systematic placement." In yoga, *vinayasa* refers to a flowing, ordered sequence—just like how this framework brings structure to messy tabular data through transformations.

This project provides a **Flask-based web application and backend framework** for performing tabular data transformations. It implements ideas from the [TabulaX research paper](https://arxiv.org/html/2411.17110v1) and uses **Google Gemini** models to generate interpretable Python transformation functions. Users can upload data, define or sample transformations, and review results through a web interface.

---

## Features

* **Web interface (Flask app)** with HTML templates for upload, review, history, and results.
* **LLM-based classification** of transformations (`String-based`, `Numerical`, `Algorithmic`, `General`).
* **Generate Python functions** from example pairs or natural language instructions.
* **Safe execution environment** for generated functions.
* **Integration with Pandas DataFrames** for applying transformations.
* **Health reports** that measure transformation success, change rate, null handling, and sample examples.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/vinayasa.git
cd vinayasa
pip install -r requirements.txt
```

### Requirements

Dependencies are listed in [requirements.txt](./requirements.txt):

* pandas
* Flask
* google-generativeai
* pymongo
* gunicorn

---

## Configuration

Before running the app, configure your **Google API key**:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

The app uses the model:

```python
GOOGLE_MODEL_NAME = "gemini-1.5-flash-latest"
```

---

## Running the Application

Start the Flask app with:

```bash
python app.py
```

By default, the application runs at:
`http://127.0.0.1:5000/`

You can then:

* Upload source and target data.
* Define transformation examples or prompts.
* Review generated functions.
* Inspect transformation results and health reports.

---

## Usage Examples (Backend API)

You can also use the backend functions directly in Python scripts.

### Generate a function from examples

```python
from transformer import get_llm_transformation_function, create_and_execute_transform_function

pairs = [("hello", "Hello"), ("world", "World")]
code = get_llm_transformation_function(pairs)
transform = create_and_execute_transform_function(code)

print(transform("hello"))  # Output: Hello
```

### Generate a function from a user prompt

```python
from transformer import get_llm_transformation_function_from_prompt, create_and_execute_transform_function

prompt = "Remove the prefix 'ID-'"
examples = ["ID-123", "ID-456"]

code = get_llm_transformation_function_from_prompt(prompt, examples)
transform = create_and_execute_transform_function(code)

print(transform("ID-123"))  # Output: 123
```

---

## Evaluation

Transformations are evaluated using a **health report system** that provides:

* **Success Rate**: rows successfully transformed
* **Value Change Rate**: proportion of values modified
* **Data Type Tracking**: resulting column type
* **Null Handling**: null counts before and after
* **Before/After Samples**: quick examples of transformation results

---

## Project Structure

```
├── app.py                   # Flask application entry point
├── transformer.py           # Core backend transformation logic
├── requirements.txt         # Dependencies
├── Tablaux.pdf              # Research paper for background
├── templates/               # HTML templates for the web UI
│   ├── column_select.html
│   ├── history.html
│   ├── intro.html
│   ├── login.html
│   ├── merged_result.html
│   ├── result.html
│   ├── review_transformation.html
│   ├── sample_display.html
│   ├── target_input.html
│   └── upload.html
└── README.md

