import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for, send_file, render_template
import urllib.parse
import io
import os
import transformer as transformer_service
from transformer import gemini_model
import google.generativeai as genai
from pymongo import MongoClient

app = Flask(__name__)

# Connect to your MongoDB Compass setup
client = MongoClient("mongodb://localhost:27017/")
db = client["Tabulax"]  # ← match your Compass DB
users_collection = db["users"]

uploaded_dataframes = {}
df_counter = 0
merged_df_info = None  # Store merged DataFrame info

# --- JINJA FILTER ---
def url_encode_filter(s): return urllib.parse.quote_plus(str(s))
app.jinja_env.filters['url_encode'] = url_encode_filter

# --- INTRO PAGE ---
@app.route('/')
def intro():
    return render_template("intro.html")

# --- LOGIN PAGE ---
@app.route('/login')
def login():
    return render_template("login.html")

# --- SIGNUP HANDLER ---
@app.route('/signup_user', methods=['POST'])
def signup_user():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    # Check if user exists in MongoDB
    if users_collection.find_one({"email": email}):
        return "User already exists. Please log in.", 400

    # Insert new user into MongoDB
    users_collection.insert_one({"email": email, "password": password, "name": name})
    return redirect(url_for('login'))

# --- LOGIN HANDLER ---
@app.route('/login_auth', methods=['POST'])
def login_auth():
    email = request.form['email']
    password = request.form['password']

    user = users_collection.find_one({"email": email})
    if user and user['password'] == password:
        return redirect(url_for('index'))  # index → /upload
    else:
        return "Invalid credentials. Try again.", 401

# --- FILE UPLOAD PAGE ---
@app.route('/upload')
def index():
    global df_counter, merged_df_info
    uploaded_dataframes.clear()
    df_counter = 0
    merged_df_info = None
    return render_template("upload.html")

# --- FILE UPLOAD HANDLER ---
@app.route('/upload', methods=['POST'])
def upload_files():
    global df_counter, merged_df_info
    uploaded_dataframes.clear()
    df_counter = 0
    merged_df_info = None
    files = request.files.getlist('file')
    if not files or not files[0].filename:
        return redirect(url_for('index'))
    for file in files:
        if file.filename:
            try:
                if file.filename.lower().endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.filename.lower().endswith('.json'):
                    df = pd.read_json(file)
                else:
                    continue
                df_id = str(df_counter)
                df_counter += 1
                uploaded_dataframes[df_id] = {'df': df.copy(), 'original_df': df.copy(), 'filename': file.filename}
            except Exception as e:
                return f"Error processing file {file.filename}: {e}", 400
    return redirect(url_for('define_transformations'))

# --- COLUMN SELECT PAGE ---
@app.route('/define_transformations')
def define_transformations():
    if not uploaded_dataframes:
        return redirect(url_for('index'))
    return render_template("column_select.html", files=uploaded_dataframes)

# --- SAMPLE DISPLAY ---
@app.route('/display_samples', methods=['POST'])
def display_samples():
    samples_by_file = {}
    for df_id, df_info in uploaded_dataframes.items():
        selected_columns = request.form.getlist(f'selected_columns_{df_id}')
        if not selected_columns:
            samples_by_file[df_id] = {}
            continue
        df = df_info['df']
        samples = {
            col: df[col].dropna().astype(str).sample(n=min(len(df[col].dropna()), 5)).tolist()
            for col in selected_columns
        }
        samples_by_file[df_id] = samples
    return render_template("sample_display.html", samples_by_file=samples_by_file, files=uploaded_dataframes)

# --- TARGET INPUT PAGE ---
@app.route('/provide_targets', methods=['POST'])
def provide_targets():
    columns_by_file = {}
    for df_id in uploaded_dataframes:
        cols_for_this_file = {}
        for key in request.form:
            if key.startswith(f'samples_{df_id}_'):
                col_name = urllib.parse.unquote_plus(key.replace(f'samples_{df_id}_', '', 1))
                cols_for_this_file[col_name] = urllib.parse.unquote_plus(request.form.get(key)).split('|||')
        columns_by_file[df_id] = cols_for_this_file
    return render_template("target_input.html", columns_by_file=columns_by_file, files=uploaded_dataframes)

# --- CLASSIFY TYPE USING PROMPT ---
def _classify_type_for_prompt(user_prompt, sample_source_values):
    if not user_prompt:
        return "Prompt Error"
    if gemini_model is None:
        return "API Configuration Error"
    samples_str = "\n".join([f"- '{s}'" for s in sample_source_values[:3]])
    classification_prompt = f"""Analyze the user instruction and sample data. Classify into: String-based, Numerical, Algorithmic, General.
- String-based: Text manipulation. - Numerical: Mathematical formula. - Algorithmic: Clear algorithm, no external knowledge. - General: Requires external, real-world knowledge.
Instruction: "{user_prompt}"\nSample Data: {samples_str}\nOutput one word. Category:"""
    try:
        response = gemini_model.generate_content(classification_prompt.strip(), generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=30))
        if not response.parts:
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
        print(f"API Error (classify for prompt): {e}")
        return f"API Error: {e}"

# --- REVIEW TRANSFORMATIONS ---
@app.route('/review_transformations', methods=['POST'])
def review_transformations():
    review_data_by_file, tasks_by_col = {}, {}
    for key, value in request.form.items():
        if key.startswith('input_mode_'):
            parts = key.split('_')
            df_id, col_name_encoded = parts[2], parts[3]
            col_name = urllib.parse.unquote_plus(col_name_encoded)
            if df_id not in tasks_by_col:
                tasks_by_col[df_id] = {}
            if value == 'prompt':
                tasks_by_col[df_id][col_name] = {
                    'mode': 'prompt',
                    'prompt': request.form.get(f'prompt_text_{df_id}_{col_name_encoded}'),
                    'samples': urllib.parse.unquote_plus(request.form.get(f'source_samples_{df_id}_{col_name_encoded}')).split('|||')
                }
            else:
                tasks_by_col[df_id][col_name] = {'mode': 'examples', 'examples': []}

    for key, value in request.form.items():
        if key.startswith('source_val_'):
            parts = key.split('_')
            df_id, col_name_encoded, index = parts[2], parts[3], parts[4]
            col_name = urllib.parse.unquote_plus(col_name_encoded)
            if tasks_by_col.get(df_id, {}).get(col_name, {}).get('mode') == 'examples':
                tasks_by_col[df_id][col_name]['examples'].append((urllib.parse.unquote_plus(value), request.form.get(f'target_val_{df_id}_{col_name_encoded}_{index}')))

    for df_id, tasks in tasks_by_col.items():
        review_data_by_file[df_id] = []
        for col_name, task_info in tasks.items():
            review_item = {'col_name': col_name}
            if task_info['mode'] == 'prompt':
                classified_type = _classify_type_for_prompt(task_info['prompt'], task_info['samples'])
                review_item.update({'input_method': 'prompt', 'prompt': task_info['prompt']})
            else:
                classified_type = transformer_service.classify_transformation_type_llm(task_info['examples'])
                review_item.update({'input_method': 'examples', 'examples': task_info['examples']})
            review_item['type'] = classified_type

            if classified_type == 'General':
                if task_info['mode'] == 'prompt':
                    review_item['new_column_names'] = transformer_service.get_llm_multiple_new_columns(task_info['prompt'])
                else:
                    review_item['new_column_names'] = ["New_Column_from_Examples"]
            else:
                if task_info['mode'] == 'prompt':
                    review_item['code'] = transformer_service.get_llm_transformation_function_from_prompt(task_info['prompt'], task_info['samples'])
                else:
                    review_item['code'] = transformer_service.get_llm_transformation_function(task_info['examples'], classified_type)
                review_item['code'] = review_item.get('code') or "LLM failed to generate code."
            review_data_by_file[df_id].append(review_item)

    uploaded_dataframes['master_review_data'] = review_data_by_file
    return render_template("review_tranformation.html", review_data_by_file=review_data_by_file, files=uploaded_dataframes)

# --- APPLY TRANSFORMATION ---
@app.route('/process_transformation', methods=['POST'])
def process_transformation():
    return _apply_transformations_and_render()

# --- REATTEMPT ---
@app.route('/reattempt', methods=['POST'])
def reattempt_transformation():
    df_id = request.form.get('df_id')
    if not df_id:
        return "Error: No file specified.", 400
    rules = uploaded_dataframes.get('master_review_data', {}).get(df_id, [])
    for rule in rules:
        is_failed_general = rule['type'] == 'General' and "Error" in rule.get('new_column_names', [''])[0]
        is_failed_standard = rule['type'] != 'General' and ("LLM failed" in rule.get('code', '') or "API Error" in rule.get('code', ''))
        if is_failed_general or is_failed_standard:
            print(f"Reattempting generation for column: {rule['col_name']}")
            if rule['type'] == 'General':
                if rule['input_method'] == 'prompt':
                    rule['new_column_names'] = transformer_service.get_llm_multiple_new_columns(rule['prompt'])
            else:
                if rule['input_method'] == 'prompt':
                    rule['code'] = transformer_service.get_llm_transformation_function_from_prompt(rule['prompt'], rule['samples'])
                else:
                    rule['code'] = transformer_service.get_llm_transformation_function(rule['examples'], rule['type'])
                rule['code'] = rule.get('code') or "LLM failed to generate code on reattempt."
    return _apply_transformations_and_render()

# --- APPLY TRANSFORMATIONS FUNCTION ---
def _apply_transformations_and_render():
    rules_by_file = uploaded_dataframes.get('master_review_data', {})
    if not rules_by_file:
        return "Error: No transformation rules found.", 400

    batch_results = {}
    for df_id, rules_to_apply in rules_by_file.items():
        if df_id not in uploaded_dataframes:
            continue

        df_info = uploaded_dataframes[df_id]
        current_df = df_info['original_df'].copy()
        successful_reports, errors = {}, {}

        for rule in rules_to_apply:
            col_name = rule['col_name']
            try:
                if rule['type'] == 'General':
                    new_column_names = rule.get('new_column_names', [])
                    if not new_column_names or "Error" in new_column_names[0]:
                        errors[col_name] = f"Failed to get new column names: {new_column_names[0] if new_column_names else 'None specified'}"
                        continue

                    source_values = current_df[col_name].dropna().unique().tolist()
                    batch_results_dict = transformer_service.get_llm_general_batch_enrichment(source_values, new_column_names)

                    if 'error' in batch_results_dict:
                        errors[col_name] = f"Batch API call failed: {batch_results_dict['error']}"
                        continue

                    for new_col_name in new_column_names:
                        temp_col_name = new_col_name
                        if temp_col_name in current_df.columns:
                            temp_col_name = f"{temp_col_name}_new"

                        current_df[temp_col_name] = current_df[col_name].map(lambda x: batch_results_dict.get(x, {}).get(new_col_name, ''))
                        found_count = sum(1 for v in current_df[temp_col_name] if v)
                        successful_reports[temp_col_name] = {
                            'Action': f"Added column based on '{col_name}'",
                            'Success_Rate': f"{(found_count / len(source_values)) * 100:.2f}% ({found_count}/{len(source_values)} unique values found)"
                        }
                else:
                    function_code = rule.get('code', 'LLM failed to generate code.')
                    if "LLM failed" in function_code or "API Error" in function_code:
                        errors[col_name] = function_code
                        continue
                    transform_func = transformer_service.create_and_execute_transform_function(function_code)
                    if not transform_func:
                        errors[col_name] = "Failed to create a valid function from generated code."
                        continue

                    new_column_data = transformer_service.apply_transformation_to_column(current_df, col_name, transform_func)
                    report_col_name = f"{col_name} (Transformed)"
                    successful_reports[report_col_name] = transformer_service.generate_health_report(df_info['original_df'][col_name], new_column_data)
                    current_df[col_name] = new_column_data
            except Exception as e:
                errors[col_name] = f"Runtime error: {e}"

        df_info['df'] = current_df
        batch_results[df_id] = {'filename': df_info['filename'], 'successful_reports': successful_reports, 'errors': errors}

    return render_template("result.html", batch_results=batch_results, show_merge_option=True)

# --- MERGE FILES PAGE ---
@app.route('/merge_files', methods=['POST'])
def merge_files():
    global merged_df_info
    # Merge all DataFrames row-wise (axis=0)
    dfs = [df_info['df'] for df_id, df_info in uploaded_dataframes.items() if isinstance(df_info, dict) and 'df' in df_info]
    if not dfs:
        return "No files to merge.", 400
    try:
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        merged_df_info = {
            'df': merged_df,
            'filename': 'merged_files.csv'
        }
        # Show download options for merged file
        return render_template("merged_result.html", merged_filename='merged_files.csv')
    except Exception as e:
        return f"Error merging files: {e}", 400

# --- DOWNLOAD MERGED FILE ---
@app.route('/download-merged/<file_format>')
def download_merged_file(file_format):
    global merged_df_info
    if not merged_df_info or 'df' not in merged_df_info:
        return "Merged file not found.", 404
    df = merged_df_info['df']
    download_filename = f"merged_files.{file_format}"
    buffer = io.StringIO()
    if file_format == 'csv':
        df.to_csv(buffer, index=False)
        mimetype = 'text/csv'
    elif file_format == 'json':
        df.to_json(buffer, orient='records', indent=4)
        mimetype = 'application/json'
    else:
        return "Error: Invalid file format.", 400
    buffer.seek(0)
    return send_file(io.BytesIO(buffer.getvalue().encode('utf-8')), mimetype=mimetype, as_attachment=True, download_name=download_filename)

# --- DOWNLOAD ALL AS ZIP ---
@app.route('/download-all-zip')
def download_all_zip():
    import zipfile
    global merged_df_info

    # Create in-memory ZIP
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for df_id, df_info in uploaded_dataframes.items():
            # Skip non-dataframe entries like 'master_review_data'
            if not isinstance(df_info, dict) or 'df' not in df_info:
                continue

            df = df_info['df']
            base_name, _ = os.path.splitext(df_info['filename'])

            # Save as CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            zf.writestr(f"{base_name}_transformed.csv", csv_buffer.getvalue())

            # Save as JSON
            json_buffer = io.StringIO()
            df.to_json(json_buffer, orient='records', indent=4)
            json_buffer.seek(0)
            zf.writestr(f"{base_name}_transformed.json", json_buffer.getvalue())

        # Add merged file if exists
        if merged_df_info and 'df' in merged_df_info:
            merged_df = merged_df_info['df']
            # CSV
            csv_buffer = io.StringIO()
            merged_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            zf.writestr("merged_files.csv", csv_buffer.getvalue())
            # JSON
            json_buffer = io.StringIO()
            merged_df.to_json(json_buffer, orient='records', indent=4)
            json_buffer.seek(0)
            zf.writestr("merged_files.json", json_buffer.getvalue())

    memory_file.seek(0)
    return send_file(memory_file,
                     mimetype='application/zip',
                     download_name='all_transformed_files.zip',
                     as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# import pandas as pd
# from flask import Flask, request, render_template_string, redirect, url_for, send_file, render_template
# import urllib.parse
# import io
# import os
# import transformer as transformer_service
# from transformer import gemini_model
# import google.generativeai as genai

# app = Flask(__name__)

# # --- IN-MEMORY USER DICTIONARY ---
# users = {}  # Format: {email: {"password": ..., "name": ...}}

# uploaded_dataframes = {}
# df_counter = 0

# # --- JINJA FILTER ---
# def url_encode_filter(s): return urllib.parse.quote_plus(str(s))
# app.jinja_env.filters['url_encode'] = url_encode_filter

# # --- INTRO PAGE ---
# @app.route('/')
# def intro():
#     return render_template("intro.html")

# # --- LOGIN PAGE ---
# @app.route('/login')
# def login():
#     return render_template("login.html")

# # --- SIGNUP HANDLER ---
# @app.route('/signup_user', methods=['POST'])
# def signup_user():
#     name = request.form['name']
#     email = request.form['email']
#     password = request.form['password']

#     if email in users:
#         return "User already exists. Please log in.", 400

#     users[email] = {"password": password, "name": name}
#     return redirect(url_for('login'))

# # --- LOGIN HANDLER ---
# @app.route('/login_auth', methods=['POST'])
# def login_auth():
#     email = request.form['email']
#     password = request.form['password']

#     user = users.get(email)
#     if user and user['password'] == password:
#         return redirect(url_for('index'))  # index → /upload
#     else:
#         return "Invalid credentials. Try again.", 401

# # --- FILE UPLOAD PAGE ---
# @app.route('/upload')
# def index():
#     global df_counter
#     uploaded_dataframes.clear()
#     df_counter = 0
#     return render_template("upload.html")

# # --- FILE UPLOAD HANDLER ---
# @app.route('/upload', methods=['POST'])
# def upload_files():
#     global df_counter
#     uploaded_dataframes.clear()
#     df_counter = 0
#     files = request.files.getlist('file')
#     if not files or not files[0].filename:
#         return redirect(url_for('index'))
#     for file in files:
#         if file.filename:
#             try:
#                 if file.filename.lower().endswith('.csv'):
#                     df = pd.read_csv(file)
#                 elif file.filename.lower().endswith('.json'):
#                     df = pd.read_json(file)
#                 else:
#                     continue
#                 df_id = str(df_counter)
#                 df_counter += 1
#                 uploaded_dataframes[df_id] = {'df': df.copy(), 'original_df': df.copy(), 'filename': file.filename}
#             except Exception as e:
#                 return f"Error processing file {file.filename}: {e}", 400
#     return redirect(url_for('define_transformations'))

# # --- COLUMN SELECT PAGE ---
# @app.route('/define_transformations')
# def define_transformations():
#     if not uploaded_dataframes:
#         return redirect(url_for('index'))
#     return render_template("column_select.html", files=uploaded_dataframes)

# # --- SAMPLE DISPLAY ---
# @app.route('/display_samples', methods=['POST'])
# def display_samples():
#     samples_by_file = {}
#     for df_id, df_info in uploaded_dataframes.items():
#         selected_columns = request.form.getlist(f'selected_columns_{df_id}')
#         if not selected_columns:
#             samples_by_file[df_id] = {}
#             continue
#         df = df_info['df']
#         samples = {
#             col: df[col].dropna().astype(str).sample(n=min(len(df[col].dropna()), 5)).tolist()
#             for col in selected_columns
#         }
#         samples_by_file[df_id] = samples
#     return render_template("sample_display.html", samples_by_file=samples_by_file, files=uploaded_dataframes)

# # --- TARGET INPUT PAGE ---
# @app.route('/provide_targets', methods=['POST'])
# def provide_targets():
#     columns_by_file = {}
#     for df_id in uploaded_dataframes:
#         cols_for_this_file = {}
#         for key in request.form:
#             if key.startswith(f'samples_{df_id}_'):
#                 col_name = urllib.parse.unquote_plus(key.replace(f'samples_{df_id}_', '', 1))
#                 cols_for_this_file[col_name] = urllib.parse.unquote_plus(request.form.get(key)).split('|||')
#         columns_by_file[df_id] = cols_for_this_file
#     return render_template("target_input.html", columns_by_file=columns_by_file, files=uploaded_dataframes)

# # --- CLASSIFY TYPE USING PROMPT ---
# def _classify_type_for_prompt(user_prompt, sample_source_values):
#     if not user_prompt:
#         return "Prompt Error"
#     if gemini_model is None:
#         return "API Configuration Error"
#     samples_str = "\n".join([f"- '{s}'" for s in sample_source_values[:3]])
#     classification_prompt = f"""Analyze the user instruction and sample data. Classify into: String-based, Numerical, Algorithmic, General.
# - String-based: Text manipulation. - Numerical: Mathematical formula. - Algorithmic: Clear algorithm, no external knowledge. - General: Requires external, real-world knowledge.
# Instruction: "{user_prompt}"\nSample Data: {samples_str}\nOutput one word. Category:"""
#     try:
#         response = gemini_model.generate_content(classification_prompt.strip(), generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=30))
#         if not response.parts:
#             return "Malformed Response"
#         raw_classification = response.text.strip().lower()
#         if "string" in raw_classification:
#             return "String-based"
#         if "numerical" in raw_classification:
#             return "Numerical"
#         if "algorithmic" in raw_classification:
#             return "Algorithmic"
#         return "General"
#     except Exception as e:
#         print(f"API Error (classify for prompt): {e}")
#         return f"API Error: {e}"

# # --- REVIEW TRANSFORMATIONS ---
# @app.route('/review_transformations', methods=['POST'])
# def review_transformations():
#     review_data_by_file, tasks_by_col = {}, {}
#     for key, value in request.form.items():
#         if key.startswith('input_mode_'):
#             parts = key.split('_')
#             df_id, col_name_encoded = parts[2], parts[3]
#             col_name = urllib.parse.unquote_plus(col_name_encoded)
#             if df_id not in tasks_by_col:
#                 tasks_by_col[df_id] = {}
#             if value == 'prompt':
#                 tasks_by_col[df_id][col_name] = {
#                     'mode': 'prompt',
#                     'prompt': request.form.get(f'prompt_text_{df_id}_{col_name_encoded}'),
#                     'samples': urllib.parse.unquote_plus(request.form.get(f'source_samples_{df_id}_{col_name_encoded}')).split('|||')
#                 }
#             else:
#                 tasks_by_col[df_id][col_name] = {'mode': 'examples', 'examples': []}

#     for key, value in request.form.items():
#         if key.startswith('source_val_'):
#             parts = key.split('_')
#             df_id, col_name_encoded, index = parts[2], parts[3], parts[4]
#             col_name = urllib.parse.unquote_plus(col_name_encoded)
#             if tasks_by_col.get(df_id, {}).get(col_name, {}).get('mode') == 'examples':
#                 tasks_by_col[df_id][col_name]['examples'].append((urllib.parse.unquote_plus(value), request.form.get(f'target_val_{df_id}_{col_name_encoded}_{index}')))

#     for df_id, tasks in tasks_by_col.items():
#         review_data_by_file[df_id] = []
#         for col_name, task_info in tasks.items():
#             review_item = {'col_name': col_name}
#             if task_info['mode'] == 'prompt':
#                 classified_type = _classify_type_for_prompt(task_info['prompt'], task_info['samples'])
#                 review_item.update({'input_method': 'prompt', 'prompt': task_info['prompt']})
#             else:
#                 classified_type = transformer_service.classify_transformation_type_llm(task_info['examples'])
#                 review_item.update({'input_method': 'examples', 'examples': task_info['examples']})
#             review_item['type'] = classified_type

#             if classified_type == 'General':
#                 if task_info['mode'] == 'prompt':
#                     review_item['new_column_names'] = transformer_service.get_llm_multiple_new_columns(task_info['prompt'])
#                 else:
#                     review_item['new_column_names'] = ["New_Column_from_Examples"]
#             else:
#                 if task_info['mode'] == 'prompt':
#                     review_item['code'] = transformer_service.get_llm_transformation_function_from_prompt(task_info['prompt'], task_info['samples'])
#                 else:
#                     review_item['code'] = transformer_service.get_llm_transformation_function(task_info['examples'], classified_type)
#                 review_item['code'] = review_item.get('code') or "LLM failed to generate code."
#             review_data_by_file[df_id].append(review_item)

#     uploaded_dataframes['master_review_data'] = review_data_by_file
#     return render_template("review_tranformation.html", review_data_by_file=review_data_by_file, files=uploaded_dataframes)

# # --- APPLY TRANSFORMATION ---
# @app.route('/process_transformation', methods=['POST'])
# def process_transformation():
#     return _apply_transformations_and_render()

# # --- REATTEMPT ---
# @app.route('/reattempt', methods=['POST'])
# def reattempt_transformation():
#     df_id = request.form.get('df_id')
#     if not df_id:
#         return "Error: No file specified.", 400
#     rules = uploaded_dataframes.get('master_review_data', {}).get(df_id, [])
#     for rule in rules:
#         is_failed_general = rule['type'] == 'General' and "Error" in rule.get('new_column_names', [''])[0]
#         is_failed_standard = rule['type'] != 'General' and ("LLM failed" in rule.get('code', '') or "API Error" in rule.get('code', ''))
#         if is_failed_general or is_failed_standard:
#             print(f"Reattempting generation for column: {rule['col_name']}")
#             if rule['type'] == 'General':
#                 if rule['input_method'] == 'prompt':
#                     rule['new_column_names'] = transformer_service.get_llm_multiple_new_columns(rule['prompt'])
#             else:
#                 if rule['input_method'] == 'prompt':
#                     rule['code'] = transformer_service.get_llm_transformation_function_from_prompt(rule['prompt'], rule['samples'])
#                 else:
#                     rule['code'] = transformer_service.get_llm_transformation_function(rule['examples'], rule['type'])
#                 rule['code'] = rule.get('code') or "LLM failed to generate code on reattempt."
#     return _apply_transformations_and_render()

# # --- APPLY TRANSFORMATIONS FUNCTION ---
# def _apply_transformations_and_render():
#     rules_by_file = uploaded_dataframes.get('master_review_data', {})
#     if not rules_by_file:
#         return "Error: No transformation rules found.", 400

#     batch_results = {}
#     for df_id, rules_to_apply in rules_by_file.items():
#         if df_id not in uploaded_dataframes:
#             continue

#         df_info = uploaded_dataframes[df_id]
#         current_df = df_info['original_df'].copy()
#         successful_reports, errors = {}, {}

#         for rule in rules_to_apply:
#             col_name = rule['col_name']
#             try:
#                 if rule['type'] == 'General':
#                     new_column_names = rule.get('new_column_names', [])
#                     if not new_column_names or "Error" in new_column_names[0]:
#                         errors[col_name] = f"Failed to get new column names: {new_column_names[0] if new_column_names else 'None specified'}"
#                         continue

#                     source_values = current_df[col_name].dropna().unique().tolist()
#                     batch_results_dict = transformer_service.get_llm_general_batch_enrichment(source_values, new_column_names)

#                     if 'error' in batch_results_dict:
#                         errors[col_name] = f"Batch API call failed: {batch_results_dict['error']}"
#                         continue

#                     for new_col_name in new_column_names:
#                         temp_col_name = new_col_name
#                         if temp_col_name in current_df.columns:
#                             temp_col_name = f"{temp_col_name}_new"

#                         current_df[temp_col_name] = current_df[col_name].map(lambda x: batch_results_dict.get(x, {}).get(new_col_name, ''))
#                         found_count = sum(1 for v in current_df[temp_col_name] if v)
#                         successful_reports[temp_col_name] = {
#                             'Action': f"Added column based on '{col_name}'",
#                             'Success_Rate': f"{(found_count / len(source_values)) * 100:.2f}% ({found_count}/{len(source_values)} unique values found)"
#                         }
#                 else:
#                     function_code = rule.get('code', 'LLM failed to generate code.')
#                     if "LLM failed" in function_code or "API Error" in function_code:
#                         errors[col_name] = function_code
#                         continue
#                     transform_func = transformer_service.create_and_execute_transform_function(function_code)
#                     if not transform_func:
#                         errors[col_name] = "Failed to create a valid function from generated code."
#                         continue

#                     new_column_data = transformer_service.apply_transformation_to_column(current_df, col_name, transform_func)
#                     report_col_name = f"{col_name} (Transformed)"
#                     successful_reports[report_col_name] = transformer_service.generate_health_report(df_info['original_df'][col_name], new_column_data)
#                     current_df[col_name] = new_column_data
#             except Exception as e:
#                 errors[col_name] = f"Runtime error: {e}"

#         df_info['df'] = current_df
#         batch_results[df_id] = {'filename': df_info['filename'], 'successful_reports': successful_reports, 'errors': errors}

#     return render_template("result.html", batch_results=batch_results)

# # --- DOWNLOAD FILE ---
# @app.route('/download/<df_id>/<file_format>')
# def download_file(df_id, file_format):
#     if df_id not in uploaded_dataframes:
#         return "Error: DataFrame not found.", 404
#     df_info = uploaded_dataframes[df_id]
#     df = df_info['df']
#     base_name, _ = os.path.splitext(df_info['filename'])
#     download_filename = f"{base_name}_transformed.{file_format}"
#     buffer = io.StringIO()
#     if file_format == 'csv':
#         df.to_csv(buffer, index=False)
#         mimetype = 'text/csv'
#     elif file_format == 'json':
#         df.to_json(buffer, orient='records', indent=4)
#         mimetype = 'application/json'
#     else:
#         return "Error: Invalid file format.", 400
#     buffer.seek(0)
#     return send_file(io.BytesIO(buffer.getvalue().encode('utf-8')), mimetype=mimetype, as_attachment=True, download_name=download_filename)

# # --- DOWNLOAD ALL AS ZIP ---
# @app.route('/download-all-zip')
# def download_all_zip():
#     import zipfile

#     # Create in-memory ZIP
#     memory_file = io.BytesIO()
#     with zipfile.ZipFile(memory_file, 'w') as zf:
#         for df_id, df_info in uploaded_dataframes.items():
#             # Skip non-dataframe entries like 'master_review_data'
#             if not isinstance(df_info, dict) or 'df' not in df_info:
#                 continue

#             df = df_info['df']
#             base_name, _ = os.path.splitext(df_info['filename'])

#             # Save as CSV
#             csv_buffer = io.StringIO()
#             df.to_csv(csv_buffer, index=False)
#             csv_buffer.seek(0)
#             zf.writestr(f"{base_name}_transformed.csv", csv_buffer.getvalue())

#             # Save as JSON
#             json_buffer = io.StringIO()
#             df.to_json(json_buffer, orient='records', indent=4)
#             json_buffer.seek(0)
#             zf.writestr(f"{base_name}_transformed.json", json_buffer.getvalue())

#     memory_file.seek(0)
#     return send_file(memory_file,
#                      mimetype='application/zip',
#                      download_name='all_transformed_files.zip',
#                      as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)