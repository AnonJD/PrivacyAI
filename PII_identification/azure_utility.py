# azure_utility.py

import requests
import time
import json
import os

def send_pii_detection_request(text_content, document_id, api_key, api_endpoint, api_version="2023-04-15-preview", show_stats="True"):
    """
    Sends a PII detection request for a single document.

    Parameters:
        text_content (str): The text content.
        document_id (str): Unique identifier for the text.
        api_key (str): Azure subscription key.
        api_endpoint (str): The endpoint URL for the API (should include 'language/analyze-text/jobs').
        api_version (str): API version.
        show_stats (str): Whether to include document and transaction counts in the response.

    Returns:
        str: Operation location if the request is accepted, None otherwise.
    """
    documents = [{
        "id": str(document_id),
        "language": "en",
        "text": text_content
    }]

    payload = {
        "displayName": f"PII Detection Task for {document_id}",
        "analysisInput": {
            "documents": documents
        },
        "tasks": [
            {
                "kind": "PiiEntityRecognition",
                "taskName": f"PII Detection Task {document_id}",
                "parameters": {
                    "model-version": "latest",
                    "piiCategories": ["Person", "URL", "Email", "PhoneNumber"]
                }
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": api_key
    }

    try:
        response = requests.post(
            f"{api_endpoint}?api-version={api_version}&showStats={show_stats}",
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 202:
            operation_location = response.headers.get("operation-location")
            print(f"Request for document ID {document_id} accepted.")
            print(f"Operation Location: {operation_location}")
            return operation_location
        else:
            print(f"Failed to submit request for document ID {document_id}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception occurred while sending request for document ID {document_id}: {str(e)}")
        return None

def save_table(table, file_path):
    """
    Saves a table (dictionary) to a JSON file.

    Parameters:
        table (dict): The table to save.
        file_path (str): The path where the table will be saved.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(table, f, ensure_ascii=False, indent=4)
    print(f"Table saved to {file_path}")

def load_table(file_path):
    """
    Loads a table (dictionary) from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded table.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        table = json.load(f)
    print(f"Table loaded from {file_path}")
    return table

def check_all_jobs_finished(operation_table, api_key):
    """
    Checks if all jobs in the operation table are finished.

    Parameters:
        operation_table (dict): Mapping from document IDs to operation locations.
        api_key (str): Azure subscription key.

    Returns:
        bool: True if all jobs are finished, False otherwise.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    all_finished = True
    for doc_id, operation_location in operation_table.items():
        try:
            response = requests.get(operation_location, headers=headers)
            if response.status_code == 200:
                job_status = response.json().get("status")
                print(f"Job Status for Document ID {doc_id}: {job_status}")

                if job_status in ["notStarted", "running"]:
                    all_finished = False
            else:
                print(f"Failed to get job status for Document ID {doc_id}. Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                all_finished = False
        except Exception as e:
            print(f"Exception occurred while checking job status for Document ID {doc_id}: {str(e)}")
            all_finished = False

    return all_finished

def retrieve_all_results(operation_table, api_key):
    """
    Retrieves results for all completed jobs in the operation table.

    Parameters:
        operation_table (dict): Mapping from document IDs to operation locations.
        api_key (str): Azure subscription key.

    Returns:
        dict: Mapping from document IDs to their processed results.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    results_table = {}
    for doc_id, operation_location in operation_table.items():
        try:
            response = requests.get(operation_location, headers=headers)
            if response.status_code == 200:
                job_results = response.json()
                entities = process_results(job_results, doc_id)
                results_table[doc_id] = entities
                print(f"Results retrieved for Document ID {doc_id}.")
            else:
                print(f"Failed to retrieve results for Document ID {doc_id}. Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                results_table[doc_id] = None
        except Exception as e:
            print(f"Exception occurred while retrieving results for Document ID {doc_id}: {str(e)}")
            results_table[doc_id] = None

    return results_table

def process_results(job_results, doc_id):
    """
    Processes the job results and structures them into a list of entities.

    Parameters:
        job_results (dict): JSON response from the job results.
        doc_id (str): The document ID corresponding to the results.

    Returns:
        list: List of entities with fields [entity_text, type, positions].
    """
    processed_data = []

    tasks = job_results.get("tasks", {})
    items = tasks.get("items", [])

    for task in items:
        results = task.get("results", {})
        documents = results.get("documents", [])

        for doc in documents:
            if doc.get("id") == doc_id:
                entities = doc.get("entities", [])

                for entity in entities:
                    entity_text = entity.get("text")
                    entity_type = entity.get("category")
                    offset = entity.get("offset")
                    length = entity.get("length")
                    positions = (offset, offset + length)

                    processed_data.append({
                        "entity_text": entity_text,
                        "type": entity_type,
                        "positions": positions
                    })
    return processed_data

def process_texts(text_list, identifier_list, storage_directory):
    # Load API credentials
    with open('input/secret.json', 'r', encoding='UTF-8') as file:
        secret_json = json.load(file)
    API_KEY = secret_json['LANGUAGE_KEY']
    API_ENDPOINT = secret_json['LANGUAGE_ENDPOINT'] + 'language/analyze-text/jobs'

    if not text_list or not identifier_list:
        print("No texts or identifiers were provided. Exiting.")
        return

    if len(text_list) != len(identifier_list):
        print("The number of texts and identifiers must be the same.")
        return

    if not os.path.exists(storage_directory):
        os.makedirs(storage_directory)
        print(f"Directory '{storage_directory}' created.")

    # Initialize rate limiting variables
    max_requests_per_second = 100
    max_requests_per_minute = 1000
    requests_in_current_second = 0
    requests_in_current_minute = 0
    second_start_time = time.time()
    minute_start_time = time.time()

    operation_table = {}
    unique_identifier = os.path.basename(storage_directory)

    total_texts = len(text_list)

    for idx, (text_content, document_id) in enumerate(zip(text_list, identifier_list)):
        current_time = time.time()

        # Check per-minute rate limit
        elapsed_minute = current_time - minute_start_time
        if elapsed_minute >= 60:
            # Reset minute counters
            requests_in_current_minute = 0
            minute_start_time = current_time
        elif requests_in_current_minute + 1 > max_requests_per_minute:
            # Sleep until the next minute
            sleep_time = 60 - elapsed_minute
            print(f"Reached per-minute rate limit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            # Reset minute counters after sleep
            requests_in_current_minute = 0
            minute_start_time = time.time()

        # Check per-second rate limit
        elapsed_second = current_time - second_start_time
        if elapsed_second >= 1:
            # Reset second counters
            requests_in_current_second = 0
            second_start_time = current_time
        elif requests_in_current_second + 1 > max_requests_per_second:
            # Sleep until the next second
            sleep_time = 1 - elapsed_second
            print(f"Reached per-second rate limit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            # Reset second counters after sleep
            requests_in_current_second = 0
            second_start_time = time.time()

        # Send PII detection request for the current text
        operation_location = send_pii_detection_request(
            text_content=text_content,
            document_id=document_id,
            api_key=API_KEY,
            api_endpoint=API_ENDPOINT
        )

        if operation_location is None:
            print(f"Failed to submit request for document ID {document_id}.")
            continue

        # Update operation table
        operation_table[document_id] = operation_location

        # Update rate limiting counters
        requests_in_current_second += 1
        requests_in_current_minute += 1

        # Sleep briefly to avoid hitting the per-second limit in tight loops
        time.sleep(0.01)  # Sleep for 10 milliseconds

    # Save operation table
    operation_table_path = os.path.join(storage_directory, f"{unique_identifier}_operation_table.json")
    save_table(operation_table, operation_table_path)

    # Save text_list and identifier_list for later use
    text_list_path = os.path.join(storage_directory, f"{unique_identifier}_text_list.json")
    save_table(text_list, text_list_path)

    identifier_list_path = os.path.join(storage_directory, f"{unique_identifier}_identifier_list.json")
    save_table(identifier_list, identifier_list_path)

    print(f"Operation table, text list, and identifier list saved to {storage_directory}")
    return operation_table_path

def check_jobs_and_retrieve_results(storage_directory):
    # Load API credentials
    with open('input/secret.json', 'r', encoding='UTF-8') as file:
        secret_json = json.load(file)
    API_KEY = secret_json['LANGUAGE_KEY']

    unique_identifier = os.path.basename(storage_directory)
    operation_table_path = os.path.join(storage_directory, f"{unique_identifier}_operation_table.json")
    if not os.path.exists(operation_table_path):
        print(f"Operation table not found at {operation_table_path}")
        return

    # Load operation table
    operation_table = load_table(operation_table_path)
    all_finished = check_all_jobs_finished(operation_table, API_KEY)
    if not all_finished:
        print("Some jobs are still in progress. Please try again later.")
        return

    # Retrieve results
    results_table = retrieve_all_results(operation_table, API_KEY)

    # Load text_list and identifier_list
    text_list_path = os.path.join(storage_directory, f"{unique_identifier}_text_list.json")
    identifier_list_path = os.path.join(storage_directory, f"{unique_identifier}_identifier_list.json")

    if os.path.exists(text_list_path) and os.path.exists(identifier_list_path):
        text_list = load_table(text_list_path)
        identifier_list = load_table(identifier_list_path)
    else:
        print(f"Text list or identifier list not found in {storage_directory}")
        return

    # Map identifiers to their corresponding texts and labels
    text_label_table = {}
    for doc_id, entities in results_table.items():
        index = identifier_list.index(doc_id)
        text = text_list[index]
        text_label_table[doc_id] = {
            "text": text,
            "entities": entities
        }

    # Save the results
    results_table_path = os.path.join(storage_directory, f"{unique_identifier}_results.json")
    save_table(text_label_table, results_table_path)
    print(f"Results saved to {results_table_path}")
    return results_table_path


def extract_entities_azure(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    result = []
    for doc_id, content in data.items():
        cleaned_doc_id = doc_id.lstrip("doc")
        for entity in content.get("entities", []):
            if entity["type"] in {"Person", "NAME_STUDENT"}:
                pass
            entity_text = entity["entity_text"]
            positions = tuple(entity["positions"])
            result.append([cleaned_doc_id, entity_text, entity['type'], str(positions)])
    
    return result

def extract_entities_azure_tscc(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    result = []
    for doc_id, content in data.items():
        cleaned_doc_id = doc_id.lstrip("doc")
        for entity in content.get("entities", []):
            if entity["type"] in {"Person", "NAME_STUDENT"}:
                entity_text = entity["entity_text"]
                positions = tuple(entity["positions"])
                result.append([cleaned_doc_id, entity_text, entity['type'], str(positions)])
    
    return result