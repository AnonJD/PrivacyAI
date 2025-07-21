from openai import OpenAI
import re
import json
import os
from typing import List
import time

with open('input/secret.json', 'r', encoding='UTF-8') as file:
        secret_json = json.load(file)
API_KEY = secret_json['OPENAI_API_KEY']
client = OpenAI(api_key =API_KEY)
FULL_TEXT = 0
D = {'NAME_STUDENT': ('@@@','###'), 
      'URL_PERSONAL':('&&&','$$$'),
      'EMAIL':('QQQ','^^^'),
      'PHONE_NUM':(r'%%%',r'~~~'),
     }
PATTERN = r'@@@(.*?)###|&&&(.*?)\$\$\$|QQQ(.*?)\^\^\^|%%%(.*?)~~~'

IndexTable =[key for key in D.keys()]

def count_special_token(text:str)->int:
    #params: text: string of annotated text
    #output: number of special tokens(in num_chars) in the input text, int
    acc = 0
    for val in D.values():
 
        tok1,tok2 = val
        #print(f'{tok1} count:',text.count(tok1))
        #print(f'{tok2} count:',text.count(tok2))
        acc += text.count(tok1)*3 + text.count(tok2)*3
    return acc

def parse_return(text:str): 
    #param: string of annotated text
    #output: string * string * int * int list, string of entity name, string of entity type, int * int of start and end index 

    matches = re.finditer(PATTERN, text)
    extracted_matches = []
    #extract all the matched sub-string
    
    for match in matches:
        for i in range(1,len(match.groups())+1):
            match_text = match.group(i)

            if match_text != None:
                label = IndexTable[i-1]
                offset = count_special_token(text[:match.start(i)])
                start_index = match.start(i) - offset
                end_index = match.end(i) - offset

                extracted_matches.append([match_text, label,start_index, end_index])
    return extracted_matches 


def get_gpt_response(input_text:str) -> str:
    #param: string of text to be annotated 
    #out: string of text annotated 

    system_prompt = 'You are an expert in labeling personally identifiable information'

    ins_prompt = 'Label the entity of the following text: @@@,### to label student name; &&&,$$$ to label personal URL; QQQ,^^^ to label personal email; %%%,~~~ to label phone number\n'

    response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:cmu-plus::A7fVfDc8",
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": ins_prompt + input_text}],
    temperature = 0)
    return response.choices[0].message.content 


def make_batch(text_list:List[str], identifier_list:List[str],store_directory:str)-> None: 
    system_prompt = 'You are an expert in labeling personally identifiable information'
    ins_prompt = 'Label the entity of the following text: @@@,### to label name\n'
    model="ft:gpt-4o-mini-2024-07-18:cmu-plus::A7fVfDc8"
    try:
        os.makedirs(store_directory, exist_ok=True)
        print(f"Directory '{store_directory}' is ready.")
    except Exception as e:
        print(f"Error creating directory '{store_directory}': {e}")
    res = []
    for i in range(len(text_list)): 
        request = {"custom_id": f'{identifier_list[i]}', "method": "POST", 
                  "url": "/v1/chat/completions", 
                  "body": {"model": model, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": ins_prompt + text_list[i]}],
                  "temperature":0}}
        res.append(request)
    
    store_directory += '/batch_request.jsonl'
    with open(store_directory, 'w',encoding='utf-8') as f:
        for entry in res:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')


def submit_finetune_job(store_directory:str)->None: 
    batch_directory = store_directory + '/batch_request.jsonl'

    batch_input_file = client.files.create(
    file=open(batch_directory, "rb"),
    purpose="batch")
    time.sleep(1)

    batch_input_file_id = batch_input_file.id

    response = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "nightly eval job"
    })

    file_path = store_directory +"/batch_response.json"
    id = response.id
    response = {'id': id}
    # Write the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(response, json_file, indent=4)


def check_progress(store_directory:str): 
    with open(store_directory + "/batch_response.json" , "r") as json_file:
        response_data = json.load(json_file)
    id = response_data['id']
    print(result:=client.batches.retrieve(id))
    if result.status == 'completed': 
        return result.output_file_id
    else: 
        return None

def save_file_response_as_jsonl(file_response, store_directory:str)->None:
    """
    Saves the content of a file_response to a JSONL file.

    Parameters:
    - file_response: The response object containing the file content.
    - output_file_path (str): The path where the JSONL file will be saved.
    """
    try:
        lines = file_response.text.strip().split('\n')
        output_file_path = store_directory + '/result.jsonl'

        with open(output_file_path, 'w', encoding='utf-8') as file:
            for line_number, line in enumerate(lines, start=1):
                if line.strip() == '':
                    continue
                try:
                    json_obj = json.loads(line)
                    json_line = json.dumps(json_obj)
                    file.write(json_line + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_number} is not valid JSON and was skipped.")
        print(f"JSONL file successfully written to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_jsonl_file(store_directory:str)->List[str]:

    data_list = []
    file_path = store_directory + '/result.jsonl'

    try:
        # Open the JSONL file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read each line and parse it as a JSON object
            for line in file:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line.strip())
                    data_list.append(json_obj)  # Append the JSON object to the list
        print(f"Successfully read {len(data_list)} entries from {file_path}.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return data_list


def resort_data(data,identifier_list): 
    new_data = []
    for each in identifier_list: 
        for l in data: 
            if l["custom_id"] == each: 
                l = (each,l['response']['body']['choices'][0]['message']['content']) 
                new_data.append(l)

    return new_data 


def add_document(
    main_dict,
    identifier: str,
    text: str,
    entities
) -> None:
    """
    Adds a document with its text and entities to the main dictionary.

    Parameters:
    - main_dict (Dict[str, Dict]): The main dictionary holding all documents.
    - identifier (str): A unique identifier for the document (e.g., "doc1").
    - text (str): The main text of the document.
    - entities (List[List]): A list of entities, each represented as
      [entity_text (str), entity_type (str), start_pos (int), end_pos (int)].

    Returns:
    - None: The function updates the main_dict in place.
    """
    main_dict[identifier] = {
        "text": text,
        "entities": [
            {
                "entity_text": entity[0],
                "type": entity[1],
                "positions": [
                    entity[2],
                    entity[3]
                ]
            }
            for entity in entities
        ]
    }


def batch_extract_parse_store(text_list,identifier_list,directory):
    res = dict()
    for ((lb,text),label) in zip(text_list, identifier_list): 
        assert lb == label
        response = text
        entities = parse_return(response)
        add_document(res,label,text,entities)
    file_path = os.path.join(directory, "gpt_result.json")
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' is ready.")
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")
        return
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(res, json_file, indent=4)
        print(f"JSON data successfully saved to '{file_path}'.")
    except Exception as e:
        print(f"Error saving JSON to '{file_path}': {e}")


def extract_parse_store(text_list,identifier_list,directory):
    res = dict()
    for (text,label) in zip(text_list, identifier_list): 
        response = get_gpt_response(text)
        entities = parse_return(response)
        add_document(res,label,text,entities)
    file_path = os.path.join(directory, "gpt_result.json")
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' is ready.")
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")
        return
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(res, json_file, indent=4)
        print(f"JSON data successfully saved to '{file_path}'.")
    except Exception as e:
        print(f"Error saving JSON to '{file_path}': {e}")
    

def extract_entities(file_path):
 
    with open(file_path, 'r') as file:
        data = json.load(file)
    result = []
    for doc_id, content in data.items():
        cleaned_doc_id = doc_id.lstrip("doc")
        for entity in content.get("entities", []):
            if entity["type"] in {"Person", "NAME_STUDENT"}:
                entity_text = entity["entity_text"]
                positions = tuple(entity["positions"])
                result.append([cleaned_doc_id, entity_text, positions])
    
    return result