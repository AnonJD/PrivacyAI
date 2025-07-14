import re
import pandas as pd
import typing
from typing import List,Tuple, Optional,NoReturn, Dict
import tiktoken
from openai import OpenAI
from functools import reduce 
import json
import copy 
import csv 
import random 
import time 
import os 

with open('input/secret.json', 'r', encoding='UTF-8') as file:
    secret_json = json.load(file)
api_key = secret_json['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

FULL_TEXT = 0
D = {'NAME_STUDENT': ('@@@','###'), 
      'URL_PERSONAL':('&&&','$$$'),
      'EMAIL':('QQQ','^^^'),
      'PHONE_NUM':(r'%%%',r'~~~'),
     }
PATTERN = r'@@@(.*?)###|&&&(.*?)\$\$\$|QQQ(.*?)\^\^\^|%%%(.*?)~~~'

IndexTable =[key for key in D.keys()]


def read_csv(path = 'data/pii_true_entities.csv'):
    df = pd.read_csv(path,encoding='utf-8')
    return df

def read_json(path = 'data/obfuscated_data_06.json'):
    df = pd.read_json(path, orient="records",encoding='utf-8')
    return df

df = read_json()
label = read_csv()

def count_special_token(text:str)->int:
    #params: text: string of annotated text
    #output: number of special tokens(in num_chars) in the input text, int
    acc = 0
    for val in D.values():
        tok1,tok2 = val
        acc += text.count(tok1)*3 + text.count(tok2)*3
    return acc

def mkTrainingExample(text: str, L: List[Tuple[str,str, int, int]]) -> str:
    #params: text: string of unlabeled text 
    #param: L: string *string * int * int list, where the first string is entity name(not used), second string is the type of entity, int * int is the start:end index of the entity in text
    #output: labeld_text: string of labeled text
    
    offset = 0
    for entity in L: 
        _,label, start, end = entity
        start = start + offset
        end = end + offset

        start_mark,end_mark = D[label]
        offset += 6
        text = text[:start] + start_mark + text[start:end] + end_mark + text[end:]
    return text     

def parse_return(text:str) -> List[Tuple[str,str,int,int]]: 
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

def cmp(l1): 
    return int(l1[2])

def get_all_labels(file_idx:int,df :pd.DataFrame) -> List[Tuple[str,str,int,int]]: 
    #params: file_idx as the index of unannotated text 
    #params: df as the .csv file containing entity information
    #output: string * string * int * int list, string of entity name, string of entity type, int * int of start and end index
    df = df[df['file_idx'] == file_idx] 
    L = df.values.tolist()

    for i in range(len(L)):
        L[i] = L[i][1:]
        last = L[i][-1]
        last = last[1:-1]
        last = last.split(',')
        
        L[i] = L[i][:-1]
        L[i].append(int(last[0]))
        L[i].append(int(last[1]))
    L.sort(key =cmp)
    return L

def get_train_example(idx:int)-> str: 
    return df.iloc[idx,FULL_TEXT]

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

def get_train_example(idx:int)-> str: 
    return df.iloc[idx,FULL_TEXT]

def get_testing_list(path)-> List[int]:
    with open(path,'r') as file:
        string = file.read() 
    string = string [1:-1]
    L = [int(each) for each in string.split(',')]
    return L 

def initialize_csv(csv_path): 
    L  = ['file_idx','entity_text','type','positions']
    df = pd.DataFrame(columns=L)
    df.to_csv(csv_path, index=False,encoding = 'utf-8')

def load_csv_iteratively(file_idx:int, csv_path:str,entity_list)-> NoReturn: 
    df = pd.read_csv(csv_path,encoding = 'utf-8')
    L = [] 
    for entity in entity_list: 
        start,end = entity[-2],entity[-1]
        entity = entity[:-2]
        entity.append(f'({start}, {end})')
        entity = [file_idx] + entity
        L.append(entity)
    new_df = pd.DataFrame(L, columns=['file_idx', 'entity_text', 'type', 'positions'])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, encoding='utf-8')

def save_file_response_as_jsonl(file_response, output_file_path):
    """
    Saves the content of a file_response to a JSONL file.

    Parameters:
    - file_response: The response object containing the file content.
    - output_file_path (str): The path where the JSONL file will be saved.
    """
    try:
        lines = file_response.text.strip().split('\n')
        
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

def read_jsonl_file(file_path):

    data_list = []
    
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

def read_cot_jsonl_file(file_path):
    datas = {('25', 'NCOT'): [], ('50', 'NCOT'): [], ('100', 'NCOT'): [],
             ('25', 'COT'): [], ('50', 'COT'): [], ('100', 'COT'): []}

    try:
        # Open the JSONL file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read each line and parse it as a JSON object
            for line in file:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line.strip())  # Parse the line as JSON
                    length, idx, label = tuple(json_obj['custom_id'].split('_'))
                    json_obj['custom_id'] = idx  # Update custom_id to idx only
                    datas[(length, label)].append(json_obj)  # Accumulate the object in the list

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    for key in datas.keys():
        datas[key].sort(key=lambda x: int(x['custom_id']))  # Sort the list by custom_id

    return datas

    def filter_detected_entities():
        for json_lst in [json_lst_ncot, json_lst_cot1, json_lst_cot2]:
            res_lst = []
            for item in json_lst: 
                response = item['response']['body']['choices'][0]['message']['content']
                res_lst.append(response[-1])
                
            mask = [x == 'T' for x in res_lst]
            print(len(res_lst))
            filtered_df = pii_ft_detected_2[mask]
            if json_lst == json_lst_ncot:
                filtered_df.to_csv(f'output/pii_ft_detected_ncot.csv', index=False)
            elif json_lst == json_lst_cot1:
                filtered_df.to_csv(f'output/pii_ft_detected_cot1.csv', index=False)
            else:
                filtered_df.to_csv(f'output/pii_ft_detected_cot2.csv', index=False)

def parse_gpt_result(input_dict: List, output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_idx", "entity_text", "type", "positions"])
        
        for idx in range(len(input_dict)):
            try:
                print(idx)
                response = input_dict[idx]['response']['body']['choices'][0]['message']['content']
                entities = parse_return(response)
                file_idx = input_dict[idx]['custom_id']
                
                for entity_text, entity_type, start, end in entities:
                    writer.writerow([file_idx, entity_text, entity_type, f"({start}, {end})"])
            except Exception as e:
                raise e 
                #print(e)
                #print(f"Error parsing index {idx}: {e}")

    
def structure_training_set(inpath,outpath,system_prompt = ""):
    with open(inpath,'r') as file:
        string = file.read()
    
    string = string[1:-1]
    D = [int(each) for each in string.split(',')]

    out = []
    for i in range(len(D)): 
        file_idx = D[i]
        L = get_all_labels(file_idx,label)
        system = {"role": "system", "content": system_prompt}  
        input = {"role": "user", "content": get_train_example(file_idx)}
        output = {"role": "assistant", "content": mkTrainingExample(get_train_example(file_idx),L)}
        l = [system, input, output]
        item = {"messages": l}
        out.append(item)

    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in out:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
        
    return None

ins_prompt ='''Label the entity of the following text: @@@,### to label student name; &&&,$$$ to label personal URL; QQQ,^^^ to label personal email; %%%,~~~ to label phone number\n'''
def structure_training_set_from_list(d,outpath,system_prompt = 'You are an expert in labeling personally identifiable information'):
                    
    out = []
    for i in range(len(d)): 
        file_idx = d[i]
        L = get_all_labels(file_idx,label)
        system = {"role": "system", "content": system_prompt}  
        input = {"role": "user", "content": ins_prompt + get_train_example(file_idx)}
        output = {"role": "assistant", "content": mkTrainingExample(get_train_example(file_idx),L)}
        l = [system, input, output]
        item = {"messages": l}
        out.append(item)

    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in out:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
        
    return None

def mkTrainingBatch(testing_indexs): 
    model = "gpt-4o-mini"
    
    system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"

    ins_prompt = '''Label the entity of the following text: @@@,### to label student name;
&&&,$$$ to label personal URL; QQQ,^^^ to label personal email; %%%,~~~ to label phone number\n
Ensure that the rest of the text remains unchanged, word for word.
Maintain the original punctuation, quotation marks, spaces, and line breaks. 
If the text does not contain any PII, return it as is.
For example, if the input is:
COURSERA - University of Virginia, Darden School of Business - Design Thinking Assignment    Dharmendra Asiri    Washington,DC / March 8, 2019    email djones@gmail.com  linkedIn https://www.linkedin.com/in/mmartinez
The output should be:
COURSERA - University of Virginia, Darden School of Business - Design Thinking Assignment    @@@Dharmendra Asiri###    Washington,DC / March 8, 2019    email QQQdjones@gmail.com^^^  linkedIn &&&https://www.linkedin.com/in/mmartinez$$$
Another example:
I do conclude, my assignment by thanking Lecturers, University of Virginia and other  partners who contributed to this online courses.\n\nMay God bless you.\n\nEslam Abo Fatma\n\nRwanda- Africa\n\nEmail: murraythomas@gmail.com\n\nTel: (223)392-2765\n\n'
The output should be:
I do conclude, my assignment by thanking Lecturers, University of Virginia and other  partners who contributed to this online courses.\n\nMay God bless you.\n\n@@@Eslam Abo Fatma###\n\nRwanda- Africa\n\nEmail: QQQmurraythomas@gmail.com^^^\n\nTel: %%%(223)392-2765~~~\n\n'
Another example:
An article was published which  described one of the most successful entrepreneurs in the world, Jeff Bezos. It was mentioned  that Bezos insists that no PPTs are shown during the board meetings but stories are told.
The output should be exactly the same as input:
An article was published which  described one of the most successful entrepreneurs in the world, Jeff Bezos. It was mentioned  that Bezos insists that no PPTs are shown during the board meetings but stories are told.
Please repeat this process with the following file:\n
'''
    file_name ='output/prompt_test'
    out = []
    for file_idx in testing_indexs: 
 
        request = {"custom_id": f'{file_idx}', "method": "POST", 
                  "url": "/v1/chat/completions", 
                  "body": {"model": model, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": ins_prompt + get_train_example(file_idx)}],
                  "temperature":0}}
        
        out.append(request)
    
    outpath = f'{file_name}.jsonl'
    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in out:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f'finish writing batch file: {outpath}')

def mkTrainingBatch_ft(testing_indexs, model, output_path): 
 
    
    system_prompt = "You are an expert in labeling personally identifiable information"

    ins_prompt = '''Label the entity of the following text: @@@,### to label student name; &&&,$$$ to label personal URL; QQQ,^^^ to label personal email; %%%,~~~ to label phone number\n'''
    file_name = output_path
    out = []
    for file_idx in testing_indexs: 
 
        request = {"custom_id": f'{file_idx}', "method": "POST", 
                  "url": "/v1/chat/completions", 
                  "body": {"model": model, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": ins_prompt + get_train_example(file_idx)}],
                  "temperature":0}}
        
        out.append(request)
    
    outpath = f'{file_name}.jsonl'
    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in out:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f'finish writing batch file: {outpath}')


def stratify(csv_path = 'data/train_set_2.csv'): 
    df = pd.read_csv(csv_path,encoding='utf-8')

    phone_list = df[df['type'] == 'PHONE_NUM'].iloc[:, 0].tolist()
    print(len(phone_list))
    url_list = df[df['type'] == 'URL_PERSONAL'].iloc[:, 0].tolist()
    print(len(url_list))
    email_list = df[df['type'] == 'EMAIL'].iloc[:, 0].tolist()
    print(len(email_list))

    student_list = df[df['type'] == 'NAME_STUDENT'].iloc[:, 0].tolist()
    print(len(student_list))

    train_idxs = random.sample(phone_list,max(1,len(phone_list)))+ \
                 random.sample(url_list,max(1,len(url_list))) + \
                 random.sample(email_list,max(len(email_list),1))+ \
                 random.sample(student_list,max(len(student_list),1))
    
    
    return train_idxs 
 
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


#def get_gpt_response(input_text:str) -> str:
    #param: string of text to be annotated 
    #out: string of text annotated 

    system_prompt = '''As an undergraduate student working for me, your task is to identify all personally identifiable information (PII), especially names, in a piece of text. I will be satisfied only if you do your utmost to avoid false negatives and cautiously infer all personally identifiable names from the context.\n
Example 1: “Jakndnd is talking about her brother pIoM jKpjhobj, who's skipped her birthday party a few times.”\n
You should label Jakndnd and pIoM jKpjhobj as PII since it appears that Jakndnd exhibits human behavior, and pIoM jKpjhobj is her brother.\n
Example 2: “Occasionally, Anfoaipjqd likes to have a coffee chat with her friends.”\n
You should label Anfoaipjqd as PII since only humans can have coffee chats.\n
Example 3: “Ascar likes to talk about gioibajhjjo.”\n
You should label both Ascar and gioibajhjjo since you cannot be sure whether gioibajhjjo is a person or not, so you should consider it to avoid false negatives. Ascar is talking, so he is also a human.\n
'''
    ins_prompt = 'Label the entity of the following text: @@@,### to label student name; &&&,$$$ to label personal URL; QQQ,^^^ to label personal email; %%%,~~~ to label phone number\n'

    response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:cmu-plus::A7fVfDc8",
    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": ins_prompt + input_text}],
    temperature = 0)
    return response.choices[0].message.content 


#def make_batch(text_list:List[str], identifier_list:List[str],store_directory:str)-> None: 
    system_prompt = '''As an undergraduate student working for me, your task is to identify all personally identifiable information (PII), especially names, in a piece of text. I will be satisfied only if you do your utmost to avoid false negatives and cautiously infer all personally identifiable names from the context.\n
Example 1: “Jakndnd is talking about her brother pIoM jKpjhobj, who's skipped her birthday party a few times.”\n
You should label Jakndnd and pIoM jKpjhobj as PII since it appears that Jakndnd exhibits human behavior, and pIoM jKpjhobj is her brother.\n
Example 2: “Occasionally, Anfoaipjqd likes to have a coffee chat with her friends.”\n
You should label Anfoaipjqd as PII since only humans can have coffee chats.\n
Example 3: “Ascar likes to talk about gioibajhjjo.”\n
You should label both Ascar and gioibajhjjo since you cannot be sure whether gioibajhjjo is a person or not, so you should consider it to avoid false negatives. Ascar is talking, so he is also a human.\n
'''

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
__all__ = [name for name in globals() if not name.startswith("_") and name != "df"]

