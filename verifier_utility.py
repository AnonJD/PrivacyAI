import re
import pandas as pd
import typing
from typing import List,Tuple, Optional,NoReturn
import tiktoken
from openai import OpenAI
from functools import reduce 
import json
import copy 
import ast
from gpt_utility import client, read_json
from gpt_utility import  mkTrainingBatch, get_testing_list, save_file_response_as_jsonl, read_jsonl_file, parse_gpt_result

df = read_json()

def get_context(L:Tuple[int,str,str,str], df:pd.DataFrame,context_length:int)-> str: 
    indexes = [int(each) for each in L[-1].replace('(','').replace(')','').split(',')]
    tokens =  df['tokens'].iloc[L[0]]
    doc = df['full_text'].iloc[L[0]]
    entity = L[1] 
    error_log = []
    
    tokens_idx = []
    
    deleted = 0
    for i,each in enumerate(tokens): 
        start = doc.find(each) + deleted
        end = start + len(each)
        deleted += len(doc) - len(doc[doc.find(each) + len(each):])
        doc = doc[doc.find(each) + len(each):]
        tokens_idx.append((start,end-1,i))
    
    old_tokens_idx = copy.deepcopy(tokens_idx)
    
    def filter_fn(x):
        x_start,x_end,_  = x
        return (indexes[0] <= x_start <= indexes[1]) or (indexes[0] <= x_end <= indexes[1]) or (x_start <= indexes[0] <= x_end) or (x_start <= indexes[1] <= x_end)
 
    tokens_idx = list(filter(filter_fn,tokens_idx)) 
 
    tokens_idx.sort(key = lambda x: x[-1])

    try:
        res_idx0 = tokens_idx[0][-1]
        res_idx1 = tokens_idx[-1][-1]
    except IndexError:

        print('the indexes are:',L[-1])
        print('the file_idx',L[0])
        print('the tokens are', tokens) 
        print('the old_token_idxs are',old_tokens_idx)
        return None
    res = tokens[max(0,res_idx0-context_length):res_idx0] + ['***'] + tokens[res_idx0:min(len(tokens),res_idx1+context_length+1)]

    result = ' '.join(res)
    try:
        assert(''.join(entity.split()) in ''.join(result.split()))
    except AssertionError:
        print(f'entity: {entity} not in {result}')
        error_log.append(f'entity: {entity} not in {result}')
    return result,error_log


def create_experiment_jsonl(path_to_experiment_set_csv:str,df:pd.DataFrame,context_lengths:List[int]): 
    res = [] 
    model = "gpt-4o-mini"
    Error_log = []


    system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"
    file_name ='output/ft+prompting'
    for context_length in context_lengths: 
        data = pd.read_csv(path_to_experiment_set_csv)
        for index, row in data.iterrows():
            row = row.tolist()
            context,error_log = get_context(row,df,context_length)
            entity = row[1]
            index = index
            non_cot_prompt = f'Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, only output T or F'
            request = {"custom_id": f'{context_length}_{index}_NCOT', "method": "POST", 
                  "url": "/v1/chat/completions", 
                  "body": {"model": model, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": non_cot_prompt}],
                  "temperature":0}}
            res.append(request)
            if len(error_log)>= 1: 
                Error_log.append(error_log[0])
    

    for context_length in context_lengths: 
        data = pd.read_csv(path_to_experiment_set_csv)
        for index, row in data.iterrows():
            row = row.tolist()
            context,error_log = get_context(row,df,context_length)
            entity = row[1]
            index = index
            cot_prompt = f'''Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, 
                            think step by step before outputting T or F, format your response as (your reasoning) + [Response:] T or F '''
            request = {"custom_id": f'{context_length}_{index}_COT', "method": "POST", 
                  "url": "/v1/chat/completions", 
                  "body": {"model": model, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": cot_prompt}],
                  "temperature":0}}
            res.append(request)
 
    
    outpath = f'{file_name}.jsonl'
    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in res:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    return Error_log
    
def get_gpt_response(entity, context, temperature): 
    system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"
    usr_prompt =f'''Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, 
                            think step by step before outputting T or F, format your response as (your reasoning) + [Response:] T or F'''
    
    response =  client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content":  usr_prompt}
  ],
  temperature = temperature) 
    res = response.choices[0].message.content
    if res[-1] == 'T' or res[-1] == 'F':
        return res,res[-1]
    elif res[-2] == 'T' or res[-2] == 'F':
        return res,res[-2]
    else: 
        return res,'UND'
    
def initialize_csv(csv_path): 
    L  = ['file_idx','entity_text','type','positions','label']
    df = pd.DataFrame(columns=L)
    df.to_csv(csv_path, index=False,encoding = 'utf-8')

def generate_cot_train(training_df):
    system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"
    out = []
    error_log = []
    for index,row in training_df.iterrows():
        print(index) 
        label = row['label']
        input = tuple(row[:-1])
        entity = input[1] 
        context = get_context(input,df,100)
        if context == None: 
            continue 
        else:
            context,_ = context
        usr_prompt =f'''Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, 
                                think step by step before outputting T or F, format your response as (your reasoning) + [Response:] T or F'''
        counter = 0
        flag = False
        while not flag:
            if counter == 0:
                temperature = 0
            else:
                temperature = 0.3
            response,idd = get_gpt_response(entity,context,temperature)
            if idd == label:
                system = {"role": "system", "content": system_prompt}  
                input = {"role": "user", "content": usr_prompt}
                output = {"role": "assistant", "content": response}
                l = [system, input, output]
                item = {"messages": l}
                out.append(item)
                flag = True
            else: 
                if counter > 5:
                    system = {"role": "system", "content": system_prompt}  
                    input = {"role": "user", "content": usr_prompt}
                    output = {"role": "assistant", "content": 'Default to True to prevent false negatives. [Response:] T'}
                    l = [system, input, output]
                    item = {"messages": l}
                    out.append(item)
                    
                    error_log.append(row)
                    break
                counter+=1
                continue


    with open('output/ft_filter_cot.jsonl', 'w',encoding='utf-8') as f:
            for entry in out:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + '\n')

    df_new_rows = pd.DataFrame(error_log, columns=['file_idx','entity_text','type','positions','label'])
    
    csv_file = 'pii_detected_ft_train_error_log.csv'
    
    df_new_rows.to_csv(csv_file, index=False)


def generate_ncot_train(training_df):
    system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"
    out = []
 
    for index,row in training_df.iterrows():
        print(index) 
        label = row['label']
        input = tuple(row[:-1])
        entity = input[1] 
        context = get_context(input,df,100)
        if context == None: 
            continue 
        else:
            context,_ = context
        usr_prompt =f'''Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, only output T or F.'''
        system = {"role": "system", "content": system_prompt}  
        input = {"role": "user", "content": usr_prompt}
        output = {"role": "assistant", "content": label}
        l = [system, input, output]
        item = {"messages": l}
        out.append(item)

        with open('output/ft_filter_ncot.jsonl', 'w',encoding='utf-8') as f:
            for entry in out:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + '\n')

 


def make_verifier_train_set():
    detected_path = 'output/batch_request.jsonl'
    prompted_result = read_jsonl_file('output/batch_request.jsonl')
    parse_gpt_result(prompted_result, 'output/gpt_ft_result.csv')

    ref_path = 'data/pii_true_entities.csv'
    pred_path = 'output/gpt_ft_result.csv'
    indices_path = 'data/train_indices_2.txt'

    df_ref = pd.read_csv(ref_path)
    df_pred = pd.read_csv(pred_path)
    with open(indices_path, 'r') as f:
        line = f.read().strip()   
        indices = ast.literal_eval(line)
    df_ref = df_ref[df_ref['file_idx'].isin(indices)]
    df_pred = df_pred[df_pred['file_idx'].isin(indices)]
    df_pred['label'] = ''
    for index, row in df_pred.iterrows():
        if ((df_ref == row[df_ref.columns]).all(axis=1)).any():
            df_pred.at[index, 'label'] = 'T'
        else:
            df_pred.at[index, 'label'] = 'F' 
    df_pred.to_csv('output/verifier_train.csv', index=False)


system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"


def get_context(L:Tuple[int,str,str,str], df:pd.DataFrame, context_length=100)-> str: 
    indexes = [int(each) for each in L[-1].replace('(','').replace(')','').split(',')]
    tokens =  df['tokens'].iloc[L[0]]
    doc = df['full_text'].iloc[L[0]]
    entity = L[1] 
    error_log = []
    tokens_idx = []
    
    deleted = 0
    for i,each in enumerate(tokens): 
        start = doc.find(each) + deleted
        end = start + len(each)
        deleted += len(doc) - len(doc[doc.find(each) + len(each):])
        doc = doc[doc.find(each) + len(each):]
        tokens_idx.append((start,end-1,i))
    
    old_tokens_idx = copy.deepcopy(tokens_idx)
    
    def filter_fn(x):
        x_start,x_end,_  = x
        return (indexes[0] <= x_start <= indexes[1]) or (indexes[0] <= x_end <= indexes[1]) or (x_start <= indexes[0] <= x_end) or (x_start <= indexes[1] <= x_end)
 
    tokens_idx = list(filter(filter_fn,tokens_idx)) 
 
    tokens_idx.sort(key = lambda x: x[-1])

    try:
        res_idx0 = tokens_idx[0][-1]
        res_idx1 = tokens_idx[-1][-1]
    except IndexError:

        print('the indexes are:',L[-1])
        print('the file_idx',L[0])
        print('the tokens are', tokens) 
        print('the old_token_idxs are',old_tokens_idx)
        return None
    res = tokens[max(0,res_idx0-context_length):res_idx0] + ['***'] + tokens[res_idx0:min(len(tokens),res_idx1+context_length+1)]

    result = ' '.join(res)
   
    return result,error_log



def create_jsonl_NCOT(path_to_set_csv:str, df:pd.DataFrame, model_name: str):
    ft_detected_2 = pd.read_csv(path_to_set_csv)
    res = []
    Error_log = []

    for index, row in ft_detected_2.iterrows():
        row = row.tolist()
        context,error_log = get_context(row, df)
        entity = row[1]
        index = index
        non_cot_prompt = f'Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, only output T or F'
        request = {"custom_id": f'{index}_NCOT', "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {"model": model_name,
                         "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": non_cot_prompt}],
                         "temperature": 0,
                         'logprobs': True,
                         'top_logprobs': 20}}
        res.append(request)
        if len(error_log)>= 1: 
            Error_log.append(error_log[0])
    
    outpath = 'output/ft+ft_testing_ncot.jsonl'
    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in res:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Output file saved to {outpath}")
    return Error_log


def create_jsonl_COT(path_to_set_csv:str, df:pd.DataFrame, model_name: str ):

    ft_detected_2 = pd.read_csv(path_to_set_csv)
    res = []
    
    for index, row in ft_detected_2.iterrows():
        row = row.tolist()
        context,error_log = get_context(row, df)
        entity = row[1]
        index = index
        cot_prompt = f'''Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, 
                        think step by step before outputting T or F, format your response as (your reasoning) + [Response:] T or F '''
        request = {"custom_id": f'{index}_COT1', "method": "POST", 
                   "url": "/v1/chat/completions", 
                   "body": {"model": model_name,
                            "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": cot_prompt}],
                            "temperature": 0,
                            'logprobs': True,
                            'top_logprobs': 20}}
        res.append(request)
    
    outpath = 'output/ft+ft_testing_cot.jsonl'
    with open(outpath, 'w',encoding='utf-8') as f:
        for entry in res:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Output file saved to {outpath}")


def get_response_label(text):
    if text[-1] == 'T' or text[-1] == 'F':
        return text[-1]
    elif text[-2] == 'T' or text[-2] == 'F':
        return text[-1]
    else:
        return 'T'