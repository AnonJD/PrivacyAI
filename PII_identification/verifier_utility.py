import re
import pandas as pd
import typing
from typing import List,Tuple, Optional,NoReturn
import tiktoken
from openai import OpenAI
from functools import reduce 
import json
import copy 
from gpt_utility import client



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
    file_name ='context_experiment'
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
    
def get_gpt_response(entity, context): 
    system_prompt = "You are an expert in labeling Personally Identifiable Information. Start your response rightaway without adding any prefix(such as Response:) and suffix"
    usr_prompt =f'''Determine if {entity} is a privately identifiable information in its context: {context}, think carefully before saying no to protect against PII leakage, 
                            think step by step before outputting T or F, format your response as (your reasoning) + [Response:] T or F'''
    
    response =  client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content":  usr_prompt}
  ],
  temperature = 0.3) 
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