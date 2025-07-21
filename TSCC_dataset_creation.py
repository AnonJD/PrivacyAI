import json
import pandas as pd
import copy
import random
from names_dataset import NameDataset, NameWrapper
nd = NameDataset()

#### Step 1: importing 

with open('data/original_transcripts.txt', 'r') as f:
    text_lst = json.load(f)

with open('data/placeholder_locations_new.txt','r') as f: 
    label_lst = json.load(f)
assert len(text_lst) == len(label_lst)

for i in range(len(text_lst)):
    labelis = label_lst[i]
    textis = text_lst[i]
    for l in labelis: 
        start = l[0]
        end = l[1]
        label = l[2]
        file_idx = l[3]
        assert textis[start:end] == f'<{label}>'
        assert file_idx == i

#### Step 2: create and permute the index list

index_list = [i for i in range(len(text_lst))]
random.shuffle(index_list)


#### Step 3: create culture x gender dictonary

culture = ['Asia', 'Europe', 'Africa', 'Oceania','Americas']
gender = ['M','F']
groups_dict = {}
for g in gender: 
    for c in culture: 
        groups_dict[(g,c)] = []

assert len(groups_dict.keys()) == len(culture) * len(gender)


#### Step 4: loop and assign index

keys = list(groups_dict.keys())
mode_num = len(keys)
for i in range(len(index_list)): 
    idx = i % mode_num
    key = keys[idx]
    groups_dict[key].append(index_list[i])


for key in keys: 
    assert abs(len(groups_dict[key])-(len(index_list)//mode_num)) < 2

picked_indices = [random.choice(indices) for indices in groups_dict.values()]

#### Step 5: for each group, gather all first name and last name according to gender and culture FN,LN

def create_mapping(): 
    D = {}  
    all = pd.read_csv('all.csv')
    for i in range(len(all)): 
        alpha2 = all.iloc[i]['alpha-2']
        region = all.iloc[i]['region']
        D[alpha2] = region
    return D 

country_to_culture = create_mapping()
culture_to_country = {} 
for k,v in country_to_culture.items(): 
    if v not in culture_to_country.keys():
        culture_to_country[v] = []
    culture_to_country[v].append(k)

# Convert tuple keys to string keys
string_key_dict = {str(k): v for k, v in groups_dict.items()}

# Save to JSON file
with open("groups_dict.json", "w") as file:
    json.dump(string_key_dict, file, indent=4)

print("Saved groups_dict to groups_dict.json")

import re
def remove_non_english_strings(input_set):

    def is_english_string(s):
        return bool(re.match(r'^[a-zA-Z\s]+$',s))
    return {s for s in input_set if is_english_string(s)}
groups_dict_fnln = {} 

# Get FN (first names) and LN (last names)
# Get FN (first names) and LN (last names)
groups_dict_fnln = {} 

for (gender,culture) in groups_dict.keys():
    name_sets = []

    countries = culture_to_country[culture]
    for country in countries:
        name_statistic_fn = nd.get_top_names(n=1745990813245382364, gender=gender, country_alpha2=country)
        if name_statistic_fn  == {}: 
            continue
        first_name_list = name_statistic_fn[country][gender]

        name_statistic_ln = nd.get_top_names(n=1745990813245382364, use_first_names=False,country_alpha2=country)
        if name_statistic_ln  == {}:
            continue
        last_name_list =name_statistic_ln[country] 
        n_sample = min(len(last_name_list),len(first_name_list))

        first_names = random.sample(first_name_list,n_sample)
        last_names = random.sample(last_name_list,n_sample)
        names = {f'{first_names[i]} {last_names[i]}' for i in range(n_sample)}
        name_sets.append(names)

    N = set.union(*name_sets)
    N = remove_non_english_strings(N)
    groups_dict_fnln[(gender,culture)] = N


#### Step 6: replace placeholders with names sampled from groups_dict_fnln
not_name_dict = {
    "STUDENT'S USERNAME": "student1234",
    "NAME OF THE BAND": "The Moonlight Strummers",
    "LANGUAGE SCHOOL": "Global Lingua Academy",
    "TIME": "10:45 AM",
    "AGE": "25",
    "DOB": "1998-04-23",
    "YEAR": "2019",
    "INSTAGRAM ACCOUNT": "@wanderlust_photographer",
    "STUDENT'S EMAIL ADDRESS": "student.name@email.com",
    "STUDENT'S COMPANY": "BrightPath Solutions",
    "NUMBER": "237859541908",
    "DATE": "2016-10-29",
    "RESORT": "Sunset Paradise Resort"
}

import copy
new_label_lst = copy.deepcopy(label_lst)
new_text_lst = copy.deepcopy(text_lst)
for key in groups_dict.keys(): 
    
    N = groups_dict_fnln[key]
    file_idxs = groups_dict[key]
    #for each group, get their respective candidates fn,ln, and the indices of files. 

    for file_idx in file_idxs:
        #get the ground turth text and label
        ground_truth_labels = new_label_lst[file_idx]
        text = new_text_lst[file_idx]

        #get the set of unique label
        labels = set(lb[2] for lb in ground_truth_labels)
        label_count = len(labels)

        #sample first name and last names pairs to replace names in label


        names = random.sample(list(N),label_count)

        #create a full name for each label, create a mapping from each label to the full name
        labels_replacement = [f'{names[i]}' for i in range(len(labels))]
        label_to_replacement = {}

        for i,label in enumerate(labels):
            label_to_replacement[label] = labels_replacement[i]
        
        offset = 0
        for i,l in enumerate(ground_truth_labels): 
        
            label = l[2]
            s = l[0]
            e = l[1]
            label_length = e - s

            if label in not_name_dict.keys():
                s += offset
                ori_e = e + offset
                replacement_length = len(replacement:=not_name_dict[label])
                offset += replacement_length - label_length
                e = e + offset
                ground_truth_labels[i][0] = s
                ground_truth_labels[i][1] = e
             
                new_text_lst[file_idx] = new_text_lst[file_idx][:s] + replacement + new_text_lst[file_idx][ori_e:]
                continue 


            s = s + offset
            ori_e = e + offset
           

            replacement_length = len(replacement:=label_to_replacement[label])
            
            offset += replacement_length - label_length
            e = e + offset
            ground_truth_labels[i][0] = s
            ground_truth_labels[i][1] = e
            ground_truth_labels[i][2] = replacement
            new_text_lst[file_idx] = new_text_lst[file_idx][:s] + replacement + new_text_lst[file_idx][ori_e:]

for file_idx in index_list: 
    old_lab = label_lst[file_idx]
    new_lab = new_label_lst[file_idx]
    old_text = text_lst[file_idx]
    new_text = new_text_lst[file_idx]
    assert len(old_lab) == len(new_lab)
    for i in range(len(old_lab)):
        old = old_lab[i] 
        new = new_lab[i]
        assert old[3] == new[3]
        (old_s,old_e) = old[0],old[1]
        (new_s,new_e) = new[0],new[1]

        try:
            assert old_text[old_s-20:old_s] == new_text[new_s-20:new_s]
        except:
            assert '<' in old_text[old_s-20:old_s] or '>' in old_text[old_s-20:old_s]
         

        try:
            assert old_text[old_e:old_e+20] == new_text[new_e:new_e+20]
        except: 
            assert '<' in old_text[old_e:old_e+20] or '>' in old_text[old_e:old_e+20]

def is_english_string(s):
        return bool(re.match(r'^[a-zA-Z0-9\s!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]+$',s))

assert len(new_text_lst) == len(new_label_lst)
for i in range(len(new_label_lst)):
    new_entities = new_label_lst[i]
    new_text = new_text_lst[i]
    for entity in new_entities:
        s, e, entity_text, file_idx = entity
        assert file_idx == i
      
for i in range(len(new_label_lst)):
    new_label_lst[i] = list(filter(lambda l : l[2] not in not_name_dict.keys(),new_label_lst[i]))

assert len(new_text_lst) == len(new_label_lst)
for i in range(len(new_label_lst)):
    new_entities = new_label_lst[i]
    new_text = new_text_lst[i]
    for entity in new_entities:
        s, e, entity_text, file_idx = entity
        assert file_idx == i
     
        assert new_text[s:e] == entity_text
        assert is_english_string(entity_text)
#### Step 7: Save new_text_lst and new_label_lst to local
# Save the processed transcripts to a text file
with open('output/new_texts.txt', 'w') as f:
    f.write(json.dumps(new_text_lst, indent=4))
import pandas as pd

data = []
for entity_list in new_label_lst:
    for entity in entity_list:
        file_idx = entity[3]
        entity_text = entity[2]
        positions = (entity[0], entity[1])
        data.append([file_idx, entity_text, positions])

df = pd.DataFrame(data, columns=['file_idx', 'entity_text', 'positions'])

output_file = 'output/new_labels.csv'
df.to_csv(output_file, index=False)

print(f"CSV file saved as {output_file}")


new_labels = pd.read_csv('output/new_labels.csv')

data = [] 
init_file_idx = 0
acc = [] 
for i,row in new_labels.iterrows():
    if row['file_idx'] != init_file_idx: 
        data.append(acc)
        acc = [] 
        init_file_idx = row['file_idx']
    row = row.tolist()
    _row = [0,0,0,0]
    _row[1] = 'NAME_STUDENT'
    _row[0] = row[1]
    tup = eval(row[2])
    _row[2],_row[3]= tup[0],tup[1]
    acc.append(tuple(_row))

data.append(acc)

with open ('output/new_texts.txt','r') as f: 
    new_text_lst = json.load(f)

from typing import List,Tuple
FULL_TEXT = 0
D = {'NAME_STUDENT': ('@@@','###'), 
      'URL_PERSONAL':('&&&','$$$'),
      'EMAIL':('QQQ','^^^'),
      'PHONE_NUM':(r'%%%',r'~~~'),
     }
PATTERN = r'@@@(.*?)###|&&&(.*?)\$\$\$|QQQ(.*?)\^\^\^|%%%(.*?)~~~'

IndexTable =[key for key in D.keys()]

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

with open('picked_indices.txt', 'w') as file:
    for idx in picked_indices:
        file.write(f"{idx}\n")   

training_examples = [new_text_lst[i] for i in picked_indices]
training_examples_label = [data[i] for i in picked_indices]
assert(len(training_examples_label ) == len(training_examples))
training_examples = [mkTrainingExample(training_examples[i],training_examples_label[i]) for i in range(len(training_examples))]

#function to create training example and parse return
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

ret = [parse_return(training_examples[i]) for i in range(len(training_examples))]

for i in range(len(ret)): 
    for j in range(len(ret[i])): 
        ret[i][j] = tuple(ret[i][j])

original_examples = [new_text_lst[i] for i in picked_indices]


system_prompt = '''You are an expert in labeling personally identifiable information. Start your re-
sponse right away without adding any prefix (such as “Response:”) or suffix'''
ins_prompt = 'Label the entity of the following text: use @@@,### to label name\n'
outpath = 'output/ft_tscc.jsonl'
out = []
for i in range(len(training_examples)): 
    id = i
    system = {"role": "system", "content": system_prompt}  
    input = {"role": "user", "content": ins_prompt + original_examples[i]}
    output = {"role": "assistant", "content": training_examples[i]}
    l = [system, input, output]
    item = {"messages": l}
    out.append(item)

with open(outpath, 'w',encoding='utf-8') as f:
    for entry in out:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + '\n')