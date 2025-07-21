from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider, TransformersNlpEngine, NerModelConfiguration
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
import names
import random
from random import randrange
from datetime import timedelta, datetime
import spacy
import json
import re
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForTokenClassification
from faker import Faker
from typing import List, Tuple
import pandas as pd
import argparse
import ast 
from tqdm import tqdm
from gpt_utility import get_testing_list

#remember to also do pip install -U "spacy>=3.6" spacy-curated-transformers
#remember to install C++ build tool 

type pii_entity = Tuple[int, str, str, Tuple[int, int]]

def read_file(filepath: str):
    return pd.read_json(filepath, orient="records")


# Create configuration containing engine name and models
def get_configuration(spaCy_model: str):
    configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": spaCy_model}],
    }
    return configuration

def get_conf_file(spaCy_model: str, transformer_model: str = None):
    snapshot_download(repo_id=transformer_model)
    # Instantiate to make sure it's downloaded during installation and not runtime
    AutoTokenizer.from_pretrained(transformer_model)
    AutoModelForTokenClassification.from_pretrained(transformer_model)
    config_dict = {
        "en_core_web_lg + obi/deid_roberta_i2b2": "Config/lg+roberta.yaml",
        "en_core_web_lg + StanfordAIMI/stanford-deidentifier-base": "Config/lg+stanford.yaml",
        "en_core_web_trf + obi/deid_roberta_i2b2": "Config/trf+roberta.yaml",
        "en_core_web_trf + StanfordAIMI/stanford-deidentifier-base": "Config/trf+stanford.yaml",
    }

    # Create configuration containing engine name and models
    conf_file = config_dict[spaCy_model + ' + ' + transformer_model]
    return conf_file

# Function to create NLP engine based on configuration
def create_nlp_engine(spaCy_model: str, transformer_model: str = None):
    if spaCy_model not in ["en_core_web_lg", "en_core_web_trf"]:
        raise ValueError("Input spaCy model is not supported.")
    if transformer_model is not None:
        if transformer_model not in ["obi/deid_roberta_i2b2", "StanfordAIMI/stanford-deidentifier-base"]:
            print(transformer_model)
            raise ValueError("Input transformer model is not supported.")
    
    # spaCy model only
    if transformer_model is None:
        configuration = get_configuration(spaCy_model)
        provider = NlpEngineProvider(nlp_configuration=configuration)

    # spaCy model with transformer
    else:
        conf_file = get_conf_file(spaCy_model, transformer_model)
        print(conf_file)
        provider = NlpEngineProvider(conf_file=conf_file)
    
    nlp_engine = provider.create_engine()
    return nlp_engine

def de_identify_pii(text_transcript,nlp_engine):
    # Initialize the analyzer and anonymizer
    analyzer = AnalyzerEngine(
        nlp_engine = nlp_engine,
        supported_languages=["en"]
    )
    anonymizer = AnonymizerEngine()

    # Define date range for generating random dates and generate a random date
    # d1 = datetime.strptime('1/1/2008 1:30 PM', '%m/%d/%Y %I:%M %p')
    # d2 = datetime.strptime('1/1/2009 4:50 AM', '%m/%d/%Y %I:%M %p')
    # random_date = (d1 + timedelta(days=random.randint(0, (d2 - d1).days))).strftime('%m/%d/%Y')

    fake = Faker()

    # Function to generate a unique fake name
    def generate_fake_name(existing_names):
        while True:
            fake_name = names.get_first_name()
            if fake_name not in existing_names:
                return fake_name
    
    # Function to generate a unique fake email
    def generate_fake_email(fake_name):
        return f"foo"
    
    # Function to generate a unique fake location
    def generate_fake_location():
         return 'foo'  # Generate a fake city name using Faker

    # Function to generate a unique fake phone number
    def generate_fake_phone_number():
        return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    # define entities that you want Presidio to detect
    entities = ["PERSON", "EMAIL_ADDRESS", "URL", "PHONE_NUMBER"]

    # Analyze the text to find PII
    results_analyzed = analyzer.analyze(text=text_transcript, language="en", entities=entities, 
                                        score_threshold=None, return_decision_process=True)

    # Create a mapping of original names to unique fake names
    name_mapping = {}
    existing_names = set()
    for result in results_analyzed:
        if result.entity_type == "PERSON":
            original_name = text_transcript[result.start:result.end]
            if original_name not in name_mapping:
                fake_name = generate_fake_name(existing_names)
                name_mapping[original_name] = fake_name
                existing_names.add(fake_name)

    # Email mapping to ensure consistent fake emails
    email_mapping = {}
    for result in results_analyzed:
        if result.entity_type == "EMAIL_ADDRESS":
            original_email = text_transcript[result.start:result.end]
            if original_email not in email_mapping:
                fake_name = generate_fake_name(existing_names)
                fake_email = generate_fake_email(fake_name)
                email_mapping[original_email] = fake_email
    
    # Phone number mapping to ensure consistent fake phone numbers
    phone_mapping = {}
    for result in results_analyzed:
        if result.entity_type == "PHONE_NUMBER":
            original_phone = text_transcript[result.start:result.end]
            if original_phone not in phone_mapping:
                fake_phone = generate_fake_phone_number()
                phone_mapping[original_phone] = fake_phone

    operators = {
        "PERSON": OperatorConfig("custom", {"lambda": lambda text : name_mapping.get(text, text)}),
        # "DATE_TIME": OperatorConfig("replace", {"new_value": random_date}),
        # Add more categories
        "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda text: email_mapping.get(text, text)}),
        # "LOCATION": OperatorConfig("replace", {"new_value": generate_fake_location()}),
        "PHONE_NUMBER": OperatorConfig("custom", {"lambda": lambda text: phone_mapping.get(text, text)}),
        "URL": OperatorConfig("replace", {"new_value": fake.url()}),
    }

    # Anonymize the text
    results_anonymized = anonymizer.anonymize(
        text=text_transcript,
        analyzer_results=results_analyzed,
        operators=operators
    )

    return results_analyzed, results_anonymized

# Function to analyze text with Presidio and return PII entities
def analyze_texts_with_presidio(df: pd.DataFrame, nlp_engine) -> List[pii_entity]:
    pii_entities: List[pii_entity] = []
    
    for i, row in tqdm(df.iterrows()):
   
        text = row.full_text
        results_analyzed, results_anonymized = de_identify_pii(text,nlp_engine)
        
        for result in results_analyzed:
            start = result.start
            end = result.end  # Presidio's end index is exclusive
            entity_text = text[start:end]
            pii_entities.append((i, entity_text, result.entity_type, (start, end)))
    
    return pii_entities

# Function to append PII entities to an existing file
def append_entities_to_file(entities: List[pii_entity], file_path: str, indent: int = 4):
    with open(file_path, 'a',encoding='utf-8') as f:
        for entity in entities:
            indent_space = ' ' * indent
            entity_str = f"{entity}\n"
            f.write(indent_space + entity_str)
# Using only spaCy model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple script to process a file.")
    parser.add_argument('model', choices=['trf', 'lg'], help="The model to use: 'trf' or 'lg'.")
    args = parser.parse_args()
    model = args.model

    with open(f'output/pii_detected_{model}.txt', 'w') as file:
        pass

    df = read_file("data/obfuscated_data_06.json")
    nlp_engine = create_nlp_engine(spaCy_model = f"en_core_web_{model}")
    pii_entities_detected = analyze_texts_with_presidio(df, nlp_engine)

    # Append the new detected PII entities to the existing file
    output_file = f"output/pii_detected_{model}.txt"
    append_entities_to_file(pii_entities_detected, output_file)
    print(f"Appended detected PII entities to {output_file}")

    entities = []
    with open(f'output/pii_detected_{model}.txt', 'r') as file:
        for line in file:
            entity = ast.literal_eval(line.strip())
            entities.append(entity)

    # Create the DataFrame
    df = pd.DataFrame(entities, columns=['file_idx', 'entity_text', 'type', 'positions'])
    # Save DataFrame to a CSV file
    df.to_csv(f'output/pii_detected_{model}.csv', index=False)

    test_indices_2 = get_testing_list('data/test_indices_2.txt')
    pii_trf = pd.read_csv('output/pii_detected_trf.csv')
    pii_trf_2 = pii_trf[pii_trf['file_idx'].isin(test_indices_2)]
    pii_trf_2.to_csv('output/pii_detected_trf_2.csv', index=False)

    print("Filtered detected entities have been saved to 'output/pii_detected_trf_2.csv'.")


