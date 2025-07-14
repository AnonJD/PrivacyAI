import pandas as pd
import random

#file to create base train set, verifier train set, and test set.

def read_csv(path = 'data/pii_true_entities.csv'):
    df = pd.read_csv(path,encoding='utf-8')
    return df

def read_json(path = 'data/obfuscated_data_06.json'):
    df = pd.read_json(path, orient="records",encoding='utf-8')
    return df

def count_entities(df, file_indices):
    """
    Counts the number of true entities for each category and overall entities in the given file indices.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the entity information.
        file_indices (list): A list of file indices to consider for counting.
    
    Returns:
        dict: A dictionary containing the counts for each entity type and the total count.
    """
    # Filter the DataFrame for the given file indices
    df_filtered = df[df['file_idx'].isin(file_indices)]
    
    # Group by type and count the number of occurrences
    entity_counts = df_filtered['type'].value_counts().to_dict()

    # Calculate the total number of entities
    total_entities = sum(entity_counts.values())
    
    # Add the total count to the dictionary
    entity_counts['TOTAL'] = total_entities
    
    return entity_counts


if __name__ == '__main__':
    df_true = pd.read_csv('data/pii_true_entities.csv')

    # Set the seed for reproducibility
    random.seed(46)

    # Step 1: Define the total range of indices
    total_indices = list(range(22688))

    # Step 2: Randomly sample 25% of the elements for training indices
    train_indices = random.sample(total_indices, int(len(total_indices) * 0.25))

    # Step 3: Determine the remaining elements for testing indices
    test_indices = list(set(total_indices) - set(train_indices))

    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)

    train_entity_counts = count_entities(df_true, train_indices)
    test_entity_counts = count_entities(df_true, test_indices)

    # Display the results
    print(f"Total number of train indices: {len(train_indices)}")
    print(f"Total number of test indices: {len(test_indices)}")
    print("Entity counts in train files:", train_entity_counts)
    print("Entity counts in test files:", test_entity_counts)

    df_true_train = df_true[df_true['file_idx'].isin(train_indices)]
    df_true_test = df_true[df_true['file_idx'].isin(test_indices)]
    df_true_train.to_csv('data/train_set.csv', index=False)
    df_true_test.to_csv('data/test_set.csv', index=False)


    random.seed(5)


    # Step 2: Randomly sample 20% of the elements for training indices
    train_indices_2 = random.sample(test_indices, int(len(test_indices) * 0.2))
    # Step 3: Determine the remaining elements for testing indices
    test_indices_2 = list(set(test_indices) - set(train_indices_2))

    # Optionally, sort the indices to maintain order
    train_indices_2 = sorted(train_indices_2)
    test_indices_2 = sorted(test_indices_2)

    # Display the results
    print(f"Total number of train indices 2: {len(train_indices_2)}")
    print(f"Total number of test indices 2: {len(test_indices_2)}")


    
    # Calculate the number of true entities for each category and overall for train files
    train_entity_counts_2 = count_entities(df_true, train_indices_2)
    # Calculate the number of true entities for each category and overall for test files
    test_entity_counts_2 = count_entities(df_true, test_indices_2)
    # Display the results
    print("Entity counts in train files 2:", train_entity_counts_2)
    print("Entity counts in test files 2:", test_entity_counts_2)

    # Save df_true_train and df_true_test to CSV
    df_true_train_2 = df_true[df_true['file_idx'].isin(train_indices_2)]
    df_true_test_2 = df_true[df_true['file_idx'].isin(test_indices_2)]

    df_true_train_2.to_csv('data/train_set_2.csv', index=False)
    df_true_test_2.to_csv('data/test_set_2.csv', index=False)

    with open('data/train_indices.txt', 'w') as f:
        f.write(str(train_indices))

    with open('data/test_indices.txt', 'w') as f:
        f.write(str(test_indices))
    
    with open('data/train_indices_2.txt', 'w') as f:
        f.write(str(train_indices_2))

    with open('data/test_indices_2.txt', 'w') as f:
        f.write(str(test_indices_2))

    print("Train and test DataFrames have been saved to 'train_set_2.csv' and 'test_set_2.csv'.")
    print("Train and test DataFrames have been saved to 'train_set.csv' and 'test_set.csv'.")
