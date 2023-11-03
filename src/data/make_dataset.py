import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

def unzip(path='./data/raw/filtered_paranmt.zip'):
    
    # Open the ZIP file for reading
    with zipfile.ZipFile(path, 'r') as z:
        
        # Extract all the contents to the specified directory
        z.extractall('./data/raw/')

# Function to read the dataset
def read_dataset(path='./data/raw/filtered.tsv'):

    if '.tsv' in path:
        dataset = pd.read_csv(path, sep='\t', index_col=0)
        
    else:

        dataset = pd.read_csv(path, index_col=0)

    return dataset


# Function to preprocess the dataset
def preprocess_data(dataset):
    
    # Extract the columns values and assign them to the variables
    reference = dataset['reference'].values
    translation = dataset['translation'].values
    ref_tox = dataset['ref_tox'].values
    trn_tox = dataset['trn_tox'].values

    for i in range(len(dataset)):
        # Check if the toxicity level of the reference text is less than the toxicity level of the translation text
        if ref_tox[i] < trn_tox[i]:
            ref_tox[i], trn_tox[i] = trn_tox[i], ref_tox[i]  # Swap the toxicity levels
            reference[i], translation[i] = translation[i], reference[i] # Swap the corresponding text contents

    # Write the changed values to the DataFrame
    dataset['reference'] = reference
    dataset['translation'] = translation
    dataset['ref_tox'] = ref_tox
    dataset['trn_tox'] = trn_tox

    return dataset


# Function to filter the dataset based on specified conditions
def cut_dataset(dataset):

    return dataset.loc[(dataset['ref_tox'] > 0.9) & (dataset['trn_tox'] < 0.15) & (dataset['similarity'] > 0.7)]


# Function to transform the dataset into source and target columns
def transform(dataset):

    source = dataset['reference']
    target = dataset['translation']

    data = {'source': source, 'target': target}

    return pd.DataFrame(data)


# Function to split the dataset into train and test set
def dataset_split(dataset):
    
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    return train, test


# Function to execute the entire data processing pipeline
def make_dataset():

    # Read the ZIP file
    unzip()

    # Read the dataset
    df = read_dataset()

    # Preprocess the dataset
    df = preprocess_data(df)

    # Filter the dataset
    df = cut_dataset(df)
    
    # Save the dataset with all columns to an interim CSV file
    df.to_csv('./data/interim/dataset_all_columns.csv', index=False)

    # Transform the dataset into source and target columns
    df = transform(df)

    # Save the dataset with all columns 'source' and 'target' to an interim CSV file
    df.to_csv('./data/interim/dataset.csv', index=False)

    # Split the dataset into train and test sets
    train, test = dataset_split(df)

    # Save the train and test datasets to interim CSV files
    train.to_csv('./data/interim/train.csv', index=False)
    test.to_csv('./data/interim/test.csv', index=False)

    


if __name__ == '__main__':
    make_dataset()
