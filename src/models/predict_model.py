import torch
from transformers import BertTokenizer, BertForMaskedLM
import re
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from tqdm import tqdm


def read_test_data(path='./data/interim/test.csv'):

    # Read test data from a CSV file

    test_df = pd.read_csv(path)

    X = test_df['source'].to_list()
    y = test_df['target'].to_list()

    return X, y


def read_toxic_list(path='./data/external/profanity_words_en.txt'):

    # Read toxic words from an external file

    toxic_words = []

    with open(path, "r") as f:
        for word in f.readlines():
            toxic_words.append(word[:-1])

    return toxic_words


def load_model():

    # Load the BERT tokenizer and model
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    return model, tokenizer

    
def add_mask(sentence, toxic_words):
    
    # Tokenize the sentence
    tokenizer = RegexpTokenizer(r"\b\w+\b|[.,!?'\"]")
    tokens = tokenizer.tokenize(sentence)
    
    mask_sentence = []
    
    for token in tokens:
        
        if token.lower() in toxic_words:
            # If it's toxic, replace it with [MASK]
            mask_sentence.append('[MASK]')
        else:
            # Otherwise, keep the token as is
            mask_sentence.append(token)
            
    result_sentence = ' '.join(mask_sentence)

    # Correctly handle contractions like "It's"
    result_sentence = re.sub(r"(\w+) ' (\w+)", r"\1'\2", result_sentence)

    # Remove spaces before punctuation
    result_sentence = re.sub(r" ([.,!?])", r"\1", result_sentence)

    return result_sentence


def detoxify_sentences(sentences, toxic_words, tokenizer, model):
    
    detoxified_sentences = []
    
    for i in tqdm(range(len(sentences))):
    
        mask_sentence = add_mask(sentences[i], toxic_words)

        if '[MASK]' in mask_sentence:

            # Tokenize the sentence
            inputs = tokenizer(mask_sentence, return_tensors="pt", padding=True, truncation=True)
            
            # Find the positions of the [MASK] tokens in the input
            mask_positions = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)

            # Predict the words for the [MASK] token
            with torch.no_grad():
                predictions = model(**inputs).logits[0]
                
            # Initialize a list to store predicted words
            predicted_words = []

            # Extract the predicted words for each [MASK] token
            for position in mask_positions[0]:
                predicted_word_index = torch.argmax(predictions[position]).item()
                predicted_word = tokenizer.convert_ids_to_tokens(predicted_word_index)
                predicted_words.append(predicted_word)

            # Replace the [MASK] tokens with the predicted words
            for predicted_word in predicted_words:
                mask_sentence = mask_sentence.replace("[MASK]", predicted_word, 1)

        detoxified_sentences.append(mask_sentence)
        
    return detoxified_sentences



def predict():
    
    # Read the dataset
    X, y = read_test_data()
    toxic_words = read_toxic_list()
    model, tokenizer = load_model()

    # Detoxify the sentences
    output = detoxify_sentences(X, toxic_words, tokenizer, model)
    
    for i in [6, 15, 41, 65, 78, 106]:
        print('Predicted: {output}\nTarget: {y}\nSource: {X}\n')

    return output


if __name__ == '__main__':
    output = predict() 
