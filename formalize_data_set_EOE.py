from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaForCausalLM
import torch
import numpy as np
# from question_refine_generate import question_distillation, get_domain_and_d_score, select_distinct_data_points, Shrink_dataset, efficient_select_distinct_data_points
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import random
import json, torch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b").to(device)

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2Model.from_pretrained("gpt2").to(device)

def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        # Ensure that the model outputs hidden states
        outputs = model(**inputs, output_hidden_states=True)
    # Get the last hidden state
    last_hidden_state = outputs.hidden_states[-1]
    # Retrieve the last token's embedding
    return last_hidden_state[:, -1, :].cpu().numpy()


# def get_embedding(text, tokenizer, model, device):
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)  # Moved to device
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Return the embedding of the last token and move it back to CPU
#     return outputs.last_hidden_state[:, -1, :].cpu().numpy()


def calculate_EOE(text, embedding, tokenizer):
    def calculate_normalized_entropy(embedding, num_tokens):
        hist, bin_edges = np.histogram(embedding, bins=100, density=True)
        probabilities = hist * np.diff(bin_edges)
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        normalized_entropy = entropy / (np.log2(num_tokens + 1))
        return normalized_entropy

    num_tokens = len(tokenizer.tokenize(text))
    return calculate_normalized_entropy(embedding, num_tokens)


def calculate_DSS(text, lexicon_path='./lexicon_v5'):
    def read_lexicons(path='./lexicon_v5'):
        lexicons = {}
        files = os.listdir(path)
        for lexicon_file in files:
            file_path = os.path.join(path, lexicon_file)
            with open(file_path, 'r') as f:
                for line in f:
                    _, words = line.split('\t', 1)
                    lexicons[lexicon_file[:-4]] = [word.strip() for word in words.split(',')]
        return lexicons

    lexicon_dict = read_lexicons(lexicon_path)
    tokenized_text = word_tokenize(text)
    lexicon_scores = {}

    # Calculate how many words in the text are in each lexicon
    for lexicon_name, lexicon_words in lexicon_dict.items():
        lexicon_set = set(lexicon_words)
        matching_words = [word for word in tokenized_text if word.lower() in lexicon_set]
        score = len(matching_words) / len(tokenized_text) if tokenized_text else 0
        lexicon_scores[lexicon_name] = score

    # Find the lexicon type with the highest ratio (i.e., domain tag)
    highest_lexicon = max(lexicon_scores, key=lexicon_scores.get) if lexicon_scores else None
    avg_lexicon_score = sum(lexicon_scores.values()) / len(lexicon_dict) if lexicon_dict else 0

    return highest_lexicon, avg_lexicon_score


def calculate_IDD(primary_sector, text_emb, data_dict):
    emb_list = []
    for t in primary_sector:
        emb_list.append(data_dict[t]['emb'])
    cos_sim = []
    for emb in emb_list:
        cos_sim.append(cosine_similarity(text_emb, emb))
    return np.mean(cos_sim)

def get_buffer_size(data_buffer):
    buffer_size = 0
    for domain in data_buffer:
        buffer_size += len(data_buffer[domain])
    return buffer_size


# Organize the entire dataflow
data_dict = {
    'text_content': {'emb': 0, 'EOE': 0, 'DSS': 0, 'IDD': 0, 'domain': 'string', 'original_input': 'string'},
}

# Initialize data buffer
domains = os.listdir('./lexicon_v5')
domains = [domain[:-4] for domain in domains]
data_buffer = {}
for domain in domains:
    data_buffer[domain] = []
buffer_size = 64
corresponding_size = buffer_size / 3

# Initialize data
dataset_name = 'dataset_1-alpaca'
num = 1
save_path = f'./new_final_data/exp3/EOE/{dataset_name}'
data_path = f'./final_data/initial_data/{dataset_name}/dataset_{str(num)}_ft_bank.json'
with open(data_path, 'r') as f:
    prepared_data = json.load(f)

# Check if the directory exists, and create it if it does not
if not os.path.exists(save_path):
    os.makedirs(save_path)

# checkpoints
checkpoints = [0, 800, 1600, 2400, 3200, 4000, 4600]
# checkpoints = [0, 4600]

checkpoint_index = 0
to_be_write = []
for input in prepared_data:
    if checkpoint_index == 0:
        current_checkpoint = checkpoints[1]
    elif checkpoint_index > 4600:
        break
    else:
        for j in range(len(checkpoints)):
            if checkpoints[j - 1] < checkpoint_index < checkpoints[j]:
                current_checkpoint = checkpoints[j]
                break
    if checkpoint_index == current_checkpoint:
        print(f'Checkpoint {current_checkpoint} reached.')

    text_content = input['instruction'] + ' ' + input['input'] + ' ' + input['output']
    emb = get_embedding(text_content, tokenizer, model, device)
    EOE = calculate_EOE(text_content, emb, tokenizer)
    domain, DSS = calculate_DSS(text_content)
    if len(data_buffer[domain]) != 0:
        IDD = calculate_IDD(data_buffer[domain], emb, data_dict)
    else:
        IDD = 0

    data_dict[text_content] = {
        'emb': emb,
        'EOE': EOE,
        'DSS': DSS,
        'IDD': IDD,  # Initialize IDD
        'domain': domain,
        'original_input': input
    }

    if get_buffer_size(data_buffer) < buffer_size:
        data_buffer[domain].append(text_content)

    # if checkpoint_index < current_checkpoint:
    #     temp_domain_buffer = data_buffer[domain]
    #     ### To do: find the text content whose EOE, DSS, and IDD are the lowest, compare and see the if current one is better. if yes, replace
    # if current_checkpoint == current_checkpoint:
    #     ''' TO DO
    #     save all "original_input" into to_be_write
    #     jsom.dump(to_be_write, f'{save_path}/{dataset_name}_{current_checkpoint}.json')
    #     clean data_buffer's list

    #     '''
    if checkpoint_index < current_checkpoint:
        temp_domain_buffer = data_buffer[domain]
        # Find text content with the lowest EOE, DSS, and IDD
        if temp_domain_buffer:
            lowest_values_content = min(temp_domain_buffer,
                                        key=lambda x: (data_dict[x]['EOE']))
            if data_dict[lowest_values_content]['EOE'] > EOE:
                # Replace the text content in the buffer
                temp_domain_buffer[temp_domain_buffer.index(lowest_values_content)] = text_content

    if checkpoint_index == current_checkpoint:
        # Save all "original_input" into to_be_write
        temp_to_be_write = [data_dict[text]['original_input'] for domain in data_buffer for text in data_buffer[domain]]
        to_be_write += temp_to_be_write
        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)

        # Save to JSON file
        with open(f'{save_path}/{dataset_name}_{current_checkpoint}.json', 'w') as f:
            json.dump(to_be_write, f)

        # Clean data_buffer's list
        for domain in data_buffer:
            data_buffer[domain].clear()

        # Move to the next checkpoint
        if checkpoint_index < checkpoints[-1]:
            current_checkpoint = checkpoints[checkpoints.index(current_checkpoint) + 1]

    checkpoint_index += 1
