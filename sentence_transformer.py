from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings

    print('token_embeddings')
    print(token_embeddings)

    input_mask_expanded_unsqueezed = attention_mask.unsqueeze(-1)

    print('input_mask_expanded_unsqueezed')
    print(input_mask_expanded_unsqueezed)

    size = token_embeddings.size()

    print('size')
    print(size)

    input_mask_expanded_from_size = input_mask_expanded_unsqueezed.expand(size)

    print('input_mask_expanded_from_size')
    print(input_mask_expanded_from_size)

    input_mask_expanded = input_mask_expanded_from_size.float()

    print('input_mask_expanded')
    print(input_mask_expanded)

    torch_sum = torch.sum(token_embeddings * input_mask_expanded, 1)

    print('torch_sum')
    print(torch_sum)

    input_mask_expanded_sum = input_mask_expanded.sum(1)

    print('input_mask_expanded_sum')
    print(input_mask_expanded_sum)

    torch_clamp = torch.clamp(input_mask_expanded_sum, min=1e-9)

    print('torch_clamp')
    print(torch_clamp)

    final = torch_sum / torch_clamp

    print('final')
    print(final)

    return final


# Sentences we want sentence embeddings for
sentences = ['Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

