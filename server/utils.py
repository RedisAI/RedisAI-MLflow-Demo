import redisai
import pickle
from transformers import GPT2Tokenizer
import torch

tokenizer_class = GPT2Tokenizer
pretrained_weights = 'gpt2'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
temperature = 0.7
top_k = 0
top_p = 0.9
max_length = 200

con = redisai.Client()


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def init_context():
    with open('past.tensor', 'rb') as f:
        return pickle.load(f).detach().numpy()


def init_conversation(prompt_text):
    encoded_prompts = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded_prompts
    for i in range(input_ids.shape[1] - 1):
        single = input_ids[:, i].unsqueeze(-1).numpy()
        con.tensorset('conversation', single)
        con.modelrun('gptmodel', inputs=['conversation', 'context'], outputs=['out', 'context'])
    return input_ids[:, -1].unsqueeze(-1).numpy()


def process_output(out):
    next_token_logits = torch.from_numpy(out)[:, -1, :]
    next_token_logits = next_token_logits / temperature
    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1).unsqueeze(0).detach().numpy()


def ids2text(words):
    return tokenizer.decode(words, clean_up_tokenization_spaces=True)


def print2terminal(prompt_text, generated_sequence):
    print("=== GENERATED SEQUENCE ===")
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    total_sequence = prompt_text + text
    print(total_sequence)
