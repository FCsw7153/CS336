import regex as re
from collections import defaultdict, Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size:int, special_tokens: list[str]):
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    special_tokens_id = {}
    for special_token in special_tokens:
        encoded = special_token.encode('utf-8')
        if encoded not in vocab.values():
            token_id = len(vocab)
            special_tokens_id[special_token] = token_id
            vocab[token_id] = encoded

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = []
    if special_tokens:
        special_pattern = f"({'|'.join(re.escape(st) for st in special_tokens)})"
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    corpus = []
    for chunk in chunks:
        if chunk in special_tokens:
            token_id = special_tokens_id[chunk]
            corpus.append((token_id,))
        else:
            for match in re.findall(PAT, chunk):
                corpus.append(tuple(match.encode('utf-8')))
    
    word_freq = Counter(corpus)
    pair_counts = defaultdict(int)
    word_pairs = defaultdict(list)

    for word, freq in word_freq.items():
        if len(word) > 1:
            pairs = list(zip(word, word[1:]))
            word_pairs[word] = pairs
            for pair in pairs:
                pair_counts[pair] += freq
    
    num_merges = vocab_size - len(vocab)

    merges = []
    for _ in range(num_merges):
        if not pair_counts:
            break

        top_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))

        p1, p2 = top_pair
        merges.append((vocab[p1], vocab[p2]))
        new_token = len(vocab)
        vocab[new_token] = vocab[p1] + vocab[p2]
        
        new_word_freq = defaultdict(int)
        words_to_process = []
        for word, freq in word_freq.items():
            if top_pair in word_pairs[word]:
                words_to_process.append((word, freq))
            else:
                new_word_freq[word] = freq
        
        for word, freq in words_to_process:
            for pair in word_pairs[word]:
                pair_counts[pair] -= freq
            
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i + 1] == p2:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if len(new_word) > 1:
                new_pairs = list(zip(new_word, new_word[1:]))
                word_pairs[tuple(new_word)] = new_pairs
                for pair in new_pairs:
                    pair_counts[pair] += freq

            new_word_freq[tuple(new_word)] = freq
            
            del word_pairs[word]

        word_freq = new_word_freq

        del pair_counts[top_pair]


    return vocab, merges
