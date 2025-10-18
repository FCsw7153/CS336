import regex as re
from collections import defaultdict, Counter
from typing import Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode('utf-8') for i in self.special_tokens] 

        self.bytes_to_id={v:k for k,v in vocab.items()}

        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

    def _apply_merge(self, word_bytes):
        while True:
            pairs = list(zip(word_bytes, word_bytes[1:]))
            count = len(self.vocab)
            c_pair = (1, 1)
            for pair in pairs:
                if pair in self.merges:
                    tmp = self.bytes_to_id[pair[0] + pair[1]]
                    if tmp < count:
                        count = tmp
                        c_pair = (pair[0], pair[1])
            
            if count == len(self.vocab):
                break
            i = 0
            corpus: List[int] = []
            n = len(word_bytes)
            while i < n:
                if i + 1 < n and word_bytes[i] == c_pair[0] and word_bytes[i + 1] == c_pair[1]:
                    corpus.append(c_pair[0] + c_pair[1])
                    i += 2
                else:
                    corpus.append(word_bytes[i])
                    i += 1
            word_bytes = corpus
        return word_bytes

    def encode_merged(self, text):
        word_list = re.findall(PAT, text)
        tokens=[]
        for word in word_list:
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            merged_word_bytes = self._apply_merge(word_bytes)
            tokens.extend(self.bytes_to_id[i] for i in merged_word_bytes)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            yield from self.encode(text)

    def encode(self, text: str):
        chunks = []
        if not self.special_tokens:
            chunks = [text]
        else:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
            chunks = re.split(f"({pattern})", text)
        
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.bytes_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(self.encode_merged(chunk))
        return tokens

    def decode(self, ids: list[int]):
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')