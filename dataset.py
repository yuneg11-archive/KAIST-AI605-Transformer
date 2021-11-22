import os
import re
from typing import Iterable, Union
from string import digits, ascii_lowercase, ascii_uppercase

import torch
from torch.utils.data import Dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datasets


__all__ = [
    "DigitDataset",
    "TypoDataset",
]


class Vocab:
    valid_vocab_re = re.compile(r"^[0-9a-zA-Z-' ]+$")
    numeric_vocab_re = re.compile(r"^[0-9-]+$")
    vocabs = ["_", "[", "]", *digits, *ascii_lowercase, *ascii_uppercase, "-", "'", " "]
    to_idx_dict = {vocab: idx for idx, vocab in enumerate(vocabs)}
    to_vocab_dict = {idx: vocab for idx, vocab in enumerate(vocabs)}
    char_vocabs = vocabs[3:]
    num_vocabs = len(to_idx_dict)
    num_control_vocabs = 3
    num_valid_vocabs = num_vocabs - num_control_vocabs
    num_char_vocabs = len(char_vocabs)
    num_digit_vocabs = 10

    @classmethod
    def to_idx(cls, char: str) -> int:
        return cls.to_idx_dict[char]

    @classmethod
    def to_vocab(cls, idx: int) -> str:
        return cls.to_vocab_dict[idx]

    @classmethod
    def sample_digits(cls, size: int = 1, generator=None) -> str:
        nums = torch.randint(0, 10, size=(size,), generator=generator).tolist()
        tokens = "".join([str(d) for d in nums])
        return tokens

    @classmethod
    def sample_chars(cls, size: int = 1, generator=None) -> str:
        idxs = torch.randint(3, 66, size=(size,), generator=generator).tolist()
        tokens = "".join([cls.to_vocab(i) for i in idxs])
        return tokens

    @classmethod
    def is_valid_vocabs(cls, vocabs: Union[str, Iterable[str]]) -> bool:
        return cls.valid_vocab_re.fullmatch(vocabs) is not None

    @classmethod
    def is_numeric_vocabs(cls, vocabs: Union[str, Iterable[str]]) -> bool:
        return cls.numeric_vocab_re.fullmatch(vocabs) is not None


def get_collate_fn(pad_tensor=None):
    if pad_tensor is None:
        def collate_fn(batch):
            source = torch.vstack([d[0].unsqueeze(dim=0) for d in batch])
            target = torch.vstack([d[1].unsqueeze(dim=0) for d in batch])
            source_mask = target_mask = None
            return source, target, source_mask, target_mask
    else:
        def collate_fn(batch):
            max_len_source = max(len(d[0]) for d in batch)
            max_len_target = max(len(d[1]) for d in batch)
            source = torch.full((len(batch), max_len_source), pad_tensor, dtype=torch.long)
            target = torch.full((len(batch), max_len_target), pad_tensor, dtype=torch.long)
            source_mask = torch.full((len(batch), max_len_source), True, dtype=torch.bool)
            target_mask = torch.full((len(batch), max_len_target), True, dtype=torch.bool)
            for i, d in enumerate(batch):
                source[i, :len(d[0])] = d[0]
                target[i, :len(d[1])] = d[1]
                source_mask[i, :len(d[0])] = False
                target_mask[i, :len(d[1])] = False
            return source, target, source_mask, target_mask
    return collate_fn


class NLPDataset(Dataset):
    pad_token = "_"
    start_token = "["
    end_token = "]"
    vocab_size = Vocab.num_vocabs
    collate_fn = get_collate_fn(pad_tensor=Vocab.to_idx(pad_token))

    def __init__(self, seed: int = 109):
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

    def tokens_to_idx(self, tokens: Union[str, Iterable[str]]):
        return torch.tensor([Vocab.to_idx(self.start_token)]
                          + [Vocab.to_idx(v) for v in tokens]
                          + [Vocab.to_idx(self.end_token)], dtype=torch.long)

    def idx_to_tokens(self, idx: torch.Tensor):
        start_idx = Vocab.to_idx(self.start_token)
        end_idx = Vocab.to_idx(self.end_token)
        tokens = []
        for i in (idx[1:] if idx[0] == start_idx else idx).tolist():
            if i == end_idx:
                break
            tokens.append(Vocab.to_vocab(i))
        return "".join(tokens)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx: int):
        return self.source[idx], self.target[idx]


class DigitDataset(NLPDataset):
    pad_token = None
    vocab_size = Vocab.num_control_vocabs + Vocab.num_digit_vocabs
    collate_fn = get_collate_fn(pad_tensor=None)

    def __init__(self, num_data: int, token_len: int, seed: int = 109):
        super().__init__(seed=seed)
        data = [self.tokens_to_idx(Vocab.sample_digits(token_len, generator=self.rng))
                for _ in range(num_data)]
        self.source = self.target = data


class TypoDataset(NLPDataset):
    def __init__(
        self,
        num_data: int,
        split: str = "train",
        level: str = "word",
        noise_rate: float = 0.9,
        seed: int = 109,
    ):
        super().__init__(seed=seed)
        self.split = split
        self.level = level
        self.noise_rate = noise_rate
        squad = datasets.load_dataset("squad", split=("validation" if split == "valid" else split))
        sentences = []
        for context in sorted(set(squad["context"])):
            sentences.extend([s for s in (context + " ").split(". ")
                              if Vocab.is_valid_vocabs(s) and len(s) > 10])
        if level == "word":
            words = []
            for sentence in sentences:
                words.extend([w for w in sentence.split(" ")
                              if not Vocab.is_numeric_vocabs(w) and len(w) > 1])
            words = sorted(set(words))
            idxs = torch.randint(len(words), size=(num_data,), generator=self.rng)
            self.raw_data = [words[i] for i in idxs]
            should_perterb = (torch.rand((num_data,), generator=self.rng) < noise_rate).tolist()
            self.source = [self.tokens_to_idx(self._perturb(s) if p else s)
                           for s, p in zip(self.raw_data, should_perterb)]
            self.target = [self.tokens_to_idx(s) for s in self.raw_data]
        elif level == "sentence":
            sentences = [s for s in sentences if len(s) <= 48]
            idxs = torch.randint(len(sentences), size=(num_data,), generator=self.rng)
            self.raw_data = [sentences[i] for i in idxs]
            should_perterb = (torch.rand((num_data,), generator=self.rng) < noise_rate).tolist()
            self.source = [self.tokens_to_idx(self._perturb(s) if p else s)
                           for s, p in zip(self.raw_data, should_perterb)]
            self.target = [self.tokens_to_idx(s) for s in self.raw_data]
        else:
            raise ValueError(f"Invalid level: {level}")

    def _perturb(self, tokens: str):
        token_len = len(tokens)
        while True:
            t = torch.randint(13, size=(1,), generator=self.rng).item()
            i = torch.randint(token_len - 1, size=(1,), generator=self.rng).item()
            if t < 3:  # Swap
                return tokens[:i] + tokens[i + 1] + tokens[i] + tokens[i + 2:]
            elif t < 6:  # Insert
                v = Vocab.sample_chars(generator=self.rng)
                return tokens[:i] + v + tokens[i:]
            elif not tokens[i].isdigit():
                if t < 9:  # Delete
                    return tokens[:i] + tokens[i + 1:]
                elif t < 12:  # Replace
                    v = Vocab.sample_chars(generator=self.rng)
                    return tokens[:i] + v + tokens[i + 1:]
                else:  # Change case
                    v = tokens[i].upper() if tokens[i].islower() else tokens[i].lower()
                    return tokens[:i] + v + tokens[i + 1:]
