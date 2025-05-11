"""
@author : Gabryel-lima
@when : 2025-01-30
@homepage : https://github.com/Gabryel-lima
"""
# from torchtext.legacy.data import Field, BucketIterator # --> torchtext.legacy.data <= 0.11.0
from torchtext.datasets import Multi30k # --> torchtext.legacy.datasets.translation <= 0.11.0 

class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator

# +++ #

# Acima da versão 0.11.0 do torchtext. Deve ser implementado na mão...

# Agora tem que fazer na mão...
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader as PTDataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class DataLoader:
    def __init__(self,
                 ext: tuple[str,str],
                 init_token: str = '<sos>',
                 eos_token: str = '<eos>'):
        # ext = ('.de', '.en') ou ('.en', '.de')
        self.ext = ext
        # decide os idiomas a partir de ext
        if ext == ('.de', '.en'):
            self.src_lang, self.tgt_lang = 'de', 'en'
        else:
            self.src_lang, self.tgt_lang = 'en', 'de'

        # tokenizadores spaCy (precisa ter os modelos instalados)
        self.tok_src = get_tokenizer("spacy", language=f"{self.src_lang}_core_web_sm")
        self.tok_tgt = get_tokenizer("spacy", language=f"{self.tgt_lang}_core_web_sm")

        self.init_token = init_token
        self.eos_token  = eos_token

        # vocabulários (serão criados em build_vocab)
        self.vocab_src = None
        self.vocab_tgt = None

    def make_dataset(self):
        """
        Retorna três iteradores (DataPipes) brutos de pares (src, tgt),
        sem batching nem padding.
        """
        train_dp, valid_dp, test_dp = Multi30k(
            split=("train","valid","test"),
            language_pair=(self.src_lang, self.tgt_lang)
        )
        return train_dp, valid_dp, test_dp

    def build_vocab(self, train_dp, min_freq: int = 2):
        """
        Constrói os vocabulários de fonte e alvo a partir
        do iterador de treino.
        """
        def yield_tokens(data_pipe, tokenizer, index):
            for src, tgt in data_pipe:
                text = src if index == 0 else tgt
                yield tokenizer(text)

        # vocabulário da língua fonte
        self.vocab_src = build_vocab_from_iterator(
            yield_tokens(train_dp, self.tok_src, index=0),
            specials=["<unk>", "<pad>", self.init_token, self.eos_token]
        )
        self.vocab_src.set_default_index(self.vocab_src["<unk>"])

        # vocabulário da língua alvo
        self.vocab_tgt = build_vocab_from_iterator(
            yield_tokens(train_dp, self.tok_tgt, index=1),
            specials=["<unk>", "<pad>", self.init_token, self.eos_token]
        )
        self.vocab_tgt.set_default_index(self.vocab_tgt["<unk>"])

    def _collate_fn(self, batch: list[tuple[str,str]]):
        """
        Recebe um batch cru de pares (src_str, tgt_str),
        tokeniza, numera, adiciona <sos>/<eos> e faz padding.
        Retorna dois tensores [max_len x batch_size].
        """
        src_batch, tgt_batch = [], []
        for src_str, tgt_str in batch:
            # seq com tokens iniciais e finais
            tokens_src = ([self.init_token] +
                          self.tok_src(src_str) +
                          [self.eos_token])
            tokens_tgt = ([self.init_token] +
                          self.tok_tgt(tgt_str) +
                          [self.eos_token])

            nums_src = [self.vocab_src[token] for token in tokens_src]
            nums_tgt = [self.vocab_tgt[token] for token in tokens_tgt]

            src_batch.append(torch.tensor(nums_src, dtype=torch.long))
            tgt_batch.append(torch.tensor(nums_tgt, dtype=torch.long))

        # pad to the max length in batch
        src_padded = pad_sequence(src_batch,
                                  padding_value=self.vocab_src["<pad>"])
        tgt_padded = pad_sequence(tgt_batch,
                                  padding_value=self.vocab_tgt["<pad>"])
        return src_padded, tgt_padded

    def make_iter(self,
                  train_dp, valid_dp, test_dp,
                  batch_size: int,
                  device: torch.device):
        """
        Envolve cada DataPipe num DataLoader do PyTorch
        usando o collate_fn para padding.
        """
        train_iter = PTDataLoader(train_dp,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=self._collate_fn)
        valid_iter = PTDataLoader(valid_dp,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=self._collate_fn)
        test_iter  = PTDataLoader(test_dp,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=self._collate_fn)

        # opcional: mover lotes para device já no loop de treino
        return train_iter, valid_iter, test_iter
    
