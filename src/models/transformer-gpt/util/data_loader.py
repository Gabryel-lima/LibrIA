"""
@author : Gabryel-lima
@when : 2025-01-30
@homepage : https://github.com/Gabryel-lima
"""
from torchtext.data import Field, BucketIterator # --> torchtext.legacy.data <= 0.11.0
from torchtext.datasets import Multi30k # --> torchtext.legacy.datasets.translation <= 0.11.0 


class DataLoader:
    def __init__(self, data_root, ext, tokenize_de, tokenize_en, init_token, eos_token):
        self.data_root   = data_root
        self.ext         = ext        # e.g. ('.de', '.en')
        self.tokenize_de = tokenize_de
        self.tokenize_en = tokenize_en
        self.init_token  = init_token
        self.eos_token   = eos_token
        self.source      = None
        self.target      = None

    def make_dataset(self):
        self.source = Field(tokenize=self.tokenize_de,
                    init_token=self.init_token,
                    eos_token=self.eos_token,
                    lower=True, batch_first=True)
        
        self.target = Field(tokenize=self.tokenize_en,
                    init_token=self.init_token,
                    eos_token=self.eos_token,
                    lower=True, batch_first=True)

        # Garante que os arquivos existem / baixa se precisar
        Multi30k.download(self.data_root)

        # Carrega os splits a partir do diretório correto
        train, val, test = Multi30k.splits(
            root=self.data_root,
            exts=self.ext,
            fields=(self.source, self.target)
        )
        return train, val, test
    
    def build_vocab(self, train_data, min_freq):
        """Constrói os vocabulários a partir do conjunto de treino."""
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train_data, valid_data, test_data, batch_size, device):
        """Cria iteradores (BatchIterator) para treino, validação e teste."""
        return BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            device=device,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src)
        )
