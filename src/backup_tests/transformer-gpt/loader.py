"""
@author : Gabryel-lima
@when : 2025-01-30
@homepage : https://github.com/Gabryel-lima
"""
from conf import CFG, __DEVICE__
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer


def get_dataloaders():
    tokenizer = Tokenizer()
    loader = DataLoader(
        ext=('.en', '.de'),
        tokenize_en=tokenizer.tokenize_en,
        tokenize_de=tokenizer.tokenize_de,
        init_token='<sos>',
        eos_token='<eos>'
    )

    train, valid, test = loader.make_dataset()
    loader.build_vocab(train_data=train, min_freq=2)

    train_iter, valid_iter, test_iter = loader.make_iter(
        train, valid, test,
        batch_size=CFG.batch_size,
        device=__DEVICE__
    )

    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi['<sos>']
    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)

    return {
        'train_iter': train_iter,
        'valid_iter': valid_iter,
        'test_iter': test_iter,
        'src_pad_idx': src_pad_idx,
        'trg_pad_idx': trg_pad_idx,
        'trg_sos_idx': trg_sos_idx,
        'enc_voc_size': enc_voc_size,
        'dec_voc_size': dec_voc_size,
        'tokenizer': tokenizer,
        'loader': loader
    }