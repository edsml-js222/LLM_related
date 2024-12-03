

def tokenizer():
    return None
def build_vocab(**kwards):
    return None
def en_sentence_preprocess(sentence):
    tokens = [BOS] + tokenizer(sentence) + [EOS]
    ids = en_vocab(tokens)
    return tokens, ids
    


en_data = ['i am your god', 'you made me love']

en_tokens = tokenizer(en_data)

UNK_id, PAD_id, BOS_id, EOS_id = 0, 1, 2, 3
UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'
en_vocab = build_vocab(en_tokens, special_tokens=[UNK, PAD, BOS, EOS], special_first=True, special_tokens_index=[UNK_id, PAD_id, BOS_id, EOS_id])

