import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from model import GPTConfig, GPT

class Tokenizer():
    def __init__(self,block_size,tokenizer = None,un_tokenizer = None):
        self.tokenizer = tokenizer
        self.un_tokenizer = un_tokenizer
        self.block_size = block_size
        if tokenizer:
            self.vocab_size = len(self.tokenizer.keys())
        else:
            self.vocab_size = None

    def build_tokenization(self,train_sentences):
        # character level tokenization
        vocab = set(''.join(train_sentences))
        self.tokenizer = dict(zip(vocab,range(1,len(vocab)+1)))
        self.tokenizer['<pad>'] = 0
        vocab.add('<pad>')
        self.un_tokenizer = {v:k for k,v in self.tokenizer.items()}
        self.vocab_size = len(vocab)        

    def tokenize(self, sentences):
        encoded = np.full((len(sentences), self.block_size), self.tokenizer['<pad>'], dtype=int)
        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence[:self.block_size]):
                encoded[i, j] = self.tokenizer.get(char, self.tokenizer['<pad>'])
        attention_mask = ~(encoded==self.tokenizer['<pad>'])
        return encoded,attention_mask

    def untokenize(self,encoded):
        sentences = []
        for i in range(len(encoded)):
          sentence = ''
          for j in range(len(encoded[i])):
            if encoded[i][j] != self.tokenizer["<pad>"]:
                sentence += self.un_tokenizer[encoded[i][j]]
          sentences.append(sentence)
        return sentences
    

def load_mod_data(tokenizer,data_path="part2/"):
    # Read files
    train = pd.read_csv(data_path + 'data_train.csv')
    val = pd.read_csv(data_path + 'data_val.csv')
    test = pd.read_csv(data_path + 'data_test.csv')
    
    # Create strs from equations
    convert_equation_to_str(train)
    convert_equation_to_str(val)
    convert_equation_to_str(test)

    # Build tokenizer
    tokenizer.build_tokenization(train['as_str'])

    # Tokenize
    train_tokens,train_mask = tokenizer.tokenize(train['as_str'])
    val_tokens,val_mask = tokenizer.tokenize(val['as_str'])
    test_tokens,test_mask = tokenizer.tokenize(test['as_str'])


    train = torch.from_numpy(train_tokens)
    train_mask = torch.from_numpy(train_mask)
    val = torch.from_numpy(val_tokens)
    val_mask = torch.from_numpy(val_mask)
    test = torch.from_numpy(test_tokens)
    test_mask = torch.from_numpy(test_mask)
        
        
    train_dataset = TensorDataset(train, train_mask)
    val_dataset = TensorDataset(val, val_mask)
    test_dataset = TensorDataset(test, test_mask)

    
    return train_dataset,val_dataset,test_dataset

def convert_equation_to_str(df):
  df['as_str'] = df[['a','o','b']].astype(str).agg(' '.join, axis=1) \
                      + " = " + df['c'].astype(str)
