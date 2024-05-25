import nltk
import pickle
from collections import Counter
import json
import argparse
import os
import pdb

annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'coco': ['annotations/captions_train2014.json',
             'annotations/captions_val2014.json'],
    'f8k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    '10crop_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f8k': ['dataset_flickr8k.json'],
    'f30k': ['dataset_flickr30k.json'],
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)



def from_txt(txt):
    captions = []
    with open(txt, 'r') as f:
        text = f.read()
    for line in text.split('\n'):
        desc = line.split()[1:]
        desc = ' '.join(desc)
        captions.append(desc.strip())
    return captions


def build_vocab(data_path, threshold=None):
    """Build a simple vocabulary wrapper."""
    counter = Counter()

    full_path = os.path.join(data_path, 'Flickr8k.token.txt')
    captions = from_txt(full_path)
    for i, caption in enumerate(captions):
        tokens = nltk.tokenize.word_tokenize(
            caption.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


# def main(data_path):
#     vocab = build_vocab(data_path, threshold=4)
#     data = {}
#     with open('./vocab/%s_vocab.pkl' %'flickr', 'wb') as f:
#         data['v2i'] = vocab.word2idx
#         data['i2v'] = vocab.idx2word
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#     print("Saved vocabulary file to ", './vocab/%s_vocab.pkl' %'flickr')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', default='/ssd_scratch/cvit/deep/Flickr-8K')
#     opt = parser.parse_args()
#     main(opt.data_path)