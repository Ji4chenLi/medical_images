# TODO: there is still weird word inside the text. We can clean this up later

import pickle
from collections import Counter
from utils import load_txt, is_nan


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<end>')
        self.add_word('<start>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1

    def get_word_by_id(self, id):
        return self.id2word[id]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(text_data_path, threshold):
    text_dict = load_txt(text_data_path)
    counter = Counter()

    for _, values in text_dict.items():
        findings = values['findings']
        impression = values['impression']
        if not is_nan(findings):
            text = findings.replace('.', '').replace(',', '')
            counter.update(text.lower().split(' '))
        if not is_nan(impression):
            text = impression.replace('.', '').replace(',', '')
            counter.update(text.lower().split(' '))

    words = [word for word, cnt in counter.items() if cnt > threshold and word != '']
    vocab = Vocabulary()

    for word in words:
        print(word)
        vocab.add_word(word)
    return vocab


def main(text_data_path, threshold, vocab_path):
    vocab = build_vocab(text_data_path=text_data_path,
                        threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size:{}".format(len(vocab)))
    print("Saved path in {}".format(vocab_path))


if __name__ == '__main__':
    main(text_data_path='./preprocessed/text_data.pkl',
         threshold=0,
         vocab_path='./preprocessed/vocab.pkl')
