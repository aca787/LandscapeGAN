from __future__ import print_function
import time
import random
import torch
from labelling.InferSent.models import InferSent


def get_all_labels(self, label_path='./DatasetWHScrapper/Wallhaven/**/labels.txt'):
    import glob
    label_names = []
    for f in glob.glob(label_path, recursive=True):
        label_names.append(f)
        print(f)

    all_labels = []
    all_images = []
    all_sentences = []
    for file_name in label_names:
        f = open(file_name, 'r')
        #all_data += f.read()
        for line in f:
            if("429 Too Many Requests" in line):
                continue
            all_images.append(line.split('|')[0].strip())
            labels = line.split('|')[1].split(',')
            sentence = []
            for l in labels:
                if l  in ['<s>', '\n','</s>']:
                    continue
                if ' ' not in l:
                    sentence.append(l)
                else:
                    sentence+=l.strip().split(' ')
            all_labels+=sentence 
            all_sentences.append(' '.join(sentence))
            #all_labels+=[[l] for l in labels if ' ' not in l else l.split(' ')]

    #all_sentences = [sentence for sentence in all_sentences if sentence.islower()]    
    all_labels = [label.strip() for label in all_labels if label.islower()]
    all_labels = list(set(all_labels)) #remove duplicates
    return all_labels, all_sentences
class SentEmbedding:
    def __init__(self, label_path='./DatasetWHScrapper/Wallhaven/**/labels.txt'):
        pass

    def create_vocabulary(self, model_path = 'labelling/encoder/infersent1.pkl'):
        #V = 1
        #MODEL_PATH = 'labelling/encoder/infersent%s.pkl' % V
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(model_path))

        W2V_PATH = 'labelling/GloVe/glove.840B.300d.txt'
        self.model.set_w2v_path(W2V_PATH)
        # Load embeddings of K most frequent words
        self.model.build_vocab_k_words(K=100000)

    def embed(self, sentences):
        result = self.model.encode(sentences, tokenize=True)
        return result
