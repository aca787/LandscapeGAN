from __future__ import print_function
import time
import random
import torch

import numpy as np
from scipy import spatial


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
        return self.model.encode(sentences, tokenize=True)
        
    
class GloveEmbedding:
    def __init__(self, label_path='labelling/GloVe/glove.6B.100d.txt'):
        self.label_path = label_path
        pass

    def create_vocabulary(self):
        self.embeddings_dict = {}
        with open(self.label_path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
    
    def find_closest_embeddings(self, embedding):
        return sorted(self.embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(self.embeddings_dict[word], embedding))

    def embed(self, words):
        if (isinstance(words, list)):
            return np.array([self.embeddings_dict[word] for word in words])
        elif (isinstance(words, str)): return self.embeddings_dict[words]
        else:
            raise Exception("[Word Embedding] Unsupported type")
        
    
if __name__=='__main__':
    gv = GloveEmbedding()
    gv.create_vocabulary()
    print(gv.find_closest_embeddings(gv.embed("beach"))[0:64])
    print(gv.find_closest_embeddings(gv.embed("forest"))[0:64])
    print(gv.find_closest_embeddings(gv.embed("mountain"))[0:64])
    print(gv.find_closest_embeddings(gv.embed("desert"))[0:64])
    print(gv.find_closest_embeddings(gv.embed("sea"))[0:64])
