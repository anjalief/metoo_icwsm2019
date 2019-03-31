import glob
from sklearn import preprocessing
from collections import defaultdict
import numpy as np
import h5py
import ast
import pickle
import os
import random
from sklearn.decomposition import PCA
from collections import Counter

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.symbols import nsubj, VERB

class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, vecs, vocab, normalize=True):
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = {w:i for i,w in enumerate(self.iw)}
        if normalize:
            self.normalize()

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    def oov(self, w):
        return not (w in self.wi)

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def to_file(self, file_name):
        fp = open(file_name, "w")

        for w in range(len(self.iw)):
            fp.write(self.iw[w] + " ")
            for i in self.m[w]:
                fp.write(str(i) + " ")
            fp.write("\n")
        fp.close()

    def filter_down(self, words):
        new_vocab = []
        new_idx = {}
        new_m = []
        for i,w in enumerate(words):
            if not w in self.wi:
                continue
            new_vocab.append(w)
            new_idx[w] = i
            new_m.append(self.m[self.wi[w]])
        self.m = np.vstack(new_m)
        self.dim = self.m.shape[1]
        self.iw = new_vocab
        self.wi = new_idx

    def pca(self):
        print ("Before PCA:", self.m.shape)
        self.m = PCA(n_components=300).fit_transform(self.m)
        print ("After PCA:", self.m.shape)

class TokenEmbedding(Embedding):
    """ Main difference is that self.iw is a {word:[idx, idx]} since we can have
    multiple embeddings for the same word"""
    def __init__(self, vecs, vocab, idx_dict, normalize=True, isolate_word=None, sents=None, entity_dict=None, file_to_sents=None, tupls=None):
        self.m = vecs
        self.dim = self.m.shape[1]
        self.iw = vocab
        self.wi = idx_dict
        if normalize:
            self.normalize()
        self.isolate_word = isolate_word
        self.sents = sents
        self.entity_to_idx = entity_dict
        self.file_to_sents = file_to_sents
        self.tupls = tupls

    def filter_down(self, words):
        new_vocab = []
        new_idx_dict = defaultdict(list)
        new_m = []
        i = 0
        for w in enumerate(words):
            if not w in self.wi:
                continue
            for idx in self.wi[w]:
                new_vocab.append(w)
                new_idx_dict[w].append(i)
                new_m.append(m[idx])
                i += 1

        self.m = np.vstack(new_m)
        self.dim = self.m.shape[1]
        self.iw = new_vocab
        self.wi = new_idx_dict

    # average embeddings for all other words, except this one
    # do this to pair down space
    def isolate_word(self, word):
        new_vocab = []
        new_idx_dict = defaultdict(list)
        new_m = []
        i = 0
        for w,idxs in self.wi.items():
            if w == word:
                for idx in idxs:
                    new_vocab.append(w)
                    new_idx_dict[w].append(i)
                    new_m.append(self.m[idx])
                    i += 1
            else:
                mean_vec = []
                for idx in idxs:
                    mean_vec.append(self.m[idx])

                new_vocab.append(w)
                new_idx_dict[w].append(i)
                new_m.append(np.mean(mean_vec, axis=0))
                i += 1
        return TokenEmbedding(np.vstack(new_m), new_vocab, new_idx_dict, normalize=True, isolate_word=word)

    def renormalize(self, embeds):
        print(self.m)
        new_m = np.vstack([self.m, embeds])
        print(new_m.shape)
        # print(new_m)
        preprocessing.normalize(new_m, copy=False)
        self.m = new_m[:len(self.m)]
        # print(self.m)

       # p rint(self.m.shape)
        return new_m[len(self.m):]

    def make_average(self):
        new_vocab = []
        new_idx_dict = defaultdict(list)
        new_m = []
        i = 0
        for w,idxs in self.wi.items():
            mean_vec = []
            for idx in idxs:
                mean_vec.append(self.m[idx])

            new_vocab.append(w)
            new_idx_dict[w].append(i)
            new_m.append(np.mean(mean_vec, axis=0))
            i += 1
        return TokenEmbedding(np.vstack(new_m), new_vocab, new_idx_dict, normalize=True)


    def resample(self, size, smallset):
        num_dupls = int(size / len(smallset)) + 1
        num_new_samples = size - len(smallset)
        if num_new_samples <= 0:
            return Counter(smallset)

        sample_from = []
        for i in range(num_dupls):
            sample_from += smallset

        new_sample = random.sample(sample_from, num_new_samples)
        return Counter(new_sample + smallset)

    def make_balanced(self, seed_sets, test, mult0 = 0.75, mult1 = 0.75):
        positive = seed_sets[0]
        neutral = seed_sets[1]
        negative = seed_sets[2]

        if len(positive) >= len(neutral) and len(positive) >= len(negative):
            largest = positive
            smaller = [neutral, negative]
        elif len(neutral) >= len(positive) and len(neutral) >= len(negative):
            largest = neutral
            smaller = [positive, negative]
        else:
            assert len(negative) >= len(neutral) and len(negative) >= len(neutral)
            largest = negative
            smaller = [positive, neutral]

        c = self.resample(int(len(largest) * mult0), smaller[0])
        c = c + self.resample(int(len(largest) * mult1), smaller[1])

        # assumes we don't have any overlap between train and test
        for t in test:
            c[t] = 1
        for t in largest:
            c[t] = 1

        # c tells us how many instances of each word we want in the new embedding
        new_vocab = []
        new_idx_dict = defaultdict(list)
        new_m = []
        i = 0
        for w,idxs in self.wi.items():
            num_instances = c[w]
            if num_instances == 0:
                continue

            step = int(len(idxs) / num_instances)
            if (step == 0):
                break
            for count in range(0, len(idxs) - step + 1, step):
                mean_vec = []
                for idx in idxs[count:count+step]:
                    mean_vec.append(self.m[idx])

                new_vocab.append(w)
                new_idx_dict[w].append(i)
                new_m.append(np.mean(mean_vec, axis=0))
                i += 1
        return TokenEmbedding(np.vstack(new_m), new_vocab, new_idx_dict, normalize=True)

    def to_file(self, file_name):
        fp = open(file_name, "w")

        for w in range(len(self.iw)):
            for embed in self.iw[w]:
                fp.write(self.iw[w] + " ")
                for i in self.m[embed]:
                    fp.write(str(i) + " ")
                fp.write("\n")
        fp.close()


def elmo_verbs_from_glob(elmo_glob, lexicon, weights, stem = False):
    if stem:
        words = set([lemmatizer(w, u'verb')[0] for w in lexicon])
    else:
        words = lexicon

    word_to_embeds = defaultdict(list)
    word_to_sents = defaultdict(list)

    for f in glob.iglob(elmo_glob):
        try:
            h5py_file = h5py.File(f, 'r')
            sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
            for sent,sent_idx in sent_to_idx.items():
                doc = nlp(sent)
                for word_idx,token in enumerate(doc):
                    if token.pos != VERB:
                        continue

                    if stem:
                        w = token.lemma_
                    else:
                        w = token.text

                    if w in words:
                        s1 = h5py_file.get(sent_idx)
                        word_to_embeds[w].append(s1[0][word_idx] * weights[0] +
                                                 s1[1][word_idx] * weights[1] +
                                                 s1[2][word_idx] * weights[2])
                        word_to_sents[w].append(sent)
        except:
            continue
    return word_to_embeds, word_to_sents


def elmo_embeds_from_glob(elmo_glob, lexicon, weights, stem = False):
    if stem:
        words = set([lemmatizer(w, u'verb')[0] for w in lexicon])
    else:
        words = lexicon

    word_to_embeds = defaultdict(list)
    word_to_sents = defaultdict(list)

    for f in glob.iglob(elmo_glob):
        try:
            h5py_file = h5py.File(f, 'r')
            sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
            for sent,sent_idx in sent_to_idx.items():
                for word_idx,w in enumerate(sent.split()):
                    if stem:
                        w = lemmatizer(w, u'verb')[0]

                    if w in words:
                        s1 = h5py_file.get(sent_idx)
                        word_to_embeds[w].append(s1[0][word_idx] * weights[0] +
                                                 s1[1][word_idx] * weights[1] +
                                                 s1[2][word_idx] * weights[2])
                        word_to_sents[w].append(sent)
        except:
            continue
    return word_to_embeds, word_to_sents

# elmo_glob should have h5py files outputted by allennlp command
# for each word in lexicon, we average over all words in lexicon
def get_elmo_embeddings(elmo_glob, lexicon, weights= [0, 0, 1], cache_name = None, n_normalize=True, stem=False):
    if cache_name is not None:
        if os.path.exists(cache_name):
            return pickle.load(open(cache_name, "rb" ))

    word_to_embeds, word_to_sents = elmo_embeds_from_glob(elmo_glob, lexicon, weights, stem)

    vocab = []
    vectors = []
    for word in word_to_embeds:
        vocab.append(word)
        vectors.append(np.mean(word_to_embeds[word], axis=0))

    vectors = np.vstack(vectors)
    assert vectors.shape[1] == 1024, "Weird vector length: " + str(vectors.shape)
    embed = Embedding(vectors, vocab, normalize=n_normalize)
    if cache_name is not None:
        pickle.dump(embed, open(cache_name, "wb"))
    return embed

def embeds_from_file(filename, words):
    vocab = []
    vecs = []
    for line in open(filename).readlines():
        split = line.split()
        w = split[0]
        if w in words:
            vocab.append(w)
            vecs.append(np.array([float(x) for x in split[1:]], dtype='float32'))
    vectors = np.vstack(vecs)
    embeds = Embedding(vectors, vocab, normalize=True)
    return embeds

def get_masked_embeddings(embed_glob, weights, cache_name):
    if os.path.exists(cache_name):
        return pickle.load(open(cache_name, "rb" ))

    word_to_embeds = defaultdict(list)
    word_to_sents = defaultdict(list)
    for f in glob.iglob(embed_glob):
        try:
            h5py_file = h5py.File(f, 'r')
            sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])

            keyword = os.path.basename(f).replace(".hdf5", "")

            for sent,sent_idx in sent_to_idx.items():
                for word_idx,w in enumerate(sent.split()):
                    if w == "UNK":
                        s1 = h5py_file.get(sent_idx)
                        word_to_embeds[keyword].append(s1[0][word_idx] * weights[0] +
                                                 s1[1][word_idx] * weights[1] +
                                                 s1[2][word_idx] * weights[2])
                        word_to_sents[keyword].append(sent)
        except:
            print("Skipping", f)
            continue
    print(len(word_to_embeds))
    vocab = []
    vectors = []
    word_to_idxs = defaultdict(list)
    sents = []
    i = 0
    for word, vecs in word_to_embeds.items():
        assert (len(vecs) == len(word_to_sents[word]))
        for v in vecs:
            vocab.append(word)
            vectors.append(v)
            word_to_idxs[word].append(i)
            i += 1
        for s in word_to_sents[word]:
            sents.append(s)

    vectors = np.vstack(vectors)
    assert vectors.shape[1] == 1024, "Weird vector length: " + str(vectors.shape)
    embed = TokenEmbedding(vectors, vocab, word_to_idxs, normalize=True, sents=sents)
    if cache_name is not None:
        pickle.dump(embed, open(cache_name, "wb"), protocol=4)
    return embed


def get_token_embeddings(elmo_glob, lexicon, weights= [0, 0, 1], cache_name = None, n_normalize=True, stem=False, limit_verbs=False):
    if cache_name is not None:
        if os.path.exists(cache_name):
            return pickle.load(open(cache_name, "rb" ))

    if limit_verbs:
        word_to_embeds, word_to_sent = elmo_verbs_from_glob(elmo_glob, lexicon, weights, stem)
    else:
        word_to_embeds, word_to_sent = elmo_embeds_from_glob(elmo_glob, lexicon, weights, stem)

    vocab = []
    vectors = []
    word_to_idxs = defaultdict(list)
    sents = []
    i = 0
    for word, vecs in word_to_embeds.items():
        assert (len(vecs) == len(word_to_sent[word]))
        for v in vecs:
            vocab.append(word)
            vectors.append(v)
            word_to_idxs[word].append(i)
            i += 1
        for s in word_to_sent[word]:
            sents.append(s)

    vectors = np.vstack(vectors)
    assert vectors.shape[1] == 1024, "Weird vector length: " + str(vectors.shape)
    embed = TokenEmbedding(vectors, vocab, word_to_idxs, normalize=n_normalize, sents=sents)
    if cache_name is not None:
        pickle.dump(embed, open(cache_name, "wb"), protocol=4)
    return embed



if __name__ == "__main__":
    from lexicons import load_from_json, load_connotation_frames, load_nrc_affect, load_nrc_split
    import config
    import seeds
#    lexicon = load_from_json(config.INQUIRER_LEX, remove_neutral=True)

#     lexicon = load_connotation_frames(config.CONNO_FRAME, "Perspective(wo)", binarize=True, remove_neutral=False)

# #     GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots/*.hdf5"
# #     GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/metoo_pull2/*.hdf5"
# #     GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots/19999286.xml.hdf5"
#     CACHE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/connotation_frames_nyt010_token_sents_stem_verbs.pickle"
#     GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/nyt/*.hdf5"

#     embeddings = get_token_embeddings(GLOB, list(lexicon), weights=[0, 1, 0], cache_name=CACHE, n_normalize=False, stem=True, limit_verbs=True)

    # valence_to_score = load_nrc_affect(config.NRC_AFFECT, "Dominance")
    # GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/metoo_pull2/*.hdf5"
    # CACHE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/metoo_nrc_lex010.pickle"

    # # GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots/*.hdf5"
    # CACHE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movieplots_nrc_lex010.pickle"
    # embeddings = get_token_embeddings("", {}, weights= [0, 1, 0], cache_name = CACHE, n_normalize=True, stem=False, limit_verbs=False)

    # GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots_nrc_masked/*.hdf5"
    # NEW_CACHE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots_masked_lex010.pickle"
    # train, dev, test = load_nrc_split(config.NRC_AFFECT, "Dominance", embeddings)
    # masks = [str(l) + "123456789" for l in test]
    # embeddings = get_token_embeddings(GLOB, masks, weights= [0, 1, 0], cache_name = NEW_CACHE, n_normalize=True, stem=False, limit_verbs=False)

    GLOB="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots_nrc_sampled_masked_fixed/*.hdf5"
    CACHE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/movie_plots_all_masked_lex010.pickle"
    get_masked_embeddings(GLOB, [0, 1, 0], CACHE)
