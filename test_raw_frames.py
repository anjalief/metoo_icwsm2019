import re
import pandas
import config
from representations import TokenEmbedding
from weighted_tests import do_logistic_regression, logistic_regression, build_feature_array, find_logistic_regression_weights
from lexicons import load_hannah_split, check_raw_annotations, load_raw_annotations
from weights import raw_header_to_weights, paper_header_to_weights, ideal_header_to_weights
import argparse
import os

from collections import defaultdict

import spacy
nlp = spacy.load('en_core_web_sm')

import h5py, ast
import numpy as np

from sklearn.metrics import f1_score, accuracy_score

from spacy.lang.en import English
tokenizer = English().Defaults.create_tokenizer(nlp)

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

HEADERS=[("Perspective(wo)", "Q15: How the Writer Feels About Object?"),
         ("Perspective(ws)", "Q16: How the Writer Feels About Subject?")] #, "Perspective(so)", "Effect(o)", "Effect(s)", "Value(o)", "Valuee(s)", "State(o)", "State(s)"] #, "Perspective(ro)", "Perspective(rs)", "Perspective(os)"]

EXCEPTIONS=["focus", "succeed", "proceed", "impress", "dress",
            "need", "possess", "process", "bless", "miss",
            "guess", "rent", "speed", "confess"]

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def do_prep(token_file, meta_file):
    df = pandas.read_csv(config.RAW_CONNOTATIONS)

    sent_to_sov = {}
    for i,row in df.iterrows():
        sent = cleanhtml(row["Full Sentence"])
        sent_to_sov[sent] = (row["Verb"], row["SUBJ"], str(row["OBJ"]))

    out_fp = open(token_file, "w")
    meta_fp = open(meta_file, "w")
    meta_fp.write("Verb,SUBJ,OBJ\n")
    for sent,key in sent_to_sov.items():
        tokens = tokenizer(sent)
        out_fp.write(" ".join([t.text.strip() for t in tokens]) + "\n")
        meta_fp.write(",".join(key) + "\n")

    out_fp.close()
    meta_fp.close()

def load_embeddings(embedding_cache, meta_file, weights=[0, 1, 0]):
  f = h5py.File(embedding_cache, 'r')
  sent_to_idx = ast.literal_eval(f.get("sentence_to_index")[0])

  meta = pandas.read_csv(meta_file)

  vocab = []
  vectors = []
  word_to_idxs = defaultdict(list)
  sents = []
  for sent,sent_idx in sent_to_idx.items():
      row = meta.iloc[int(sent_idx)]
      v = row["Verb"]
      if v == "fulfil":
          v = "fulfill"
      found = False
      for word_idx,w in enumerate(sent.split()):
          if not w in EXCEPTIONS:
            w = lemmatizer(w, u'verb')[0]
          if w == "fulfil":
            w = "fulfill"
          if w == v:
              vocab.append(w)
              word_to_idxs[w].append(len(vocab) - 1)

              s1 = f.get(sent_idx)
              vectors.append(s1[0][word_idx] * weights[0] +
                             s1[1][word_idx] * weights[1] +
                             s1[2][word_idx] * weights[2])

              sents.append(sent)
              found = True
              break
      if not found:
        print("Unable to find", v, "in", sent, [lemmatizer(w, u'verb')[0] for w in sent.split()])
  return TokenEmbedding(np.vstack(vectors), vocab, word_to_idxs, normalize=False, sents=sents)

# What's the as-usual type level accuracy?
def do_normal_regression(headers, avg_embeddings, weights = None):
    for h,h2 in headers:
        print(h)
        train, test, dev = load_hannah_split(config.CONNO_DIR, h, binarize=True, remove_neutral=False)

        if weights is not None:
            print("Running normal type-level regression")
            train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
            test_X, test_Y, _ = build_feature_array(test, avg_embeddings)
            logistic_regression(train_X, train_Y, test_X, test_Y, weights=weights[h], do_print=True, is_token=False, test_words_Y=None, return_preds=True)
        else:
            do_logistic_regression(train, dev, test, avg_embeddings)

# What's the accuracy of using the gold type-level label for each sentence?
def check_gold_labels(headers, embeddings):
    print("Check gold type-level labels")
    for h,h2 in headers:
      print(h)
      train, test, dev = load_hannah_split(config.CONNO_DIR, h, binarize=True, remove_neutral=False, plus_one=False)
      sent_to_score = load_raw_annotations(config.RAW_CONNOTATIONS, h2, binarize=True)
      verbs_tested = set()
      type_level = []
      sent_level = []
      for key,score in sent_to_score.items():
          if key[0] in test:
              type_level.append(test[key[0]])
              sent_level.append(score)
              verbs_tested.add(key[0])
              # if test[key[0]] != score:
              #   print (key[3], score, test[key[0]])

      print("Num tested", len(verbs_tested), len(sent_level))
      print("Macro F1", f1_score(sent_level, type_level,  average='macro'))
      print("Accuracy", accuracy_score(sent_level, type_level))

def load_sent_to_key(meta_file, token_file):
    meta = pandas.read_csv(meta_file)
    sents = open(token_file).readlines()
    assert len(meta) == len(sents)

    sent_to_key = {}
    for i,row in meta.iterrows():
      sent_to_key[sents[i].strip()] = (row["Verb"], row["SUBJ"], row["OBJ"])

    return sent_to_key

# What's the accuracy of using the learned type-level predictions on token-level test data?
def type_to_token_eval(headers, embeddings, avg_embeddings):
    print("What's the accuracy of using the learned type-level predictions on token-level test data?")
    for h,h2 in headers:
        train, test, dev = load_hannah_split(config.CONNO_DIR, h, binarize=True, remove_neutral=False, plus_one=True)
        sent_to_score = load_raw_annotations(config.RAW_CONNOTATIONS, h2, binarize=True, plus_one=True)

        print(h)
        weights = raw_header_to_weights[h]

        train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
        test_X, test_Y, test_words = build_feature_array(test, avg_embeddings)
        preds = logistic_regression(train_X, train_Y, test_X, test_Y, weights=weights, do_print=False, is_token=False, test_words_Y=None, return_preds=True)
        word_to_pred = {}

        for w,s in zip(test_words, preds):
            word_to_pred[w] = s
        print(len(word_to_pred), len(test))


        type_level = []
        sent_level = []
        for key,score in sent_to_score.items():
            if key[0] in word_to_pred:
                type_level.append(word_to_pred[key[0]])
                sent_level.append(score)

        print("Macro F1", f1_score(sent_level, type_level,  average='macro'))
        print("Accuracy", accuracy_score(sent_level, type_level))

def build_sent_array(test, embeds, sent_to_score, sent_to_key):
    X = []
    Y = []
    sents = []

    for w in test:
        if not w in embeds.wi:
            print("Skipping", w)
            continue
        idxs = embeds.wi[w]
        for i in idxs:
            sent = embeds.sents[i]
            key = sent_to_key[sent]
            if key in sent_to_score:
              Y.append(sent_to_score[key])
              X.append(embeds.m[i])
    print("Test size", len(X))
    return X,Y

# What's the accuracy of using type-level training and token-level test? i.e. what we did in the paper
def avg_token_eval(headers, embeddings, avg_embeddings, sent_to_key, weights=None):
    print("What's the accuracy of using type-level training and token-level test? i.e. what we did in the paper")
    for h,h2 in headers:
        train, test, dev = load_hannah_split(config.CONNO_DIR, h, binarize=True, remove_neutral=False, plus_one=True)
        sent_to_score = load_raw_annotations(config.RAW_CONNOTATIONS, h2, binarize=True, plus_one=True)

        print(h)

        train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
        test_X, test_Y = build_sent_array(test, embeddings, sent_to_score, sent_to_key)

        if weights is not None:
          print(weights)
          logistic_regression(train_X, train_Y, test_X, test_Y, weights[h], do_print=True)
        else:
          dev_X, dev_Y = build_sent_array(dev, embeddings, sent_to_score, sent_to_key)

          score, new_weights = find_logistic_regression_weights(train_X, train_Y, dev_X, dev_Y)
          print("Running logistic regression with weights", new_weights, "Dev F1:", score)
          logistic_regression(train_X, train_Y, test_X, test_Y, new_weights, do_print=True)

# What's the accuracy of using token-level training and token-level test? i.e. what we wish we could do
def token_eval(headers, embeddings, sent_to_key, weights = None):
    print("What's the accuracy of using token-level training and token-level test? i.e. what we wish we could do")
    for h,h2 in headers:
        train, test, dev = load_hannah_split(config.CONNO_DIR, h, binarize=True, remove_neutral=False, plus_one=True)
        sent_to_score = load_raw_annotations(config.RAW_CONNOTATIONS, h2, binarize=True, plus_one=True)

        print(h)

        train_X, train_Y = build_sent_array(train, embeddings, sent_to_score, sent_to_key)
        dev_X, dev_Y = build_sent_array(dev, embeddings, sent_to_score, sent_to_key)
        test_X, test_Y = build_sent_array(test, embeddings, sent_to_score, sent_to_key)

        if weights is not None:
          logistic_regression(train_X, train_Y, test_X, test_Y, weights[h], do_print=True)

        else:
          score, new_weights = find_logistic_regression_weights(train_X, train_Y, dev_X, dev_Y)
          print("Running token-level logistic regression with weights", new_weights, "Dev F1:", score)
          logistic_regression(train_X, train_Y, test_X, test_Y, new_weights, do_print=True)

def print_all(headers, embeds, avg_embeds, sent_to_key):
    do_normal_regression(headers, avg_embeds, raw_header_to_weights)
    print("######################################################################################################")
    check_gold_labels(headers, embeds)
    print("######################################################################################################")
    type_to_token_eval(headers, embeds, avg_embeds)
    print("######################################################################################################")
    avg_token_eval(headers, embeds, avg_embeds, sent_to_key, paper_header_to_weights)
    print("######################################################################################################")
    token_eval(headers, embeds, sent_to_key, ideal_header_to_weights)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file")
    parser.add_argument("--token_file")
    parser.add_argument("--embeddings_cache")
    parser.add_argument("--from_scratch", action='store_true', help="relearn weights instead of using cached weights from weights.py")
    args = parser.parse_args()

    if not os.path.exists(args.meta_file):
        print("Preparing annotations for ELMo extraction")
        do_prep(args.token_file, args.meta_file)
        print("Now run", "source activate py36 && allennlp elmo [token_file] [embeddings_file] --all")
        return
    if not os.path.exists(args.embeddings_cache):
        print("Error: you need to generate the embedding cache")
        return

    embeds = load_embeddings(args.embeddings_cache, args.meta_file, weights=[0, 1, 0])
    embeds.normalize()
    avg_embeddings = embeds.make_average()
    sent_to_key = load_sent_to_key(args.meta_file, args.token_file)

    if args.from_scratch:
        print("Running evals from scratch")
        # Just run the things we need to learn weights for
        do_normal_regression(HEADERS, avg_embeddings)
        print("######################################################################################################")
        avg_token_eval(HEADERS, embeds, avg_embeddings, sent_to_key)
        print("######################################################################################################")
        token_eval(HEADERS, embeds, sent_to_key)
        # This optionally does some sanity checking (make sure we can use SUBJ.VERB.OBJ key
        # check_raw_annotations(config.RAW_CONNOTATIONS)
    else:
        print_all(HEADERS, embeds, avg_embeddings, sent_to_key)


if __name__ == "__main__":
    main()
