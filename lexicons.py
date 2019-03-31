# Load all lexicons
import json
import pandas
import random
from sklearn.model_selection import StratifiedKFold
import os
from collections import Counter, defaultdict
import math

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

def make_binary(score):
    if score < -0.25:
        return -1
    elif score > 0.25:
        return 1
    else:
        return 0

def load_from_json(filename, remove_neutral=False):
    lex =  json.load(open(filename))
    if remove_neutral:
        return {x:lex[x] for x in lex if not lex[x] == 0}
    return lex

def load_connotation_frames(filename, header, binarize, remove_neutral=False, plus_one=True):
    df = pandas.read_csv(filename, sep='\t')
    lex = {}
    for i,row in df.iterrows():
        verb = row["verb"]
        if binarize:
            if row[header] < -0.25:
                lex[verb] = -1
            elif row[header] > 0.25:
                lex[verb] = 1
            else:
                if not remove_neutral:
                    lex[verb] = 0
        else:
            lex[verb] = row[header]
    if plus_one:
        for l in lex:
            lex[l] += 1
    return lex

def load_connotation_split(filename, header, binarize, remove_neutral=False):
    random.seed(1)
    lex = load_connotation_frames(filename, header, binarize, remove_neutral)
    verbs = list(lex.keys())
    random.shuffle(verbs)
    split_len = int(len(verbs) / 3)
    train = {}
    for i in range(split_len):
        v = verbs[i]
        train[v] = lex[v]
    test = {}
    for i in range(split_len, split_len * 2):
        v = verbs[i]
        test[v] = lex[v]
    dev = {}
    for i in range(split_len * 2, len(verbs)):
        v = verbs[i]
        dev[v] = lex[v]
    return train, test, dev

def load_hannah_split(path, header, binarize, remove_neutral=False, plus_one=True):
    test = load_connotation_frames(os.path.join(path, "test_frame_info.txt"), header, binarize, remove_neutral, plus_one)
    train = load_connotation_frames(os.path.join(path, "train_frame_info.txt"), header, binarize, remove_neutral, plus_one)
    dev = load_connotation_frames(os.path.join(path, "dev_frame_info.txt"), header, binarize, remove_neutral, plus_one)

    return train, test, dev

def load_connotation_kfold(filename, header, binarize, remove_neutral=False):
    random.seed(1)
    lex = load_connotation_frames(filename, header, binarize, remove_neutral)
    verbs = list(lex.keys())
    vals = []
    for v in verbs:
        vals.append(lex[v])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
    for tmp_train_index, test_index in skf.split(verbs, vals):
        tmp_train = [verbs[i] for i in tmp_train_index]
        tmp_train_vals = [vals[i] for i in tmp_train_index]
        test = {verbs[i]:vals[i] for i in test_index}
#        return {verbs[i]:vals[i] for i in tmp_train_index}, test, {}

        for train_index, dev_index in skf.split(tmp_train, tmp_train_vals):
            dev = {tmp_train[i]:tmp_train_vals[i] for i in dev_index}
            train = {tmp_train[i]:tmp_train_vals[i] for i in train_index}

            assert len(dev) + len(train) + len(test) == len(lex)
            for t in test:
                assert test[t] == lex[t]
            for t in train:
                assert train[t] == lex[t]
            for t in dev:
                assert dev[t] == lex[t]

            assert len(set(list(dev.keys()) + list(train.keys()) + list(test.keys()))) == len(lex)

            return train, test, dev

def load_power_verbs(filename):
    verbs = set()
    for line in open(filename).readlines():
        l = line.split(",")
        verbs.add(l[0])
    return verbs

def load_power_all(filename):
    train, test, dev = load_power_split(filename)
    train.update(test)
    train.update(dev)
    return train

def load_agency_all(filename):
    train, test, dev = load_agency_split(filename)
    train.update(test)
    train.update(dev)
    return train

def load_power_split(filename):
    random.seed(2)
    df = pandas.read_csv(filename)
    verb_to_score={}
    for idx,row in df.iterrows():
        v = lemmatizer(row.loc["verb"].split()[0], u'verb')[0]
        if (row.loc["power"] == "power_theme"):
            verb_to_score[v] = 0
        elif (row.loc["power"] == "power_equal"):
            verb_to_score[v] = 1
        elif (row.loc["power"] == "power_agent"):
            verb_to_score[v] = 2

    set_size = int(len(verb_to_score) / 3)
    shuffled = list(verb_to_score.keys())
    random.shuffle(shuffled)
    train = {}
    test = {}
    dev = {}

    for v in shuffled[0:set_size]:
        train[v] = verb_to_score[v]
    for v in shuffled[set_size:set_size * 2]:
        test[v] = verb_to_score[v]
    for v in shuffled[set_size * 2:]:
        dev[v] = verb_to_score[v]

    assert len(train) + len(test) + len(dev) == len(verb_to_score)
    return train, test, dev

def load_agency_split(filename):
    random.seed(10)
    df = pandas.read_csv(filename)
    verb_to_score={}
    for idx,row in df.iterrows():
        v = lemmatizer(row.loc["verb"].split()[0], u'verb')[0]
        if (row.loc["agency"] == "agency_neg"):
            verb_to_score[v] = 0
        elif (row.loc["agency"] == "agency_equal"):
            verb_to_score[v] = 1
        elif (row.loc["agency"] == "agency_pos"):
            verb_to_score[v] = 2

    set_size = int(len(verb_to_score) / 3)
    shuffled = list(verb_to_score.keys())
    random.shuffle(shuffled)
    train = {}
    test = {}
    dev = {}

    for v in shuffled[0:set_size]:
        train[v] = verb_to_score[v]
    for v in shuffled[set_size:set_size * 2]:
        test[v] = verb_to_score[v]
    for v in shuffled[set_size * 2:]:
        dev[v] = verb_to_score[v]

    assert len(train) + len(test) + len(dev) == len(verb_to_score)
    return train, test, dev

def check_raw_annotations(filename):
    import config
    df = pandas.read_csv(filename)

    headers=[("Perspective(wo)", "Q15: How the Writer Feels About Object?"),
             ("Perspective(ws)", "Q16: How the Writer Feels About Subject?")] #, "Perspective(so)", "Effect(o)", "Effect(s)", "Value(o)", "Valuee(s)", "State(o)", "State(s)"] #, "Perspective(ro)", "Perspective(rs)", "Perspective(os)"]

    for h1,h2 in headers:
        lex = load_connotation_frames(config.CONNO_FRAME, h1, binarize = False, remove_neutral=False, plus_one=False)

        verb_to_scores = defaultdict(list)
        for i,row in df.iterrows():
            if not math.isnan(row[h2]):
                verb_to_scores[row["Verb"]].append(row[h2])
        for v,scores in verb_to_scores.items():
            score = sum(scores) / len(scores)
            if abs(score - lex[v]) > 0.0001:
                print("MISMATCH", str(score) + " " + str(lex[v]) + " " + v + " " + str(len(scores)))

    for i,row1 in df.iterrows():
        for j,row2 in df.iterrows():
            if row1['Full Sentence'] == row2['Full Sentence']:
                if row1['Verb'] != row2['Verb']:
                    print("VERB", row1['Verb'], row2['Verb'], i, j)

                if row1['SUBJ'] != row2['SUBJ']:
                    print("SUBJ", row1['SUBJ'], row2['SUBJ'], i, j)

                if row1['OBJ'] != row2['OBJ']:
                    print("OBJ", row1['OBJ'], row2['OBJ'], i, j)


            if row1['Verb'] == row2['Verb'] and \
               row1['SUBJ'] == row2['SUBJ'] and \
               row1['OBJ'] == row2['OBJ']:
                if row1['Full Sentence'] != row2['Full Sentence']:
                    print("SENT", row1['Full Sentence'], row2['Full Sentence'], i, j)

def load_raw_annotations(filename, header, binarize, plus_one=False, sent_key=False):
    df = pandas.read_csv(filename)
    sents_to_scores = defaultdict(list)
    for i,row in df.iterrows():
        if sent_key:
            key = (row["Verb"], row["SUBJ"], str(row["OBJ"]), row["Full Sentence"])
        else:
            key = (row["Verb"], row["SUBJ"], str(row["OBJ"]))
        if not math.isnan(row[header]):
            sents_to_scores[key].append(row[header])
    key_to_score = {}
    for key,scores in sents_to_scores.items():
        if len(scores) < 3:
            continue
        s = sum(scores) / len(scores)
        if binarize:
            key_to_score[key] = make_binary(s)
        else:
            key_to_score[key] = s
    if plus_one:
        for k in key_to_score:
            key_to_score[k] += 1
    return key_to_score

def load_nrc_affect(filename, header):
    df = pandas.read_csv(filename, sep="\t")
    word_to_score = {}
    for i,row in df.iterrows():
        word_to_score[row["Word"]] = row[header]
    return word_to_score

def load_nrc_split(filename, header, embeddings):
    random.seed(10)
    valence_to_score = load_nrc_affect(filename, header)
    valence_to_score = {v:valence_to_score[v] for v in valence_to_score if len(embeddings.wi[v]) >= 10}

    num_sample = int(len(valence_to_score) / 10)
    sample = random.sample(list(valence_to_score.keys()), num_sample * 2)
    dev = {d:valence_to_score[d] for d in sample[0:num_sample]}
    test = {t:valence_to_score[t] for t in sample[num_sample:num_sample * 2]}
    train = {t:valence_to_score[t] for t in valence_to_score if not t in dev and not t in test}

    return train, dev, test

if __name__ == "__main__":
    import config
    train, test, dev = load_agency_split(config.POWER_AGENCY)
    print("Train size", len(train))
    train_counts = Counter()
    test_counts = Counter()
    dev_counts = Counter()

    for t,s in train.items():
        train_counts[s] += 1

    for t,s in test.items():
        test_counts[s] += 1

    for t,s in dev.items():
        dev_counts[s] += 1

    print(train_counts, len(train))
    print(test_counts, len(test))
    print(dev_counts, len(dev))
