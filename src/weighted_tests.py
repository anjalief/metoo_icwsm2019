# A lot of this is copied from test_connotation_frames, moving
# to a new file so it can be cleaned up
from lexicons import load_hannah_split, load_power_split, load_agency_split
import config
from representations import Embedding, get_token_embeddings, TokenEmbedding
import argparse

from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import weights

from sklearn.metrics import f1_score, accuracy_score
from match_parse import VerbInstance

HEADERS=["Perspective(wo)", "Perspective(ws)"]
# headers=["Perspective(wo)",	"Perspective(ws)", "Perspective(so)", "Effect(o)", "Effect(s)", "Value(o)", "Valuee(s)", "State(o)", "State(s)"] #, "Perspective(ro)", "Perspective(rs)", "Perspective(os)"]

def build_feature_array(lex, embeds, get_sents=False):
    Y = []
    X = []
    words = []
    sents = []

    for w in lex:
        if not w in embeds.wi:
            # print(w)
            continue
        idxs = embeds.wi[w]
        if type(idxs) == list:
            for i in idxs:
                X.append(embeds.m[i])
                Y.append(lex[w])
                words.append(w)
                if embeds.sents:
                    sents.append(embeds.sents[i])
        else:
            X.append(embeds.m[idxs])
            Y.append(lex[w])
            words.append(w)
    if get_sents:
        return np.vstack(X), np.array(Y), words, sents

    return np.vstack(X), np.array(Y), words

def find_logistic_regression_weights(train_X, train_Y, test_X, test_Y, verbose=True, is_token=False, test_words_Y=None):
    weights = np.linspace(0.05, 0.95, 20)
    score_to_weights = {}
    for w1 in weights:
        weights2 = np.linspace(0.04, 1.0 - w1, 20)
        for w2 in weights2:
            # for reg in weights:
            class_weights = {0: w1, 1: w2, 2: 1.0-w1-w2}
            score = logistic_regression(train_X, train_Y, test_X, test_Y, weights=class_weights, is_token=is_token, test_words_Y=test_words_Y)
            if verbose:
                print("Macro F1", score, class_weights)
            score_to_weights[score] = class_weights
    w = sorted(score_to_weights)[-1]
    return w, score_to_weights[w]

def pred_to_score(Y, preds, num_classes):
    num_words = len(set(Y))
    counts = np.zeros((num_words, num_classes))

    w_to_idx = {}
    idx = 0
    for w in set(Y):
        w_to_idx[w] = idx
        idx += 1

    for i,w in enumerate(Y):
        idx = w_to_idx[w]
        counts[idx][preds[i]] += 1

    new_labels = np.argmax(counts, axis=1)
    w_to_label = {}
    for w,idx in w_to_idx.items():
        w_to_label[w] = new_labels[idx]
    return w_to_label

def logistic_regression(train_X, train_Y, test_X, test_Y, weights=None, do_print=False, is_token=False, test_words_Y=None, return_preds=False, reg = 1.0):
    # Logistic Regression
    if reg is not None:
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial', class_weight=weights, C=reg, penalty='l2').fit(train_X, train_Y)
    else:
        clf = LogisticRegression(random_state=0, solver='lbfgs',
                                 multi_class='multinomial', class_weight=weights).fit(train_X, train_Y)
    pred = clf.predict(test_X)

    # Need to convert token level to type to get accuracies
    if is_token:
        w_to_label = pred_to_score(test_words_Y, pred, 3)
        true_to_label = pred_to_score(test_words_Y, test_Y, 3)
        final_pred = []
        test_Y = []
        for w in w_to_label:
            final_pred.append(w_to_label[w])
            test_Y.append(true_to_label[w])
    else:
        final_pred = pred

    if do_print:
        print("Logistic Regression")
        print("Macro F1", f1_score(test_Y, final_pred,  average='macro'))
        print("Accuracy", accuracy_score(final_pred, test_Y))

    if return_preds:
        return pred

    return f1_score(test_Y, final_pred,  average='macro')

def do_logistic_regression(train, dev, test, avg_embeddings):
    train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
    dev_X, dev_Y, _ = build_feature_array(dev, avg_embeddings)
    test_X, test_Y, _ = build_feature_array(test, avg_embeddings)

    score, weights = find_logistic_regression_weights(train_X, train_Y, dev_X, dev_Y)
    class_weights = weights
    print("Running logistic regression with weights", weights, "Dev F1:", score)
    logistic_regression(train_X, train_Y, test_X, test_Y, class_weights, do_print=True)

def do_token_avg_logistic_regression(train, dev, test, embeddings, avg_embeddings):
    train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
    dev_X, dev_Y, dev_words_Y = build_feature_array(dev, embeddings)
    test_X, test_Y, test_words_Y = build_feature_array(test, embeddings)

    score, weights = find_logistic_regression_weights(train_X, train_Y, dev_X, dev_Y, is_token=True, test_words_Y=dev_words_Y)
    print("Running token average logistic regression with weights", weights, "Dev F1:", score)
    logistic_regression(train_X, train_Y, test_X, test_Y, weights, do_print=True, is_token=True, test_words_Y=test_words_Y)


def majority_class(test, embeddings):
    print("Majority Class")
    ytrue=[]
    ymaj=[]
    for t in test:
        if not t in embeddings.wi:
            continue
        ytrue.append(test[t])
        ymaj.append(2)

    print("Macro F1", f1_score(ytrue, ymaj,  average='macro'))
    print("Accuracy", accuracy_score(ytrue, ymaj))

def format_runs(embeddings, avg_embeddings, train, test, dev):
    print("Starting type level")
    do_logistic_regression(train, dev, test, avg_embeddings)
    print("Starting token level")
    do_token_avg_logistic_regression(train, dev, test, embeddings, avg_embeddings)

def run_connotation_frames(embeddings, avg_embeddings):
    for header in HEADERS:
        print(header)
        train, test, dev = load_hannah_split(config.CONNO_DIR, header, binarize=True, remove_neutral=False)
        format_runs(embeddings, avg_embeddings, train, test, dev)


    print("Power")
    train, test, dev = load_power_split(config.POWER_AGENCY)
    format_runs(embeddings, avg_embeddings, train, test, dev)
    majority_class(test, embeddings)

    print("Agency")
    train, test, dev = load_agency_split(config.POWER_AGENCY)
    format_runs(embeddings, avg_embeddings, train, test, dev)
    majority_class(test, embeddings)

def do_weighted_run(train, test, dev, embeddings, avg_embeddings, type_weights, token_weights):
    print("Type level")
    train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
    dev_X, dev_Y, dev_words_Y = build_feature_array(dev, avg_embeddings)
    test_X, test_Y, test_words_Y = build_feature_array(test, avg_embeddings)
    logistic_regression(train_X, train_Y, test_X, test_Y, type_weights, do_print=True, is_token=False, test_words_Y=test_words_Y)

    print("Token level")
    dev_X, dev_Y, dev_words_Y = build_feature_array(dev, embeddings)
    test_X, test_Y, test_words_Y = build_feature_array(test, embeddings)
    logistic_regression(train_X, train_Y, test_X, test_Y, token_weights, do_print=True, is_token=True, test_words_Y=test_words_Y)

def paper_runs(embeddings, avg_embeddings):
    import weights as wgts
    for header in HEADERS:
        train, test, dev = load_hannah_split(config.CONNO_DIR, header, binarize=True, remove_neutral=False)
        print("###################################", header, "###################################")
        do_weighted_run(train, test, dev, embeddings, avg_embeddings, wgts.metoo_header_to_weights[header], wgts.metoo_header_to_avg_token_weights[header])


    print("################################### Power ###################################")
    train, test, dev = load_power_split(config.POWER_AGENCY)
    do_weighted_run(train, test, dev, embeddings, avg_embeddings, wgts.power_logistic_regression, wgts.power_token_regression)
    majority_class(test, embeddings)

    print("################################### Agency ###################################")
    train, test, dev = load_agency_split(config.POWER_AGENCY)
    do_weighted_run(train, test, dev, embeddings, avg_embeddings, wgts.agency_logistic_regression, wgts.agency_token_regression)
    majority_class(test, embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache")
    parser.add_argument("--from_scratch", action='store_true')
    args = parser.parse_args()

    # other parameters don't matter when we read from the cache
    embeddings = get_token_embeddings("", [], weights=[0,1,0], cache_name=args.cache)
    embeddings.normalize()
    avg_embeddings = embeddings.make_average()

    if args.from_scratch:
        run_connotation_frames(embeddings, avg_embeddings)
    else:
        paper_runs(embeddings, avg_embeddings)


if __name__ == "__main__":
    main()
