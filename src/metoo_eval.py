from metoo_analysis import entity_power_agency
from representations import TokenEmbedding, get_token_embeddings
import config as cfg
import pandas
from match_parse import VerbInstance
from scipy.stats import pearsonr, spearmanr
import pickle
import os
import glob
import h5py
from lexicons import load_power_all
from weighted_tests import build_feature_array, logistic_regression, pred_to_score
import weights
from collections import defaultdict, Counter
import ast
import numpy as np
from sklearn import preprocessing
import argparse
from metoo_utils import get_articles, get_aziz_map, get_annotations, pairwise_compare, score_ents_by_frequency, get_all_pairs

MIN_MENTIONS=3

FILE_TO_NAME={
    "aziz_ansari":"Aziz Ansari",
    "seth_meyers":"Seth Meyers",
    "james_franco":"James Franco",
    "guillermo_del_toro":"Guillermo del Toro",
    "grace":"Grace",
    "natalie_portman":"Natalie Portman",
    "oprah_winfrey":"Oprah Winfrey",
    "harvey_weinstein":"Harvey Weinstein",
    "caitlin_flanagan": "Caitlin Flanagan",
    "margaret_atwood": "Margaret Atwood",
    "bari_weiss": "Bari Weiss",
    "katie_way":"Katie Way",
    "donald_trump":"Donald Trump"}

# Check how well each annotator does individually on this task
def gold_pairwise_compare(article_counts, pairs_to_keep = None):
    article_counts, keys, overall1, overall2 = get_annotations(restrict_match=False)

    keys_to_idx = {k:i for i,k in enumerate(keys)}

    avg = [(a + b)/2 for a,b in zip(overall1, overall2)]

    pairs = get_all_pairs()

    anno1_correct = 0
    anno2_correct = 0
    total  = 0
    for a in article_counts:
        for p in pairs:
            key1 = (a, p[0])
            key2 = (a, p[1])
            pair_key = (a, p)
            if pairs_to_keep is not None and not pair_key in pairs_to_keep:
                continue

            if key1 in keys_to_idx and key2 in keys_to_idx and key1:
                if avg[keys_to_idx[key1]] == avg[keys_to_idx[key2]]:
                    continue

                total += 1
                gold = avg[keys_to_idx[key1]] < avg[keys_to_idx[key2]]
                anno1 = overall1[keys_to_idx[key1]] < overall1[keys_to_idx[key2]]
                anno2 = overall2[keys_to_idx[key1]] < overall2[keys_to_idx[key2]]
                if gold == anno1:
                    anno1_correct += 1
                if gold == anno2:
                    anno2_correct += 1
    print("Anno 1", float(anno1_correct) / total, total)
    print("Anno 2", float(anno2_correct) / total, total)



def compare_corr(entity_to_our_score):
    article_counts, keys, overall1, overall2 = get_annotations(restrict_match=True)

    matched1 = []
    matched2 = []
    ours = []
    matched_keys = []
    for i,key in enumerate(keys):
        if key in entity_to_our_score:
            matched1.append(overall1[i])
            matched2.append(overall2[i])
            ours.append(entity_to_our_score[key])
            matched_keys.append(key)

    print("Num samples", len(ours))
    print("Matched 1", pearsonr(matched1, ours))
    print("Matched 2", pearsonr(matched2, ours))
    print("Matched together", pearsonr(matched2, matched1))

    avg = [(a + b)/2 for a,b in zip(matched1, matched2)]
    print("Average", pearsonr(avg, ours))

def idx_to_sign(idxs):
    clean_idxs = []
    signs = []

    for i in idxs:
        if "[" in i:
            signs.append(-1)
            clean_idxs.append(int(i.replace("[", "").replace("]", "")))
        else:
            signs.append(1)
            clean_idxs.append(int(i))
    return clean_idxs, signs

def process_idx_files(fileglob):
    key_to_embeds = defaultdict(list)
    key_to_signs = defaultdict(list)

    m = []
    for filename in glob.iglob(fileglob):
        basename = os.path.basename(filename)
        file_num = (int(int(basename.split("_")[0]) / 5) * 5) + 5
        ent_name = "_".join(basename.split("_")[1:])
        ent_name = FILE_TO_NAME[ent_name.replace(".txt", "")]

        h5py_file = h5py.File(filename, 'r')
        sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
        idx_to_sent = {int(i):s.split() for s,i in sent_to_idx.items()}

        idx_file = filename.replace("/embeddings/", "/raw_tokenized/").replace(".txt", ".txt.idx")
        for sent_idx,line in enumerate(open(idx_file).readlines()):
            idxs, signs = idx_to_sign(line.split())
            sent = h5py_file.get(str(sent_idx))

            for i,s in zip(idxs,signs):
                if not sent_idx in idx_to_sent:
                    print("Skipping")
                    continue

                word = idx_to_sent[sent_idx][i]

                key_to_embeds[(file_num, ent_name)].append(len(m))
                key_to_signs[(file_num, ent_name)].append(s)
                m.append(sent[1][i]) # we take the second layer

    m = np.vstack(m)
    preprocessing.normalize(m, copy=False)
    return key_to_embeds, key_to_signs, m

def score_keyed_embeddings(key_to_embeds, key_to_signs, m, avg_embeddings):
    train = load_power_all(cfg.POWER_AGENCY)

    train_X, train_Y, _ = build_feature_array(train, avg_embeddings)

    preds = logistic_regression(train_X, train_Y, m, {}, weights.power_token_regression, do_print=False, return_preds=True, is_token=False)
    ent_to_score = {}
    for key,idxs in key_to_embeds.items():
        signs = key_to_signs[key]
        scores = [preds[i] for i in idxs]
        sum_score = sum([p * s for p,s in zip(scores, signs)])
        ent_to_score[key] = sum_score / len(scores)
    return ent_to_score

def score_ents_raw_frames(embeddings, entity_map):
    from spacy.lemmatizer import Lemmatizer
    from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    train = load_power_all(cfg.POWER_AGENCY)

    train = {lemmatizer(t, u'verb')[0]:s for t,s in train.items()}

    article_ids, article_id_to_bucket = get_articles(30)

    # Which entities are portrayed as most postitive?
    entity_to_count = Counter()
    entity_to_score = defaultdict(float)
    for e,idxs in embeddings.entity_to_idx.items():
        if not e in entity_map:
            continue
        e = entity_map[e]
        # this is a verb that has some association with e
        for idx in idxs:
            rel = embeddings.tupls[idx].relation
            verb = lemmatizer(embeddings.tupls[idx].verb_lemma, u'verb')[0]

            article_id = os.path.basename(embeddings.tupls[idx].filename).split(".")[0]
            if not article_id in article_id_to_bucket:
                continue

            if not verb in train:
                continue

            for b in article_id_to_bucket[article_id]:
                key = (b, e)

                if rel == "nsubj":
                    entity_to_score[key] += train[verb]
                    entity_to_count[key] += 1

                # if this is the object or nsubjpass we take the wo score
                if rel == "dobj"  or rel == "nsubjpass":
                    entity_to_score[key] += -1 * train[verb]
                    entity_to_count[key] += 1

    entity_to_final_score = {e:s / entity_to_count[e] for e,s in entity_to_score.items() if entity_to_count[e] >= MIN_MENTIONS}
    # entity_to_final_score = {e:s / entity_to_count[e] for e,s in entity_to_score.items()}
    return entity_to_final_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_cache")
    parser.add_argument("--score_cache", help="optionally cache scores for faster running")
    args = parser.parse_args()


    ARTICLE_COUNTS=[5, 10, 15, 20, 25, 30]
    AGREE_ONLY=False

    embeddings = get_token_embeddings("", [], [], cache_name=args.embedding_cache)
    embeddings.normalize()
    avg_embeddings = embeddings.make_average()

    AZIZ_MAP = get_aziz_map()

    if args.score_cache and os.path.exists(args.score_cache):
        entity_to_our_score = pickle.load(open(args.score_cache, "rb" ))
    else:
        entity_to_our_score = {}
        for c in ARTICLE_COUNTS:
            article_ids, qq = get_articles(c)
            print(c, article_ids, qq)
            entity_to_score, entity_to_count, _ = entity_power_agency(embeddings, avg_embeddings, power=True, filter_set=None, article_filter_set=article_ids, entity_map=AZIZ_MAP)
            for entity,score in entity_to_score.items():
                if entity_to_count[entity] >= MIN_MENTIONS:
                    entity_to_our_score[(c, entity)] = score
        if args.score_cache:
            pickle.dump(entity_to_our_score, open(args.score_cache, "wb"))


    # Gather the scores
    freq_scores = score_ents_by_frequency(embeddings, AZIZ_MAP)
    raw_scores = score_ents_raw_frames(embeddings, AZIZ_MAP)


    # We ultiamtely want evaluation over the same set of pairs, but these metrics
    # work over different pairs sets. Run the evaluation once over everyone to get
    # the pairs. Then run actual evaluations only over pairs we have all scores for

    auto_pairs = pairwise_compare(entity_to_our_score, ARTICLE_COUNTS, agree_only=AGREE_ONLY, verbose=False)
    freq_pairs = pairwise_compare(freq_scores, ARTICLE_COUNTS, agree_only=AGREE_ONLY, verbose=False)
    raw_pairs = pairwise_compare(raw_scores, ARTICLE_COUNTS, agree_only=AGREE_ONLY, verbose = False)
    merged_pairs = auto_pairs.intersection(freq_pairs, raw_pairs)


    # Now run the actual evaluations
    print("COMPLETE AUTOMATED")
    pairwise_compare(entity_to_our_score, ARTICLE_COUNTS, agree_only=AGREE_ONLY, pairs_to_keep = merged_pairs, verbose=True)
    print("FREQ SCORES")
    pairwise_compare(freq_scores, ARTICLE_COUNTS, agree_only=AGREE_ONLY, pairs_to_keep = merged_pairs, verbose=True)
    print("RAW SCORES")
    pairwise_compare(raw_scores, ARTICLE_COUNTS, agree_only=AGREE_ONLY, pairs_to_keep = merged_pairs, verbose=True)
    print("ANNOTATOR SCORES")
    gold_pairwise_compare(ARTICLE_COUNTS, pairs_to_keep = merged_pairs)
    print("#######################################################################################")

    # Run evaluations over all the pairs for the automated approach that we can
    # Compare that to the frequency mentions, since we should have frequency scores for everybody
    merged_pairs = auto_pairs.intersection(freq_pairs)
    print("COMPLETE AUTOMATED")
    pairwise_compare(entity_to_our_score, ARTICLE_COUNTS, agree_only=AGREE_ONLY, pairs_to_keep = merged_pairs, verbose=True)
    print("FREQ SCORES")
    pairwise_compare(freq_scores, ARTICLE_COUNTS, agree_only=AGREE_ONLY, pairs_to_keep = merged_pairs, verbose=True)

    # Now run the hand annotations. We only annotated the first 10 articles
    # This is independent of others
    print("##################### HAND ANNOTATIONS ####################################")
    key_to_embeds, key_to_signs, m = process_idx_files(cfg.AZIZ_HAND_PARSED)
    hand_to_score = score_keyed_embeddings(key_to_embeds, key_to_signs, m, avg_embeddings)
    hand_pairs = pairwise_compare(hand_to_score, [5, 10], agree_only=False, verbose=True)
    print("COMPLETE AUTOMATED")
    pairwise_compare(entity_to_our_score, [5, 10], agree_only=AGREE_ONLY, pairs_to_keep = hand_pairs, verbose=True)
    print("FREQ SCORES")
    pairwise_compare(freq_scores, [5, 10], agree_only=AGREE_ONLY, pairs_to_keep = hand_pairs, verbose=True)
    print("ANNOTATOR SCORES")
    gold_pairwise_compare(ARTICLE_COUNTS, pairs_to_keep = hand_pairs)


    print("###################################### CORRELATIONS #############################")
    compare_corr(entity_to_our_score)
    print("###################################### Inter-annotator correlations #############")
    get_annotations(restrict_match=True, verbose=True)



if __name__ == "__main__":
    main()
