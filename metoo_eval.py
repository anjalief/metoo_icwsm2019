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

def get_articles(num=5):
    urls = []
    for line in open(cfg.AZIZ_ANNOTATED).readlines():
        urls.append(line.split(",")[1].strip())

    df = pandas.read_csv(cfg.METOO_META, sep="\t", header=None)
    url_to_article_id = {}
    for i,row in df.iterrows():
        url_to_article_id[row.iloc[4]] = i

    article_id_to_buckets = {}
    article_ids = []
    for i,url in enumerate(urls):
        if not url in url_to_article_id:
            print("Skipping", url)
            continue

        a = str(url_to_article_id[url])
        if i < num:
            article_ids.append(a)
        first_appearance = (int((i / 5)) * 5) + 5
        article_id_to_buckets[a] = [j for j in range(first_appearance, num + 1, 5)]

    return article_ids, article_id_to_buckets

def get_annotations(restrict_match, verbose=False):
    anno1 = pandas.read_csv(cfg.AZIZ_SET1_ANNOTATIONS)
    anno2 = pandas.read_csv(cfg.AZIZ_SET2_ANNOTATIONS)

    overall1 = []
    overall2 = []
    keys = []

    # first look at correlations at each timestep
    for col in anno1.columns[1:]:
        col1 = []
        col2 = []
        for i,row in anno1.iterrows():
            assert (row[0] == anno2.iloc[i, 0])
            l1 = row[col]
            l2 = anno2.iloc[i].loc[col]
            if l1 == "absent" or l2 == "absent":
                continue
            l1 = int(l1)
            l2 = int(l2)

            if restrict_match and abs(l1 - l2) > 2:
                if verbose:
                    print("Skipping", row[0], "col idx:", col, l1, l2)
                continue
            overall1.append(l1)
            overall2.append(l2)
            keys.append((int(col), row[0]))
            col1.append(l1)
            col2.append(l2)
        if verbose:
            print(col, spearmanr(col1, col2))
    if verbose:
        print("num left", len(overall1))
        print("Overall", spearmanr(overall1, overall2))


    return [int(t) for t in anno1.columns[1:]], keys, overall1, overall2

def get_all_pairs():
    _, keys, _, _ = get_annotations(restrict_match=True)
    all_ents = list(set([key[1] for key in keys]))

    pairs = []
    for i in range(len(all_ents)):
        for j in range(i+1, len(all_ents)):
            pairs.append((all_ents[i], all_ents[j]))
    return pairs


def pairwise_compare(entity_to_our_score, my_article_counts, agree_only = False, pairs_to_keep = None, verbose = False):
    article_counts, keys, overall1, overall2 = get_annotations(restrict_match=True)

    keys_to_idx = {k:i for i,k in enumerate(keys)}

    avg = [(a + b)/2 for a,b in zip(overall1, overall2)]

    pairs = get_all_pairs()

    correct = 0
    total  = 0
    scored_pairs = set()
    for a in my_article_counts:
        for p in pairs:
            key1 = (a, p[0])
            key2 = (a, p[1])

            pair_key = (a, p)
            if pairs_to_keep is not None and not pair_key in pairs_to_keep:
                continue

            if key1 in keys_to_idx and key2 in keys_to_idx and key1 in entity_to_our_score and key2 in entity_to_our_score:
                if avg[keys_to_idx[key1]] == avg[keys_to_idx[key2]]:
                    continue
                gold = avg[keys_to_idx[key1]] < avg[keys_to_idx[key2]]

                if agree_only:
                    anno1 = overall1[keys_to_idx[key1]] < overall1[keys_to_idx[key2]]
                    anno2 = overall2[keys_to_idx[key1]] < overall2[keys_to_idx[key2]]
                    if anno1 != anno2:
                        continue

                ours = entity_to_our_score[key1] < entity_to_our_score[key2]
                total += 1
                scored_pairs.add(pair_key)
                if gold == ours:
                    correct += 1
    if verbose:
        # print(scored_pairs)
        print("Total", float(correct) / total, total)
    return scored_pairs

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

def score_ents_by_frequency(embeddings, entity_map):
    article_ids, article_id_to_buckets = get_articles(30)

    entity_to_count = Counter()
    for e,idxs in embeddings.entity_to_idx.items():
        if not e in entity_map:
            continue
        e = entity_map[e]
        for idx in idxs:
            article_id = os.path.basename(embeddings.tupls[idx].filename).split(".")[0]
            if not article_id in article_id_to_buckets:
                continue
            for b in article_id_to_buckets[article_id]:
                key = (b, e)
                entity_to_count[key] += 1

    return entity_to_count

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


    _, keys, _, _ = get_annotations(restrict_match=True)
    names = set([key[1] for key in keys])

    AZIZ_MAP = {}
    for line in open(cfg.AZIZ_ENTITIES).readlines():
        parts = line.split(",")
        for p in parts:
            AZIZ_MAP[p.strip()] = parts[0].strip()
    for n in names:
        AZIZ_MAP[n] = n

    if args.score_cache and os.path.exists(args.score_cache):
        entity_to_our_score = pickle.load(open(args.score_cache, "rb" ))
    else:
        entity_to_our_score = {}
        for c in ARTICLE_COUNTS:
            article_ids, qq = get_articles(c)
            print(c, article_ids, qq)
            entity_to_score, entity_to_count = entity_power_agency(embeddings, avg_embeddings, power=True, filter_set=None, article_filter_set=article_ids, entity_map=AZIZ_MAP)
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
