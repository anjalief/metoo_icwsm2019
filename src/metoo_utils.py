import config as cfg
import pandas
from collections import Counter
import os
from scipy.stats import pearsonr, spearmanr

def get_all_pairs():
    _, keys, _, _ = get_annotations(restrict_match=True)
    all_ents = list(set([key[1] for key in keys]))

    pairs = []
    for i in range(len(all_ents)):
        for j in range(i+1, len(all_ents)):
            if all_ents[i] < all_ents[j]:
                pairs.append((all_ents[i], all_ents[j]))
            else:
                pairs.append((all_ents[j], all_ents[i]))
    return pairs


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


def get_aziz_map():
    _, keys, _, _ = get_annotations(restrict_match=True)
    names = set([key[1] for key in keys])

    AZIZ_MAP = {}
    for line in open(cfg.AZIZ_ENTITIES).readlines():
        parts = line.split(",")
        for p in parts:
            AZIZ_MAP[p.strip()] = parts[0].strip()
    for n in names:
        AZIZ_MAP[n] = n
    return AZIZ_MAP


def pairwise_compare(entity_to_our_score, my_article_counts, agree_only = False, pairs_to_keep = None, verbose = False, lower = False):
    article_counts, keys, overall1, overall2 = get_annotations(restrict_match=True)
    pairs = get_all_pairs()

    if lower:
        keys_to_idx = {(k[0], k[1].lower()):i for i,k in enumerate(keys)}
        pairs = [(p[0].lower(), p[1].lower()) for p in pairs]
    else:
        keys_to_idx = {k:i for i,k in enumerate(keys)}

    avg = [(a + b)/2 for a,b in zip(overall1, overall2)]

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
        print("Total", float(correct) / total, total)
    return scored_pairs

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
