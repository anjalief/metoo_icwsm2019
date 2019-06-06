import pickle
from representations import TokenEmbedding, get_token_embeddings
from weighted_tests import build_feature_array, logistic_regression, pred_to_score
from lexicons import load_hannah_split, load_power_split, load_agency_split, load_connotation_frames, load_power_all, load_agency_all
import config as cfg
import weights as wgts
from collections import defaultdict, Counter
import random
import operator
import pandas
import os
import sys
from graph_builder import EntityScoreTracker, Edge, EdgeTracker
import igraph
import argparse
from scipy.stats import ttest_ind


from match_parse import VerbInstance

from datetime import datetime

PRONOUNS=["i", "me", "my", "mine", "myself",
          "you", "your", "yours", "yourself", "yourselves",
          "he", "him", "his", "himself",
          "she", "her", "hers", "herself",
          "it", "its", "itself",
          "we", "us", "our", "ours", "ourselves",
          "they", "them", "their", "theirs", "themselves"]

def get_entity_scores(train, dev=None, test=None, weights=None, embeddings=None, avg_embeddings=None):
    # First do the normal thing to make sure our cache is reasonable
    train_X, train_Y, _ = build_feature_array(train, avg_embeddings)
    if test is not None:
        test_X, test_Y, test_words_Y, sents = build_feature_array(test, embeddings, get_sents=True)
        logistic_regression(train_X, train_Y, test_X, test_Y, weights, do_print=True, return_preds=True, is_token=True, test_words_Y=test_words_Y)

    # We set is_token to false so that we skip the aggregation -- we don't need it
    preds = logistic_regression(train_X, train_Y, embeddings.m, embeddings.iw, weights, do_print=False, return_preds=True, is_token=False)

    # Out default settings +1 to scores since we use them as idxs. -1 (though it doesn't really matter, since we're always comparing, not
    # really caring about absolute scores)
    return [p - 1 for p in preds]

# Used to score entites for sentiment (can also be used for effect)
def most_positive_entities(embeddings, avg_embeddings, subject_header, object_header, by_outlet=False, filter_set=None, article_filter=None, entity_map=None):
    lex = load_connotation_frames(cfg.CONNO_FRAME, object_header, binarize = True, remove_neutral=False)
    wo_preds = get_entity_scores(lex, None, None, wgts.metoo_header_to_avg_token_weights[object_header], embeddings, avg_embeddings)

    lex = load_connotation_frames(cfg.CONNO_FRAME, subject_header, binarize = True, remove_neutral=False)
    ws_preds = get_entity_scores(lex, None, None,  wgts.metoo_header_to_avg_token_weights[subject_header], embeddings, avg_embeddings)

    entity_to_score = defaultdict(float)
    entity_to_count = Counter()
    entity_to_values = defaultdict(list)

    if by_outlet:
        id_to_outlet = load_outlets()

    # Which entities are portrayed as most postitive?
    for e,idxs in embeddings.entity_to_idx.items():
        if e.lower() in PRONOUNS:
            continue

        if entity_map is not None:
            if not e in entity_map:
                continue
            e = entity_map[e]

        if filter_set is not None and not e in filter_set:
            continue

        # this is a verb that has some association with e
        for idx in idxs:
            article_id_str = os.path.basename(embeddings.tupls[idx].filename).split(".")[0]
            if article_filter is not None and not article_id_str in article_filter:
                continue

            if by_outlet:
                article_id = int(article_id_str)
                outlet = id_to_outlet[article_id]
                key = (e, outlet)
            else:
                key = e

            rel = embeddings.tupls[idx].relation

            # if this is the subject we take the ws score
            if rel == "nsubj":
                entity_to_score[key] += ws_preds[idx]
                entity_to_count[key] += 1
                entity_to_values[key].append(ws_preds[idx])

            # if this is the object or nsubjpass we take the wo score
            elif rel == "dobj"  or rel == "nsubjpass":
                entity_to_score[key] += wo_preds[idx]
                entity_to_count[key] += 1
                entity_to_values[key].append(wo_preds[idx])

    # Normalize by counts
    for e in entity_to_score:
        entity_to_score[e] /= entity_to_count[e]

    # Only return outlets. (and only return outlets that mention the entity at least 10 times)
    if by_outlet:
        outlet_to_score = {}
        outlet_to_values = defaultdict(list)
        for e, score in sorted(entity_to_score.items(), key=operator.itemgetter(0), reverse=True):
            if entity_to_count[e] < 10:
                continue
            outlet_to_score[e[1]] = score
            outlet_to_values[e[1]] += entity_to_values[e]
        return outlet_to_score, outlet_to_values


    return entity_to_score, entity_to_count, entity_to_values

def entity_power_agency(embeddings, avg_embeddings, power=True, filter_set=None, article_filter_set=None, entity_map=None):
    if power:
        lex = load_power_all(cfg.POWER_AGENCY)
        preds = get_entity_scores(lex, None, None, wgts.power_token_regression, embeddings, avg_embeddings)
    else:
        lex = load_agency_all(cfg.POWER_AGENCY)
        preds = get_entity_scores(lex, None, None, wgts.agency_token_regression, embeddings, avg_embeddings)

    entity_to_score = defaultdict(float)
    entity_to_count = Counter()
    entity_to_values = defaultdict(list)

    for e,idxs in embeddings.entity_to_idx.items():
        if e.lower() in PRONOUNS:
            continue

        if filter_set is not None and not e in filter_set:
            continue

        if entity_map is not None:
            if not e in entity_map:
                continue
            e = entity_map[e]

        # this is a verb that has some association with e
        for idx in idxs:
            rel = embeddings.tupls[idx].relation

            article_id = os.path.basename(embeddings.tupls[idx].filename).split(".")[0]
            if article_filter_set is not None and not article_id in article_filter_set:
                continue

            # if this is the subject we take the raw score
            if rel == "nsubj":
                entity_to_score[e] += preds[idx]
                entity_to_count[e] += 1
                entity_to_values[e].append(preds[idx])

            # agency only applies to subject
            if power:
                # if this is the object or nsubjpass we take -score
                if rel == "dobj"  or rel == "nsubjpass":
                    entity_to_score[e] += -1 * preds[idx]
                    entity_to_count[e] += 1
                    entity_to_values[e].append(-1 * preds[idx])

            tupl = embeddings.tupls[idx]
            sent = embeddings.file_to_sents[os.path.basename(tupl.filename)][tupl.sent_id]

    # Normalise by counts
    for e in entity_to_score:
        entity_to_score[e] /= entity_to_count[e]
    return entity_to_score, entity_to_count, entity_to_values


def print_top100(entity_to_count, entity_to_score, print_header=""):
    print ("#########################################################################################################################################")
    print ("Ordering of most frequent entities", print_header)
    print ("#########################################################################################################################################")

    top_100 = [e[0] for e in sorted(entity_to_count.items(), key=operator.itemgetter(1), reverse=True)[:min(100, len(entity_to_count))]]
    for e, score in sorted(entity_to_score.items(), key=operator.itemgetter(1), reverse=True):
        if e in top_100:
            print(e, score, entity_to_count[e])

def load_outlets():
    df = pandas.read_csv(cfg.METOO_META, sep="\t", header=None)
    article_id_to_outlet = {}
    for i,row in df.iterrows():
        article_id_to_outlet[i] = row.iloc[2]
    return article_id_to_outlet

def load_dates():
    df = pandas.read_csv(cfg.METOO_META, sep="\t", header=None)
    article_id_to_outlet = {}
    for i,row in df.iterrows():
        d = row.iloc[3].split("T")[0]
        article_id_to_outlet[i] = datetime.strptime(d, '%Y-%m-%d')
    return article_id_to_outlet

def build_power_graph(embeddings, avg_embeddings, power=True, filter_articles=None, entity_map=None, graph_name_str="aziz_power_graph", min_count=8, vertex_scalar=1.5):
    if power:
        train = load_power_all(cfg.POWER_AGENCY)
        preds = get_entity_scores(train, None, None, wgts.power_token_regression, embeddings, avg_embeddings)
    else:
        train = load_agency_all(cfg.POWER_AGENCY)
        preds = get_entity_scores(train, None, None, wgts.agency_token_regression, embeddings, avg_embeddings)

    # we ultimately need to control for number of co-occurences in the same article
    # it's easiest to do this 1 article at a time

    # this is a map from article id to all entites in that article
    article_to_entity_idx = defaultdict(list)
    entity_to_article_count = defaultdict(list)
    edge_tracker = EdgeTracker()

    for e,idxs in embeddings.entity_to_idx.items():
        # this is a verb that has some association with e
        for idx in idxs:
            article_id = os.path.basename(embeddings.tupls[idx].filename).split(".")[0]
            if filter_articles is not None and not str(article_id) in filter_articles:
                continue
            article_to_entity_idx[article_id].append((e, idx))

    for a,val in article_to_entity_idx.items():
        per_article_entity_tracker = defaultdict(EntityScoreTracker)
        for e,idx in val:
            if e.lower() in PRONOUNS:
                continue

            if entity_map is not None:
                if not e in entity_map:
                    continue
                e = entity_map[e]

            tupl = embeddings.tupls[idx]

            if tupl.relation == "nsubj":
                per_article_entity_tracker[e].update(preds[idx], 1, False)
            elif tupl.relation in ["nsubjpass", "dobj"]:
                per_article_entity_tracker[e].update(-1 * preds[idx], 1, False)

        keys = list(per_article_entity_tracker.keys())

        # Mark that entities co-occurred in this article
        for i in range(0, len(keys)):
            e1 = keys[i]
            entity_to_article_count[e1].append(a)

            for j in range(i + 1, len(keys)):
                e2 = keys[j]

                edge_tracker.update(e1, e2, per_article_entity_tracker[e1].score, per_article_entity_tracker[e2].score, False)
    # end all articles

    # remove all vertices that didn't appear in at least 8 articles
    to_delete = []
    for v in entity_to_article_count:
        if len(entity_to_article_count[v]) < min_count:
            to_delete.append(v)

    edges, edge_weights, vertex_names, vertex_weights, missing = edge_tracker.get_edge_list(to_delete)

    # make vertex weights all positive
    m = min(vertex_weights)
    if m < 0:
        # scale up for better visualization
        vertex_weights = [(v + abs(m)) * vertex_scalar for v in vertex_weights]

    g = igraph.Graph(edges=edges, vertex_attrs={"name":vertex_names, "v_weights":vertex_weights}, edge_attrs={"weights" : edge_weights}, directed=True)

    # edge_colors = ["red" if a else "blue"  for a in missing]
    visual_style = {}
    visual_style["vertex_label"] = g.vs["name"]
    visual_style["vertex_label_size"] = 16
    visual_style["edge_width"] = g.es["weights"]
    visual_style["edge_color"] = 'gray'
    visual_style["vertex_shape"] = 'circular'
    visual_style["vertex_frame_color"] = 'gray'
    visual_style["vertex_size"] = vertex_weights
    visual_style["margin"] = 55
    # visual_style["edge_color"] = edge_colors
    ts = datetime.now().timestamp()
    graph_name = graph_name_str + str(ts) + ".png"
    print("Saving", graph_name)
    igraph.plot(g, graph_name, **visual_style)


def aziz_analysis(embeddings, avg_embeddings, do_print_statistics):
    article_list = open(cfg.AZIZ_ARTICLES).readlines()
    article_list = [x.split(".")[1].replace("/", "") for x in article_list]

    entity_map = {}

    for line in open(cfg.AZIZ_ENTITIES).readlines():
        parts = line.split(",")
        for p in parts:
            entity_map[p.strip()] = parts[0].strip()

    articles_before = []
    articles_after = []
    id_to_date = load_dates()
    date_of_babe = datetime.strptime("2018-01-13", '%Y-%m-%d')

    for i in article_list:
        d = id_to_date[int(i)]
        if d < date_of_babe:
            articles_before.append(i)
        elif d > date_of_babe:
            articles_after.append(i)
        elif d == date_of_babe:
            assert False

    # the date on this article is wrong
    articles_after.append("648")
    articles_before.remove("648")

    # Figure 5
    build_power_graph(embeddings, avg_embeddings, power=True, filter_articles=article_list, entity_map=entity_map)

    print("Num before", len(articles_before))
    print("Num after", len(articles_after))

    # Figures 6, 7, 8
    filter_set=['Aziz Ansari', 'Grace', 'Katie Way', 'Caitlin Flanagan', 'Ashleigh Banfield']
    ent_to_power, _, ent_to_power_values = entity_power_agency(embeddings, avg_embeddings, power=True, filter_set=filter_set, article_filter_set=articles_after, entity_map=entity_map)
    ent_to_agency, _, ent_to_agency_values = entity_power_agency(embeddings, avg_embeddings, power=False, filter_set=filter_set, article_filter_set=articles_after, entity_map=entity_map)
    ent_to_sent, _, ent_to_sent_values = most_positive_entities(embeddings, avg_embeddings, subject_header="Perspective(ws)", object_header="Perspective(wo)", by_outlet=False, filter_set=filter_set, article_filter=articles_after, entity_map=entity_map)
    print("######################################## Power, Agency, and Sentiment for Aziz Ansari Entities (Fig 6, 7, 8) ##############################################################")
    print("Name,Power,Agency,Sentiment")
    for ent in filter_set:
        print(ent, ent_to_power[ent], ent_to_agency[ent], ent_to_sent[ent])

    if do_print_statistics:
        print_statistics(ent_to_power_values, filter_set, "Power")
        print_statistics(ent_to_agency_values, filter_set, "Agency")
        print_statistics(ent_to_sent_values, filter_set, "Sentiment")

    # Figure 9
    print("#################################################### BEFORE Babe.net article (Fig 9) ########################################################")
    before_power, _, _ = entity_power_agency(embeddings, avg_embeddings, power=True, filter_set=["Aziz Ansari"], article_filter_set=articles_before, entity_map=entity_map)
    before_agency, _, _ = entity_power_agency(embeddings, avg_embeddings, power=False, filter_set=["Aziz Ansari"], article_filter_set=articles_before, entity_map=entity_map)
    before_sent, _, _ = most_positive_entities(embeddings, avg_embeddings, subject_header="Perspective(ws)", object_header="Perspective(wo)", filter_set=["Aziz Ansari"], by_outlet=False, article_filter=articles_before, entity_map=entity_map)
    print("Power", before_power["Aziz Ansari"])
    print("Agency", before_agency["Aziz Ansari"])
    print("Sentiment", before_sent["Aziz Ansari"])


    print("################################################### AFTER Babe.net article (Fig 9) ######################################################")
    print("Power", ent_to_power["Aziz Ansari"])
    print("Agency", ent_to_agency["Aziz Ansari"])
    print("Sentiment", ent_to_sent["Aziz Ansari"])

def print_statistics(entity_to_values, target_entities, print_header):
    print("###################################### Significance tests for", print_header, "###############################################")
    for i in range(0, len(target_entities)):
        for j in range(i + 1, len(target_entities)):
            t1 = target_entities[i]
            t2 = target_entities[j]
            print(t1, t2, ttest_ind(entity_to_values[t1], entity_to_values[t2], equal_var = False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_cache")
    parser.add_argument("--print_statistics", action='store_true')
    args = parser.parse_args()

    embeddings = get_token_embeddings("", [], [], cache_name=args.embedding_cache)
    avg_embeddings = embeddings.make_average()

    # Table 5
    entity_to_sent_score, entity_to_sent_count, entity_to_values = most_positive_entities(embeddings, avg_embeddings, subject_header="Perspective(ws)", object_header="Perspective(wo)")
    print_top100(entity_to_sent_count, entity_to_sent_score, print_header="Sentiment")

    # # Table 6
    entity_to_power_score, entity_to_power_count, entity_to_power_values = entity_power_agency(embeddings, avg_embeddings, power=True, filter_set=None, article_filter_set=None, entity_map=None)
    print_top100(entity_to_power_count, entity_to_power_score, print_header="Power")

    # # Table 7
    entity_to_agency_score, entity_to_agency_count, entity_to_agency_values = entity_power_agency(embeddings, avg_embeddings, power=False, filter_set=None, article_filter_set=None, entity_map=None)
    print_top100(entity_to_agency_count, entity_to_agency_score, print_header="Agency")

    print("######################################  Figures 2 and 3 #####################################################")
    target_entities = ["Donald Trump", "Hillary Clinton", "Al Franken", "Roy Moore", "Rose McGowan", "Leeann Tweeden", "Harvey Weinstein", "Bill Cosby"]
    for e in target_entities:
        print(e, entity_to_sent_score[e], entity_to_power_score[e])
    if args.print_statistics:
        print_statistics(entity_to_values, target_entities, "Sentiment")
        print_statistics(entity_to_power_values, target_entities, "Power")

    # Figure 4
    print("######################################  Outlet Franken Moore (Figure 4) #####################################################")
    franken_to_score, franken_to_values = most_positive_entities(embeddings, avg_embeddings, subject_header="Perspective(ws)", object_header="Perspective(wo)", by_outlet=True, filter_set=['Al Franken'])
    moore_to_score, moore_to_values  = most_positive_entities(embeddings, avg_embeddings, subject_header="Perspective(ws)", object_header="Perspective(wo)", by_outlet=True, filter_set=['Roy Moore'])

    print("Al Franken", "Roy Moore")
    outlet_key_to_values = {}
    for o in franken_to_score:
        if o in moore_to_score:
            print(o, franken_to_score[o], moore_to_score[o])
            outlet_key_to_values[("Franken", o)] = franken_to_values[o]
            outlet_key_to_values[("Moore", o)] = moore_to_values[o]
    if args.print_statistics:
        print_statistics(outlet_key_to_values, list(outlet_key_to_values.keys()), "Sentiment")

    aziz_analysis(embeddings, avg_embeddings, args.print_statistics)
    return



if __name__ == "__main__":
    main()
