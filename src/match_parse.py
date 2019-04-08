import h5py
import xml.etree.ElementTree as ET
import gzip
import ast
import sys
from xml_helpers import process_xml_text
from collections import defaultdict, namedtuple
from lexicons import load_connotation_frames, load_power_verbs
import config
import numpy as np
from representations import TokenEmbedding
import pickle
import multiprocessing
import os
import glob
import argparse

NUM_PROCESSES=20

# sample_files 23529824.xml.hdf5 23525900.xml.hdf5  235299.xml.hdf5 2353104.xml.hdf5 235331.xml.hdf5

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

VerbInstance = namedtuple('VerbInstance', ['sent_id', 'verb_id', 'verb', 'verb_lemma', 'relation', 'entity_raw', 'entity_name', 'filename'], verbose=False)

# Not explicitly called for generating the cache, but can be used to sanity check file alignment
def check_match(h5py_file, xml_file):
    xml_file = xml_file
    fp = gzip.open(xml_file)
    tree = ET.parse(fp)
    root = tree.getroot()

    sents = []
    for s in root.find('document').find('sentences').iter('sentence'):
        sent = []
        for tok in s.find('tokens').iter('token'):
            sent.append(tok.find('word').text.lower())
        sents.append(" ".join(sent))

    f = h5py_file
    h5py_file = h5py.File(f, 'r')
    sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])
    for sent,sent_idx in sent_to_idx.items():
        assert sent == sents[int(sent_idx)]

def extract_entities(filename):
    root, full_doc = process_xml_text(filename)

    name_to_verbs = defaultdict(list)
    for coref in root.find('document').find('coreference').iter('coreference'):
        verbs_to_cache = []
        name = "Unknown"
        for mention in coref.findall('mention'):
            if 'representative' in mention.attrib:
                name = mention.find('text').text

            sent_id = int(mention.find('sentence').text) - 1

            sentence = root.find('document').find('sentences')[sent_id]
            for dep in sentence.find('dependencies').iter('dep'):
                if int(dep.find('dependent').get("idx")) != int(mention.find('end').text) - 1:
                    continue

                parent_id = int(dep.find('governor').get("idx")) - 1
                parent = dep.find('governor').text

                parent_lemma = sentence.find('tokens')[int(parent_id)].find('lemma').text

                # We save the sentence id, the parent id, the entity name, the relationship, the article number
                # With sentence id and parent id we can find embedding
                if dep.get("type") in ["nsubj", "nsubjpass", "dobj"]:
                    verbs_to_cache.append(VerbInstance(sent_id, parent_id, parent, parent_lemma, dep.get("type"),  mention.find('text').text, "", filename))

        # end coreff chain
        # We do it this way so that if we set the name in the middle of the chain we keep it for all things in the chain
        if verbs_to_cache:
            name_to_verbs[name] += verbs_to_cache

    final_verb_dict = {}
    for name,tupls in name_to_verbs.items():
        for t in tupls:
            key = (t.sent_id, t.verb_id)
            final_verb_dict[key] = t._replace(entity_name=name)

    id_to_sent={}
    # Also keep all verbs that are in lex
    for s in root.find('document').find('sentences').iter('sentence'):
        sent = []
        for tok in s.find('tokens').iter('token'):
            sent.append(tok.find("word").text.lower())
            sent_id = int(s.get("id")) - 1
            verb_id = int(tok.get("id")) - 1
            key = (sent_id, verb_id)
            if key in final_verb_dict:
                continue

            if tok.find('POS').text.startswith("VB"):
                final_verb_dict[key] = VerbInstance(sent_id, verb_id, tok.find("word").text, tok.find('lemma').text.lower(), "", "", "", filename)
        id_to_sent[sent_id] = " ".join(sent)

    return final_verb_dict, id_to_sent

def get_embeddings(f, verb_dict, nlp_id_to_sent, weights=[0,1,0]):
    tupl_to_embeds = {}
    idx_to_sent = {}

    try:
        h5py_file = h5py.File(f, 'r')
        sent_to_idx = ast.literal_eval(h5py_file.get("sentence_to_index")[0])

        assert(len(h5py_file) - 1 == len(nlp_id_to_sent)), str(len(h5py_file) - 1)

        for s in sent_to_idx:
            idx = int(sent_to_idx[s])
            idx_to_sent[idx] = s.split()

        for _,tupl in verb_dict.items():
            # assert what we can, some sentences are missing cause the keys in sentence_to_index are not unique
            # we're just going to ignore the missing ones for now and hope they don't matter to much
            # We care more about ones with entities. If we're just doing this to get verb scores it's
            # not a big deal if we skip a bunch
            if not tupl.sent_id in idx_to_sent:
                sent = nlp_id_to_sent[tupl.sent_id]
                idx = int(sent_to_idx[sent])
                tupl = tupl._replace(sent_id=idx)
            else:
                idx = tupl.sent_id

            if tupl.verb.lower() != idx_to_sent[idx][tupl.verb_id]:
                print("Mismatch", tupl.verb, str(idx_to_sent[idx][tupl.verb_id]), tupl.entity_name, f)
                continue

            s1 = h5py_file.get(str(idx))
            tupl_to_embeds[tupl] = (s1[0][tupl.verb_id] * weights[0] +
                                     s1[1][tupl.verb_id] * weights[1] +
                                     s1[2][tupl.verb_id] * weights[2])
    except UnicodeEncodeError:
        print("Unicode error, probably on mismatch")
    except OSError:
        print("OSError", f)
    except KeyError:
        print("KeyError", f)

    return tupl_to_embeds, idx_to_sent

def process_file(filename):
    nlp_file = os.path.join(NLP_PATH, filename)
    verb_dict, id_to_sent = extract_entities(nlp_file)
    h5_file = os.path.join(EMBED_PATH, filename) + ".hdf5"

    # h5_file format is "9978.txt.xml.hdf5"
    tupl_to_embeds, idx_to_sent = get_embeddings(h5_file, verb_dict, id_to_sent)
    return tupl_to_embeds, idx_to_sent, filename


def make_embeddings(nlp_path, embed_path, cache_name):
    global NLP_PATH
    NLP_PATH = nlp_path
    global EMBED_PATH
    EMBED_PATH = embed_path
    # filename format is "9978.txt.xml"


    filenames = [os.path.basename(f) for f in glob.iglob(os.path.join(nlp_path, "*.xml"))]
    filenames = [f for f in filenames if os.path.exists(os.path.join(embed_path, f) + ".hdf5")]

    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    out_data = pool.map(process_file, filenames)

    # Done all files
    print("Done file processing")
    vocab = []
    vectors = []
    tupls = []
    word_to_idxs = defaultdict(list)
    entity_to_idxs = defaultdict(list)
    file_to_sents = {}
    i = 0

    for tupl_to_embeds, idx_to_sent, filename in out_data:
        file_to_sents[filename] = idx_to_sent
        for tupl, vec in tupl_to_embeds.items():
            vocab.append(tupl.verb_lemma)
            tupls.append(tupl)
            vectors.append(vec)
            word_to_idxs[tupl.verb_lemma].append(i)
            entity_to_idxs[tupl.entity_name].append(i)
            i += 1

    vectors = np.vstack(vectors)
    assert vectors.shape[1] == 1024, "Weird vector length: " + str(vectors.shape)
    embed = TokenEmbedding(vectors, vocab, word_to_idxs, entity_dict=entity_to_idxs, file_to_sents=file_to_sents, normalize=True, tupls=tupls)
    if cache_name is not None:
        pickle.dump(embed, open(cache_name, "wb"), protocol=4)
    return embed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache")
    parser.add_argument("--nlp_path")
    parser.add_argument("--embed_path")
    args = parser.parse_args()

    make_embeddings(args.nlp_path, args.embed_path, args.cache)

if __name__ == "__main__":
    main()
