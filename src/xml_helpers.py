import xml.etree.ElementTree as ET
import spacy
en_nlp = spacy.load('en')
import gzip

def process_xml_text(filename, stem=True, correct_idx=True, lower = False):
    if filename.endswith(".gz"):
        fp = gzip.open(filename)
        tree = ET.parse(fp)
        fp.close()
    else:
        tree = ET.parse(filename)
    root = tree.getroot()
    # indexing starts at 1
    if correct_idx:
        full_doc = [["DUMMY"]]
    else:
        full_doc = []
    for s in root.find('document').find('sentences').iter('sentence'):
        if correct_idx:
            sent = ["DUMMY"]
        else:
            sent = []

        for tok in s.find('tokens').iter('token'):
            if stem:
                if lower:
                    sent.append(tok.find('lemma').text.lower())
                else:
                    sent.append(tok.find('lemma').text)
            else:
                if lower:
                    sent.append(tok.find('word').text.lower())
                else:
                    sent.append(tok.find('word').text)
        full_doc.append(sent)

    return root, full_doc

def load_scores(filename):
    word_to_score = {}
    fp = open(filename)
    for line in fp.readlines():
        split = line.split(",")
        verb = split[0]
        word_to_score[en_nlp(verb)[0].lemma_] = float(split[1])
    return word_to_score
