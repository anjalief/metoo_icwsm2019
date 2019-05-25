# Do pre-processing according to description

# The ``elmo`` subcommand allows you to make bulk ELMo predictions.
# Given a pre-processed input text file, this command outputs the internal
# layers used to compute ELMo representations to a single (potentially large) file.
# The input file is previously tokenized, whitespace separated text, one sentence per line.
# The output is a hdf5 file (<http://docs.h5py.org/en/latest/>) where, with the --all flag, each
# sentence is a size (3, num_tokens, 1024) array with the biLM representations.
# For information, see "Deep contextualized word representations", Peters et al 2018.
# https://arxiv.org/abs/1802.05365

import sys
from xml_helpers import process_xml_text
import os
import argparse
import glob
import multiprocessing

NUM_PROCESSES=4
OUTPUT_DIR=""
LOWER=True

def process_article(filename):
    new_name = os.path.basename(filename).replace(".gz", "") + ".elmo"
    new_name = os.path.join(OUTPUT_DIR, new_name)

    _,doc = process_xml_text(filename, correct_idx=False, stem=False, lower = LOWER)
    outfile = open(new_name, "w")
    outfile.write("\n".join([" ".join(s) for s in doc]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob")
    parser.add_argument("--output_dir")
    parser.add_argument("--keep_case", action='store_true')
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    global LOWER
    LOWER = not args.keep_case # we lower case unless we explicitly say not to

    files = [f for f in glob.iglob(args.input_glob)]
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    pool.map(process_article, files)

if __name__ == "__main__":
    main()
