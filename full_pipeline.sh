# This file gives an overview of the steps for processing files, extracting embeddings, and running evaluations
# Running this script end-to-end has NOT been tested

# These directories need to be pre-created as specified
RAW_ARTICLES_DIR="/usr1/home/anjalief/ethics_metoo/nlp_input_pull2/" # Each article should be in a separate file, where the filename is article_id.txt
STANFORD_DIR="/usr1/home/anjalief/stanford-corenlp-full-2018-10-05" # Downloaded Stanford parser


# These are files are created by this script or directories that will be populated (directories need to exist)
NLP_OUTPUT_DIR="/usr1/home/anjalief/nlp_output_pull2" # Directory to store output of stanford parser
ELMO_INPUT_DIR="/projects/tir3/users/anjalief/elmo_embeddings/raw_tokenized/metoo_pull2" # Directory to store input to ELMo
ELMO_OUTPUT_DIR="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/metoo_pull2" # This should be the same as ELMO_INPUT_DIR but with "raw_tokenized" replaced with "embeddings"
MATCHED_EMBEDDING_CACHE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/metoo010_matched_tupl.pickle"
EVAL_SCORE_CACHE="aziz_entities_limit.pickle"


# Run stanford NLP pipleine over all texts
find $RAW_ARTICLES_DIR -name "*txt" > filelist.txt
java -cp "*" -Xmx50g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref,depparse -filelist filelist.txt -outputDirectory $NLP_OUTPUT_DIR

# Use output of parser to build tokenized files
NLP_OUTPUT_DIR="/projects/tir3/users/anjalief/corpora/nlp_output_pull2/" # I copied these over to tir
python prep_elmo.py --input_glob "$NLP_OUTPUT_DIR/*.xml" --output_dir $ELMO_INPUT_DIR

# Extract elmo embeddings over all files
./make_run_scripts.sh "$ELMO_INPUT_DIR/*.elmo"  # NOTE: may need to change locations / job numbers in make_run_scripts.sh
# Then need to run the generated run scripts: e.g. sbatch run_elmo_night.sh

# Cache all verbs and entities from elmo embeddings
python match_parse.py --cache $MATCHED_EMBEDDING_CACHE --nlp_path $NLP_OUTPUT_DIR --embed_path $ELMO_OUTPUT_DIR

# Run evaluation over verbs
python weighted_tests.py --cache $MATCHED_EMBEDDING_CACHE --from_scratch

# Run evalulations against power scripts
python metoo_eval.py --embedding_cache $MATCHED_EMBEDDING_CACHE --score_cache $EVAL_SCORE_CACHE

# Run analyses in paper
python metoo_analysis.py --embedding_cache $MATCHED_EMBEDDING_CACHE
