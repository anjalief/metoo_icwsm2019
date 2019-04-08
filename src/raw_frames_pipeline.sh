# This details the commands for running our system over the raw connotation frames
# This is a seperate file from full_pipeline because it uses a different data set
# NOTE: This script has NOT been tested end-to-end

# These files can be named anything - they will be created. The only thing that
# needs to be set is the path to the annotations in config.py
TOKEN_FILE="/projects/tir3/users/anjalief/elmo_embeddings/raw_tokenized/cf_annotations.elmo"
META_FILE="/projects/tir3/users/anjalief/elmo_embeddings/raw_tokenized/cf_annotations.meta"
EMBEDDINGS_FILE="/projects/tir3/users/anjalief/elmo_embeddings/embeddings/cf_annotations.hdf5"


# Prepare annotations for elmo extraction
python test_raw_frames.py --meta_file $META_FILE --token_file $TOKEN_FILE

# Make and cache embeddings
source activate py36 && allennlp elmo $TOKEN_FILE $EMBEDDINGS_FILE --all

# Run evaluations
python test_raw_frames.py --meta_file $META_FILE --token_file $TOKEN_FILE --embeddings_cache $EMBEDDINGS_FILE --from_scratch

# Learned weights can be saved in weights.py; then can run
python test_raw_frames.py --meta_file $META_FILE --token_file $TOKEN_FILE --embeddings_cache $EMBEDDINGS_FILE
