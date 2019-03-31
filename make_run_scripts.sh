# Prepare scripts for running in parallel
# Argument to the script is the directory containing files to process
# The input file path should contain a folder called "raw_tokenized"
# Output files will be in the folder "embeddings" instead of "raw_tokenized"
# The scripts to run will be placed in a directory called "run_scripts"

run_scripts=()
IN_DIR=$0

jobs_per_file=50
count=0
file_idx=0

run_script="run_scripts/${file_idx}.run.sh"
echo "#!/bin/bash" > $run_script
chmod +x $run_script

for f in ${IN_DIR} ; do

  # start a new script
  if [[ "$count" -gt "$jobs_per_file" ]]; then
      count=0
      file_idx=$((file_idx+1))
      run_script="run_scripts/${file_idx}.run.sh"
      echo "#!/bin/bash" > $run_script
      chmod +x $run_script
      run_scripts+=("$run_script")
  fi

  tmp_new=${f/.xml/}
  new_name=${tmp_new/.elmo/.hdf5}
  full_new_name=${new_name/raw_tokenized/embeddings}

  echo "if [ ! -f $full_new_name ]; then" >> $run_script
  echo "    source activate py36 && allennlp elmo $f $full_new_name --all" >> $run_script
  echo "fi" >> $run_script
  echo "" >> $run_script

  count=$((count+1))

done
echo Num items: ${#run_scripts[@]}
echo Data: ${run_scripts[@]}
