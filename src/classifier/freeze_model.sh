#!/usr/bin/env bash

source activate trafficsigns

input_folder=`readlink -m $1`

if [[ -z "$input_folder" ]]
then
    echo "You have to give folder with saved model to export. Gave: $input_folder"
    exit 1
fi

checkpoint_info_file=${input_folder}/checkpoint
model_checkpoint_path=${input_folder}/$(head -n 1 ${checkpoint_info_file} | grep -o 'model.ckpt.*[^"]')

echo "Using ${model_checkpoint_path} as a file with weights"

dumped_graph_name=${input_folder}/temp_graph.pbtxt
frozen_graph_name=${input_folder}/temp_frozen_graph.pb
output_graph_name=${input_folder}/frozen_inference_graph.pb

echo "Dumping graph ..."
python -m classifier.convert_model -m ${input_folder} -o ${dumped_graph_name}

echo "Freezing graph ..."
python utils/freeze_graph.py \
    --input_graph ${dumped_graph_name} \
    --input_checkpoint ${model_checkpoint_path} \
    --output_graph ${frozen_graph_name} \
    --output_node_names output

echo "Optimising graph ..."
python utils/optimise_for_inference.py \
    --input ${frozen_graph_name} \
    --output ${output_graph_name} \
    --input_names="input_images" \
    --output_names="output"

rm ${dumped_graph_name} ${frozen_graph_name}