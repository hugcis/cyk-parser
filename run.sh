#!/bin/bash

usage()
{
    echo "Parse dependencies of sentences from stdin

First argument if given is the input filename containing the tokenized sentences to be parsed.
Optional arguments:
    -tf | --train-file filename    The file containing the sentences to learn a new PCFG on.
    -h | --help                    Show this help message
    
Example usage:
    ./run.sh input_tokenized -tf train_sentences.mrg_strict"
}

re_train()
{
    echo "Retraining with file $1"
    python learn_pcfg.py $1
}

# If first argument is given, use it as input file
# else use stdin
if [[ $1 != -* ]]; then
    input_file="${1:-/dev/stdin}"
    shift
else 
    input_file=/dev/stdin
fi

output_file=/dev/stdout

# Parse optional arguments
while [ "$1" != "" ]; do
    case $1 in
        -tf | --train-file )    shift
                                filename=$1
                                re_train $filename
                                ;;
        -o | --output-file )    shift
                                output_file=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


python parse.py $output_file < $input_file