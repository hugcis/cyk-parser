#!/bin/bash

usage()
{
    echo "Parse dependencies of sentences from stdin

First argument if given is the input filename containing the tokenized sentences to be parsed.
Optional arguments:
    -tf | --train-file filename    The file containing the sentences to learn a new PCFG on.
    -o  | --output-file            Output file where the sentences are generated 
    -h  | --help                   Show this help message
    -n  | --n-proc                 Number of parallel processes to use. Default is
                                   given by Python's os.cpu_count()

Example usage:
    ./run.sh input_tokenized -tf train_sentences.mrg_strict
it is equivalent to 
    ./run.sh -tf train_sentences.mrg_strict -o evaluation_data.parser_output < input_tokenized"
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
n_processes=0
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
        -n | --n-proc )         shift
                                n_processes=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


python parse.py $output_file $n_processes < $input_file