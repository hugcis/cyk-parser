""" Script for tokenizing the content of a file passed as argument.
Writes the tokenized sentences to the second argument of the script. 
"""

import re
import sys
import fileinput
from learn_pcfg import ignores_tags, leaves

def tokenize_eval(sent):
    sent = ignores_tags.sub(r'(\g<1>', 
                            sent.strip('\n')[2:-1])
    tokens = []
    for match in leaves.finditer(sent):
        word = match.string[
            match.start(0):match.end(0)
        ].strip('()').split(' ')[1]
        tokens.append(word)
    return tokens

if __name__ == "__main__":
    out_file = open(sys.argv[2], 'w')
    for line in fileinput.input(files=(sys.argv[1])):
        tokens = tokenize_eval(line)
        out_file.write(' '.join(tokens) + '\n')