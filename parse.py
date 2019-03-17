import re
import sys
import tqdm
import fileinput
import pickle as pkl
import numpy as np
from multiprocessing import Pool, Lock
from itertools import product
from copy import deepcopy

from grammar_tree import Tree, NotInGrammarError
from learn_pcfg import leaves, ignores_tags
from oov import replace_oov, reverse_nested_dict

grammar = pkl.load(open('grammar.pkl', 'rb'))
grammar_rev = reverse_nested_dict(grammar)


def update_tbl_bck(i, k, n, table, back, grammar_rev):
    """ Update table and back in the CYK algortihm.
    """
    for (right_0, right_1) in product(table[i][k], table[k][n]):
        for left in grammar_rev.get((right_0, right_1), []):
            prob = grammar_rev[(right_0, right_1)][left]
            test = (np.log(prob) +
                    table[i][k][right_0] +
                    table[k][n][right_1])

            if table[i][n].get(left, -np.inf) < test:
                table[i][n][left] = test
                back[i][n][left] = (k, right_0, right_1)


def prob_cyk(tokens, grammar):
    """ Probabilistic CYK implementation inspired from Jurafsky's
    pseudo-code implementation.
    """
    table = []
    for s in range(len(tokens) + 1):
        table.append([])
        for _ in range(len(tokens) + 1):
            table[s].append({})
    back = deepcopy(table)

    for n in range(1, len(tokens) + 1):
        pos_tags = tokens[n - 1]
        for tag in pos_tags:
            for rule in grammar_rev.get((tag,), []):
                table[n - 1][n][rule] = np.log(
                    grammar_rev[(tag,)][rule])

            table[n - 1][n][tag] = np.log(pos_tags[tag])

        for i in range(n-2, -1, -1):
            for k in range(i+1, n):
                update_tbl_bck(i, k, n,
                               table,
                               back,
                               grammar_rev)

    one_word = (len(tokens) == 1)
    if one_word:
        back[0][1]['SENT'] = (1, 'NP',
                              min(table[0][1].items(),
                                  key=lambda x: x[1])[0])

    return table, back


def file_len(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def process_line(sent, de_binarize=True):
    tokens = sent.strip().split(' ')
    rep_tokens = replace_oov(tokens[:])
    try:
        _, back = prob_cyk(rep_tokens, grammar)
        tree = Tree.build_from_stem(back, tokens[:])
        if de_binarize:
            tree.de_binarize()
        return '( ' + str(tree) + ')\n'

    except NotInGrammarError:
        err_mess = "\nSentence could not be produced with grammar: {}"
        print(err_mess.format(sent))
        return None

    except Exception as err:
        print("\nError with sentence: {}".format(sent))
        print(err, sys.exc_info()[0])
        sys.stdout.flush()
        return None


def process_file(out_fname, n_proc):
    f_out = open(out_fname, 'w')
    data_in = fileinput.input()
    if not n_proc:
        n_proc = None

    with Pool(n_proc) as pool:
        for _, item in enumerate(pool.map(process_line, data_in)):
            f_out.write(item if item is not None else '\n')

    sys.stdout.write('\nFinished\n\r')
    sys.stdout.flush()

if __name__ == "__main__":
    process_file(sys.argv[1], sys.argv[2])
