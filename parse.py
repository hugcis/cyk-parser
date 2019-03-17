import re
import sys
import tqdm
import fileinput
import pickle as pkl 
import numpy as np
from multiprocessing import Pool, Lock
from itertools import product
from copy import deepcopy

from sklearn.metrics.pairwise import cosine_similarity 

from levenshtein import levenshtein_distance
from grammar_tree import Tree, NotInGrammarError
from learn_pcfg import leaves, ignores_tags

def reverse_nested_dict(dic):
    dict_rev = {} 
    for i in dic: 
        for j in dic[i]: 
            if j not in dict_rev: 
                dict_rev[j] = {} 
            dict_rev[j][i] = dic[i][j] 
    return dict_rev

grammar = pkl.load(open('grammar.pkl', 'rb')) 
grammar_rev = reverse_nested_dict(grammar)
 
with open('polyglot-fr.pkl', 'rb') as f: 
    u = pkl._Unpickler(f) 
    u.encoding = 'latin1' 
    all_words, embeddings = u.load()

lexicon = pkl.load(open('lexicon.pkl', 'rb'))
lexicon_rev = reverse_nested_dict(lexicon)

m = set(sum(map(lambda x: list(x.keys()), lexicon_rev.values()), []))
UNK_DICT = dict(zip(m, len(m)*[1/len(m)]))  

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

def replace_oov(tokens):
    replaced = tokens[:]
    for n, word in enumerate(tokens):
        replaced[n] = lexicon_rev.get(word)
        found = replaced[n] is not None
        if not found:
            if word in all_words:
                embed_idx = all_words.index(word)
                for index in np.argsort(
                    -cosine_similarity(
                        embeddings[embed_idx].reshape(1, -1), 
                        embeddings)
                ).reshape(-1)[1:]:
                    if all_words[index] in lexicon_rev:
                        replaced[n] = lexicon_rev[all_words[index]]
                        break
            else:
                for candidate in lexicon_rev:
                    if levenshtein_distance(candidate, word) < 2:
                        replaced[n] = lexicon_rev[candidate]
                        found = True
                        break

                if (not found and 
                    '_' in word and 
                    word.split('_')[0] in lexicon_rev):
                    replaced[n] = lexicon_rev[ word.split('_')[0]]
                elif not found:
                    replaced[n] = UNK_DICT

    return replaced

def file_len(fname):
    with open(fname) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def process_line(sent, de_binarize=True):
    tokens = sent.strip().split(' ')
    rep_tokens = replace_oov(tokens[:])
    try:
        table, back = prob_cyk(rep_tokens, grammar)
        tree = Tree.build_from_stem(back, tokens[:])
        if de_binarize: tree.de_binarize()
        return '( ' + str(tree) + ')\n'

    except NotInGrammarError:
        err_mess = "\nSentence could not be produced with grammar: {}"
        print(err_mess.format(sent))
        return None

    except:
        print("\nError with sentence: {}".format(sent))
        pkl.dump((tokens, table, back), 
                    open('tblback.pkl', 'wb'))
        return None


def process_file(out_fname):
    f_out = open(out_fname, 'w')
    data_in = fileinput.input()
    
    with Pool() as pool:
        for i, item in enumerate(pool.map(process_line, data_in)):
            f_out.write(item if item is not None else '\n')

    sys.stdout.write('\nFinished\n\r')
    sys.stdout.flush() 

if __name__ == "__main__":
    process_file(sys.argv.pop(1))