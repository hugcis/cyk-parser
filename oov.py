import numpy as np
import pickle as pkl
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from levenshtein import levenshtein_distance


def reverse_nested_dict(dic):
    dict_rev = {}
    for i in dic:
        for j in dic[i]:
            if j not in dict_rev:
                dict_rev[j] = {}
            dict_rev[j][i] = dic[i][j]
    return dict_rev


with open('polyglot-fr.pkl', 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    all_words, embeddings = u.load()

lexicon = pkl.load(open('lexicon.pkl', 'rb'))
lexicon_rev = reverse_nested_dict(lexicon)

m = set(sum(map(lambda x: list(x.keys()), lexicon_rev.values()), []))
UNK_DICT = dict(zip(m, len(m)*[1/len(m)]))


def find_in_embeddings(word):
    embed_idx = all_words.index(word)
    for index in np.argsort(
        -cosine_similarity(
            embeddings[embed_idx].reshape(1, -1),
            embeddings)
    ).reshape(-1)[1:]:
        if all_words[index] in lexicon_rev:
            return lexicon_rev[all_words[index]]


def replace_oov(tokens):
    replaced = tokens[:]
    for n, word in enumerate(tokens):
        replaced[n] = lexicon_rev.get(word)
        found = replaced[n] is not None
        if not found:
            if word in all_words:
                replaced[n] = find_in_embeddings(word)
            else:
                candidates = [
                    candidate for candidate in lexicon_rev
                    if levenshtein_distance(candidate, word, 2) < 3
                ]
                if len(candidates):
                    agg_count = sum((Counter(lexicon_rev[candidate])
                                     for candidate in candidates),
                                    Counter())
                    replace = dict(agg_count)

                    replaced[n] = dict(
                        zip(replace.keys(),
                            map(lambda x: x/sum(replace.values()),
                                replace.values()))
                    )
                    found = True

                is_grouped = (not found and '_' in word)
                if is_grouped and word.split('_')[0] in lexicon_rev:
                    replaced[n] = lexicon_rev[word.split('_')[0]]
                elif not found:
                    replaced[n] = UNK_DICT
    return replaced
