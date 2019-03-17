import re
import fileinput
import pickle as pkl
from copy import deepcopy
from collections import Counter

ignores_tags = re.compile(r'\((([A-Z.a-z]){1,10})-([A-Z:/#_]){1,20}')

all_chars = r"\w_'\".,+©<±\]\[é^&ùà$=èê°?;%:/!-"
l_pattern = r"\(([A-Za-z.+]{1,10}) ([" + all_chars + r"]+?)\)"
leaves = re.compile(l_pattern, re.UNICODE)

rule = r'( \([A-Za-z.+]{1,10}\))'
grammar_rule_pattern = r'\(([A-Za-z.+]{1,10})' + rule + r'{1,35}\)'
grammar_rule = re.compile(grammar_rule_pattern)


def get_unit(grammar):
    """ Utility function for getting the list of unit rules from a grammar.
    """
    return [(i, j) for i in grammar for j in grammar[i]
            if len(j) == 1 and j[0] in grammar and i != j[0]]


def eliminate_unit_rules(base_grammar):
    """ Eliminate unit rules from the grammar, ie rules of the form

    `A` -> `B`
    `B` -> `a`

    where `a` is a terminal symbol

    Returns: 
        the new grammar without unit rules
    """
    grammar = deepcopy(base_grammar)
    unaries = get_unit(grammar)
    while len(unaries):
        for (a, b) in unaries:
            map_mult = map(lambda x: x * grammar[a][b],
                           grammar[b[0]].values())
            multip_probs_dict = dict(zip(grammar[b[0]].keys(),
                                         map_mult))
            grammar[a] = dict(Counter(grammar[a]) +
                              Counter(multip_probs_dict))
            grammar[a].pop(b)

            # Special case that creates non coherent rules
            if a == 'NP' and b == ('PP',):
                grammar[a].pop(('NP',))

        unaries = get_unit(grammar)
    return grammar


def normalize_lexicon_probs(lexicon):
    """ Normalize lexicon counts to get probabilities.
    """
    count = sum(lexicon.values())
    normed_vals = map(lambda x: x/count, lexicon.values())
    lexicon = dict(zip(lexicon.keys(), normed_vals))

    return lexicon


def normalize_grammar_probs(grammar):
    """ Normalize grammar rules count to get probabilities.
    """
    for l_item in grammar:
        if (l_item,) in grammar[l_item]:
            grammar[l_item].pop((l_item,))
        total = sum(grammar[l_item].values())
        normed_vals = map(lambda x: x/total, grammar[l_item].values())
        grammar[l_item] = dict(zip(grammar[l_item].keys(), normed_vals))


def binarize_rules(grammar):
    """ Binarize rules in a PCFG of the form  {"left" : {"right": count}}.
    """
    grm_normal = deepcopy(grammar)

    for item in grammar:
        for tuple_terms in grammar[item]:
            if len(tuple_terms) <= 2:
                continue

            terms = list(tuple_terms)

            while len(terms) > 2:
                grm_normal['_'.join(terms[-2:])] = {
                    tuple(terms[-2:]): 1
                }
                terms[-2] = '_'.join(terms[-2:])
                terms.pop(-1)

            grm_normal[item][tuple(terms)] = grm_normal[item][
                tuple_terms]
            grm_normal[item].pop(tuple_terms)

    return grm_normal


def add_from_match(match, grammar):
    """ Match all "rule" patterns of the form `(ABC (DEF) (GHI)...)` and add
    them to the grammar.
    """
    match_string = match.string[match.start(0):match.end(0)]

    # Split head and body of the rule
    l_item, r_item = match_string[1:-1].split(' ', 1)

    # grammar has a dictionary structure with left rules
    # as roots, and corresponding right rules pointing
    # to their count
    # Ex: {
    #   "SENT": {
    #       ('NP', 'ADV'): 10,
    #       ('ADV', 'PONCT'): 3,
    #       ...
    #   }
    # }
    if l_item not in grammar:
            grammar[l_item] = {}
    children = tuple(map(lambda x: x.strip('()'),
                         r_item.split(' ')))

    grammar[l_item][children] = grammar[l_item].get(children, 0) + 1


def add_words_to_lexicon(tree, lexicon):
    """ Add tokens to lexicon
    """
    for word in leaves.finditer(tree):
        word_string = word.group(0).split(' ')[1].rstrip(')')
        pos_tag = word.group(0).split(' ')[0].lstrip('(')
        if pos_tag not in lexicon:
            lexicon[pos_tag] = {}

        lexicon[pos_tag][word_string] = lexicon[
            pos_tag].get(word_string, 0) + 1


def add_words_to_grammar(tree, grammar):
    """ Add tokens to grammar
    """
    for word in leaves.finditer(tree):
        word_string = word.group(0).split(' ')[1].rstrip(')')
        pos_tag = word.group(0).split(' ')[0].lstrip('(')
        if pos_tag not in grammar:
            grammar[pos_tag] = {}

        grammar[pos_tag][(word_string,)] = grammar[
            pos_tag].get((word_string,), 0) + 1


def process_line(line, lexicon, grammar):
    """ Substract all supplementary tags of the form ABC-DEF and replace with
    ABC only.

    Add words to lexicon and rules to grammar.
    """
    tree = ignores_tags.sub(r'(\g<1>', line.strip('\n')[2:-1])
    add_words_to_lexicon(tree, lexicon)

    # add_words_to_grammar(tree, grammar)
    # Substract token to work only with POS tags
    no_tk = leaves.sub(r'(\g<1>)', tree)
    subd = no_tk

    # Work with the string until the tree has been parsed (reduced
    # to "(SENT)")
    while subd != '(SENT)':
        for match in grammar_rule.finditer(subd):
            add_from_match(match, grammar)

        # Replace all extracted rules by their root to reduce tree
        new_subd = grammar_rule.sub(r'(\g<1>)', subd)
        if new_subd == subd:
            raise ValueError(
                "Uncorrectly formatted string passed as input")
        subd = new_subd


def chomsky_normal_form(grammar, normalize=True):
    """ Pass the given grammar in Chomsky's normal form (CNF). 

    Returns:
        The new grammar in CNF
    """
    grammar = binarize_rules(grammar)
    if normalize:
        normalize_grammar_probs(grammar)
    grammar = eliminate_unit_rules(grammar)
    return grammar


def build_grammar_from_input(lexicon_fname='./lexicon.pkl',
                             grammar_fname='./grammar.pkl',
                             normalize=True):
    """ Builds a grammar using either a file or the standard input.
    Sentences must be correctly formatted and the resulting PCFG is
    saved in Chomsky Normal Form in the file ./grammar.pkl
    """
    grammar = {}
    lexicon = {}

    for line in fileinput.input():
        process_line(line, lexicon, grammar)

    normalize_grammar_probs(lexicon)
    grammar = chomsky_normal_form(grammar)

    pkl.dump(lexicon, open('./lexicon.pkl', 'wb'))
    pkl.dump(grammar, open('./grammar.pkl', 'wb'))

if __name__ == "__main__":
    build_grammar_from_input()
