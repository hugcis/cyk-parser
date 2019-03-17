class NotInGrammarError(Exception):
    pass

class Tree:
    def __init__(self, val, children, word=None):
        self.val = val
        self.children = children
        self.word = word

    def de_binarize(self):
        if len(self.val.split('_')) > 1:
            ret = []
            for child in self.children:
                debinarized = child.de_binarize()
                if isinstance(debinarized, list):
                    ret = ret + debinarized
                else:
                    ret.append(debinarized)
            return ret
        else:
            new_children = []
            for child in self.children:
                debinarized = child.de_binarize()
                if isinstance(debinarized, list):
                    new_children = new_children + debinarized
                else:
                    new_children.append(debinarized)
            self.children = new_children
            return self

    @classmethod
    def build_from_stem(self, stem, tokens):
        subs = stem[0][len(stem) - 1].get('SENT')
        if subs is None:
            raise NotInGrammarError(
                "Sentence could not be produced with the grammar")

        if len(tokens) == 1:
            return Tree('SENT', 
                        [Tree(subs[1], 
                              [Tree(subs[2], [], tokens[0])])
                        ])
        
        
        base_tree = Tree('SENT', 
                         [Tree.build_tree(stem, 0, 
                                          subs[0], subs[1], 
                                          tokens),
                         Tree.build_tree(stem, subs[0], 
                                         len(stem) - 1, subs[2], 
                                         tokens)])
        return base_tree
        

    @classmethod
    def build_tree(self, stem, l_idx, r_idx, val, tokens):
        if r_idx - l_idx <= 1:
            return Tree(val, [], word=tokens.pop(0))
        else:
            subs = stem[l_idx][r_idx][val]
            return Tree(val, 
                        [Tree.build_tree(stem, l_idx, subs[0], 
                                         subs[1], tokens),
                        Tree.build_tree(stem, subs[0], 
                                        r_idx, subs[2], tokens)])

    def __str__(self):
        if len(self.children):
            return '({} {})'.format(str(self.val),
                                    ' '.join(map(str, self.children)))
        else:
            return '({} {})'.format(str(self.val), str(self.word))