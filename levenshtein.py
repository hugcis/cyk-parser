import numpy as np
import fileinput

def levenshtein_distance(s_a, s_b):
    m = len(s_a)
    n = len(s_b)

    if m < n:
        return levenshtein_distance(s_b, s_a)

    if not n:
        return m
      
    p_row = range(n + 1)
 
    for i, t in enumerate(s_a):
        c_row = [i + 1] 
        for j, u in enumerate(s_b):
            ins = p_row[j + 1] + 1 
            dele = c_row[j] + 1      
            sub = p_row[j] + (t != u)
            c_row.append(min(ins, dele, sub))
        p_row = c_row
    
    return p_row[-1]


if __name__ == "__main__":
    for line in fileinput.input():
        print(levenshtein_distance(*line.rstrip().split(' ')))