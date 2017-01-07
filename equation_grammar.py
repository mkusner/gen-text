
from __future__ import print_function
import nltk
from nltk import grammar, parse
import pdb
from nltk.parse.generate import generate
from molecules.utils import many_one_hot
import numpy as np
import h5py
import six

p = """S -> S '+' S
S -> S '*' S
S -> S '/' S
S -> '(' S ')'
S -> 'sin(' S ')'
S -> 'exp(' S ')'
S -> 'x'
S -> '1'
S -> '2'
S -> '3'"""
###S -> '4'
###S -> '5'
###S -> '6'
###S -> '7'
###S -> '8'
###S -> '9'"""

leaves = ['+','*','/','(',')','sin(','exp(','x']
rules = p.split('\n')


gr = nltk.CFG.fromstring(p)
parser = nltk.ChartParser(gr)
string = 'sin( x ) + 3'
AA = parser.parse(string.split(' '))
BB = AA.next()

indices = []
A = []
for prod in BB.productions():
    ind = rules.index(prod.__unicode__())
    indices.append(ind)
    A.append(gr.productions()[ind])


def prod_to_string(P,string):
    tup = P[0].rhs()
    for item in tup:
        print(P)
        if isinstance(item,six.string_types):
            string = string + ' ' + item
        else:
            P.pop(0)
            string = prod_to_string(P, string)
    return string

st = prod_to_string(A,'')
pdb.set_trace()


print("let's generate!")
GEN = list(generate(gr, depth=4))#, n=100000)
f = open('equation_dataset.txt','w')
OH = np.zeros((len(GEN),7,len(rules)+1))

count = 0
for sen in GEN:
    sen_str = ' '.join(sen)
    AA = parser.parse(sen_str.split(' '))
    BB = AA.next()
    #if len(BB.productions()) != 7:
    #    OH = np.delete(OH, count, 0)
    #    continue
    inds = []
    for prod in BB.productions():
        inds.append(rules.index(prod.__unicode__()))
    if len(inds) < 7:
        inds.extend((7-len(inds))*[len(rules)])
    OH[count,:,:] = many_one_hot(np.array(inds), len(rules)+1) # (7,17)
    count = count + 1
    f.write(sen_str + '\n')

f.close()
h5f = h5py.File('eq_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.close()
pdb.set_trace()
