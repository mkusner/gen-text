from __future__ import print_function
import nltk
from nltk import grammar, parse
import pdb
import re
from molecules.utils import many_one_hot
import zinc_grammar
import numpy as np
import h5py
import six

p = """
smiles -> chain
atom -> bracket_atom | aliphatic_organic | aromatic_organic
aliphatic_organic -> 'B' | 'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'I' | 'Cl' | 'Br'
aromatic_organic -> 'c' | 'n' | 'o' | 's'
bracket_atom -> '[' BAI ']' 
BAI  -> isotope symbol BAC | symbol BAC | isotope symbol | symbol
BAC  -> chiral BAH | BAH | chiral
BAH  -> hcount BACH | BACH | hcount
BACH -> charge class | charge | class
symbol -> aliphatic_organic | aromatic_organic
isotope -> DIGIT | DIGIT DIGIT | DIGIT DIGIT DIGIT
DIGIT -> '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8'
chiral -> '@' | '@@'
hcount -> 'H' | 'H' DIGIT
charge -> '-' | '-' DIGIT | '-' DIGIT DIGIT | '+' | '+' DIGIT | '+' DIGIT DIGIT
bond -> '-' | '=' | '#' | '/' | '\\'
ringbond -> DIGIT | bond DIGIT
branched_atom -> atom | atom RB | atom BB | atom RB BB
RB -> RB ringbond | ringbond
BB -> BB branch | branch
branch -> '(' chain ')' | '(' bond chain ')'
chain -> branched_atom | chain branched_atom | chain bond branched_atom
"""


two_char_chems = ['Cl', 'Br', '@@']

special_chems = ['Cl', 'Br']
#smiles -> terminator | chain terminator
#terminator -> SPACE | TAB | LINEFEED | CARRIAGE_RETURN | END_OF_STRING
#"""

#groucho_grammar = nltk.CFG.fromstring("""
#S -> NP VP
#PP -> P NP | PP
#NP -> Det N | Det N PP | 'I'
#VP -> V NP | VP PP
#Det -> 'an' | 'my'
#N -> 'elephant' | 'pajamas'
#V -> 'shot'
#P -> 'in'
#""")
#sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
#parser = nltk.ChartParser(groucho_grammar)
#for tree in parser.parse(sent):
#    print(tree)
#####def getNodes(parent):
#####    for node in parent:
#####        if type(node) is nltk.Tree:
#####            if node.label() == 'smiles':


f = open('molecule-autoencoder/data/250k_rndm_zinc_drugs_clean.smi','r')
L = []

count = -1
for line in f:
    line = line.strip()
    L.append(line)
f.close()

def iter_flatten(iterable):
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


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

#AA = [len(string) for string in L]
#pdb.set_trace()
#R = [string for string in L if len(string) <= 35]
#R = ['CC(C)(C)c1cc(cc([teg+]1)C(C)(C)C)c2ccccc2']
R = L

#grammar = grammar.FeatureGrammar.fromstring(p)
gr = nltk.CFG.fromstring(p)
##test = 'CN1SC=CC1=O.CN2SC(=CC2=O)C'#l' #'NCC(=O)CP(=O)(O)CC1CCCCC1' #'CC(=O)NCCc1ccccc1'
parser = nltk.ChartParser(gr)
####test = 'CN1C(=O)[Se]c2cc(ccc12)C(=O)c3cccnc3'
####test = 'c2cc(ccc12)'
####test = list('CN1C(=O)[')
####test.extend(['Se',']'])
####test = ['[','Se',']']
####print(test)
####AA = parser.parse(test)#list(test))
####BB = AA.next()
####pdb.set_trace()



#########rd_parser = nltk.ShiftReduceParser(gr)#RecursiveDescentParser(gr)
##for tree in rd_parser.parse(list(test)): #list(test)):
##    print('going')
##    print(tree)
##pdb.set_trace()
rules = zinc_grammar.p.split('\n')

OH = np.zeros((len(R),277,len(rules)))
##L2 = ['COc1ccc2[C@@H]3ClcccBrcc']
###prod_len = [0]*len(R)
total_count = 0
COU = 0
for chem in R:
    print(total_count+1)
    split_locs = []
    #print(chem)
    print(COU)
    #if COU == 41:
    #    pdb.set_trace()
    COU = COU + 1
    if '[' in chem:
        split = filter(None,re.split('(\[.+?\])',chem))
        count = 0
        for el in split:
            if el[0] == '[':
                for c in two_char_chems:
                    if c in el:
                        result = filter(None,re.split('(' + c + ')', el))
                        split[count] = result
            count = count + 1
        C = [i for i in iter_flatten(split)]
    else:
        C = [chem]

    for c in special_chems:
        count = 0
        happened = 0
        for part in C:
            if c in part:
                #s =  [e+c for e in chem.split(c)]
                result = filter(None,re.split('(' + c + ')',part))
                C[count] = result
                happened = 1
            count = count + 1
        C = [i for i in iter_flatten(C)]
    count = 0
    for part in C:
        if part not in two_char_chems:
            C[count] = list(C[count])
        count = count + 1
    C = [i for i in iter_flatten(C)]
    #print(C)
    try:
        AA = parser.parse(C)
    except ValueError as err:
        print(err)
        continue
    BB = AA.next()
    print(len(BB.productions()))
    #prod_len[total_count] = len(BB.productions())
    #if len(BB.productions()) > 277:
    #    continue
    inds = []
    for prod in BB.productions():
        inds.append(rules.index(prod.__unicode__()))
    if len(inds) < 277:
        inds.extend((277-len(inds))*[len(rules)-1])
    OH[total_count,:,:] = many_one_hot(np.array(inds), len(rules)) # (7,17)
    total_count = total_count + 1
pdb.set_trace()
h5f = h5py.File('zinc_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.close()

        #if happened:
        #    pdb.set_trace()
                #pdb.set_trace()
###            B = [m.start() for m in re.finditer(c, chem)]
###            split_locs.extend(B)
###    
###    split_loc.sort() 
###    for sl in split_loc:
###
###    zipped = zip(split_loc, split_loc+2)
###    if 0 in split_loc:
###        zipped`
###    if len(split_loc)-1 in (split_loc+2):
###
###    split_locs.insert(0,0)
    


