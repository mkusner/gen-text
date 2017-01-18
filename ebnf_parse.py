from __future__ import print_function
import nltk
from nltk import grammar, parse
import pdb
import re
from molecules.utils import many_one_hot
import mol_grammar
import numpy as np
import h5py
import six

p = """
smiles -> chain
atom -> bracket_atom | aliphatic_organic | aromatic_organic | '*'
aliphatic_organic -> 'B' | 'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'Cl' | 'Br' | 'I'
aromatic_organic -> 'b' | 'c' | 'n' | 'o' | 's' | 'p'
bracket_atom -> '[' BAI ']' 
BAI  -> isotope symbol BAC | symbol BAC | isotope symbol | symbol
BAC  -> chiral BAH | BAH | chiral
BAH  -> hcount BACH | BACH | hcount
BACH -> charge class | charge | class
symbol -> element_symbols | aromatic_symbols | '*'
isotope -> DIGIT | DIGIT DIGIT | DIGIT DIGIT DIGIT
DIGIT -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
element_symbols -> 'H' | 'He' | 'Li' | 'Be' | 'B' | 'C' | 'N' | 'O' | 'F' | 'Ne' | 'Na' | 'Mg' | 'Al' | 'Si' | 'P' | 'S' | 'Cl' | 'Ar' | 'K' | 'Ca' | 'Sc' | 'Ti' | 'V' | 'Cr' | 'Mn' | 'Fe' | 'Co' | 'Ni' | 'Cu' | 'Zn' | 'Ga' | 'Ge' | 'As' | 'Se' | 'Br' | 'Kr' | 'Rb' | 'Sr' | 'Y' | 'Zr' | 'Nb' | 'Mo' | 'Tc' | 'Ru' | 'Rh' | 'Pd' | 'Ag' | 'Cd' | 'In' | 'Sn' | 'Sb' | 'Te' | 'I' | 'Xe' | 'Cs' | 'Ba' | 'Hf' | 'Ta' | 'W' | 'Re' | 'Os' | 'Ir' | 'Pt' | 'Au' | 'Hg' | 'Tl' | 'Pb' | 'Bi' | 'Po' | 'At' | 'Rn' | 'Fr' | 'Ra' | 'Rf' | 'Db' | 'Sg' | 'Bh' | 'Hs' | 'Mt' | 'Ds' | 'Rg' | 'Cn' | 'Fl' | 'Lv' | 'La' | 'Ce' | 'Pr' | 'Nd' | 'Pm' | 'Sm' | 'Eu' | 'Gd' | 'Tb' | 'Dy' | 'Ho' | 'Er' | 'Tm' | 'Yb' | 'Lu' | 'Ac' | 'Th' | 'Pa' | 'U' | 'Np' | 'Pu' | 'Am' | 'Cm' | 'Bk' | 'Cf' | 'Es' | 'Fm' | 'Md' | 'No' | 'Lr'
aromatic_symbols -> 'b' | 'c' | 'n' | 'o' | 'p' | 's' | 'se' | 'as' | 'te'
chiral -> '@' | '@@'
hcount -> 'H' | 'H' DIGIT
charge -> '-' | '-' DIGIT | '-' DIGIT DIGIT | '+' | '+' DIGIT | '+' DIGIT DIGIT
class -> ':' DIGIT | ':' DIGIT DIGIT | ':' DIGIT DIGIT DIGIT
bond -> '-' | '=' | '#' | '$' | ':' | '/' | '\\'
ringbond -> DIGIT | bond DIGIT | '%' DIGIT DIGIT | bond '%' DIGIT DIGIT
branched_atom -> atom | atom RB | atom BB | atom RB BB
RB -> RB ringbond | ringbond
BB -> BB branch | branch
branch -> '(' chain ')' | '(' bond chain ')' | '(' dot chain ')'
chain -> branched_atom | chain branched_atom | chain bond branched_atom | chain dot branched_atom
dot -> '.'
"""


two_char_chems = ['He' , 'Li' , 'Be' , 'Ne' , 'Na' , 'Mg' , 'Al' , 'Si' , 'Cl' , 'Ar' , 'Ca' , 'Sc' , 'Ti' ,'Cr' , 'Mn' , 'Fe' , 'Co' , 'Ni' , 'Cu' , 'Zn' , 'Ga' , 'Ge' , 'As' , 'Se' , 'Br' , 'Kr' , 'Rb' , 'Sr' , 'Zr' , 'Nb' , 'Mo' , 'Tc' , 'Ru' , 'Rh' , 'Pd' , 'Ag' , 'Cd' , 'In' , 'Sn' , 'Sb' , 'Te' , 'Xe' , 'Cs' , 'Ba' , 'Hf' , 'Ta' , 'Re' , 'Os' , 'Ir' , 'Pt' , 'Au' , 'Hg' , 'Tl' , 'Pb' , 'Bi' , 'Po' , 'At' , 'Rn' , 'Fr' , 'Ra' , 'Rf' , 'Db' , 'Sg' , 'Bh' , 'Hs' , 'Mt' , 'Ds' , 'Rg' , 'Cn' , 'Fl' , 'Lv' , 'La' , 'Ce' , 'Pr' , 'Nd' , 'Pm' , 'Sm' , 'Eu' , 'Gd' , 'Tb' , 'Dy' , 'Ho' , 'Er' , 'Tm' , 'Yb' , 'Lu' , 'Ac' , 'Th' , 'Pa' , 'Np' , 'Pu' , 'Am' , 'Cm' , 'Bk' , 'Cf' , 'Es' , 'Fm' , 'Md' , 'No' , 'Lr', 'se', 'as', 'te', '@@']

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


f = open('chembl_21_chemreps.txt','r')
L = ['']*1583897

count = -1
for line in f:
    parts = line.split()
    count = count+1
    if count == 0:
        continue
    L[count-1] = parts[1]
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


R = [string for string in L if len(string) <= 35]
#R = ['CC(C)(C)c1cc(cc([teg+]1)C(C)(C)C)c2ccccc2']

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
rules = mol_grammar.p.split('\n')

OH = np.zeros((len(R),100,len(rules)))
##L2 = ['COc1ccc2[C@@H]3ClcccBrcc']

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
    #print(len(BB.productions()))
    if len(BB.productions()) > 100:
        continue
    #if len(BB.productions()) == 84:
    #    pdb.set_trace()
    #pdb.set_trace()
    inds = []
    for prod in BB.productions():
        inds.append(rules.index(prod.__unicode__()))
    if len(inds) < 100:
        inds.extend((100-len(inds))*[len(rules)-1])
    OH[total_count,:,:] = many_one_hot(np.array(inds), len(rules)) # (7,17)
    total_count = total_count + 1

h5f = h5py.File('mol100_dataset.h5','w')
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
    


