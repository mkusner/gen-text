from __future__ import print_function
import nltk
from nltk import grammar, parse
import pdb

p = """
smiles -> chain
atom -> aliphatic_organic | aromatic_organic 
aliphatic_organic -> 'B' | 'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'Cl' | 'Br' | 'I'
aromatic_organic -> 'b' | 'c' | 'n' | 'o' | 's' | 'p'
symbol -> element_symbols | aromatic_symbols 
DIGIT -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
element_symbols -> 'H' | 'He' | 'Li' | 'Be' | 'B' | 'C' | 'N' | 'O' | 'F' | 'Ne' | 'Na' | 'Mg' | 'Al' | 'Si' | 'P' | 'S' | 'Cl' | 'Ar' | 'K' | 'Ca' | 'Sc' | 'Ti' | 'V' | 'Cr' | 'Mn' | 'Fe' | 'Co' | 'Ni' | 'Cu' | 'Zn' | 'Ga' | 'Ge' | 'As' | 'Se' | 'Br' | 'Kr' | 'Rb' | 'Sr' | 'Y' | 'Zr' | 'Nb' | 'Mo' | 'Tc' | 'Ru' | 'Rh' | 'Pd' | 'Ag' | 'Cd' | 'In' | 'Sn' | 'Sb' | 'Te' | 'I' | 'Xe' | 'Cs' | 'Ba' | 'Hf' | 'Ta' | 'W' | 'Re' | 'Os' | 'Ir' | 'Pt' | 'Au' | 'Hg' | 'Tl' | 'Pb' | 'Bi' | 'Po' | 'At' | 'Rn' | 'Fr' | 'Ra' | 'Rf' | 'Db' | 'Sg' | 'Bh' | 'Hs' | 'Mt' | 'Ds' | 'Rg' | 'Cn' | 'Fl' | 'Lv' | 'La' | 'Ce' | 'Pr' | 'Nd' | 'Pm' | 'Sm' | 'Eu' | 'Gd' | 'Tb' | 'Dy' | 'Ho' | 'Er' | 'Tm' | 'Yb' | 'Lu' | 'Ac' | 'Th' | 'Pa' | 'U' | 'Np' | 'Pu' | 'Am' | 'Cm' | 'Bk' | 'Cf' | 'Es' | 'Fm' | 'Md' | 'No' | 'Lr'
aromatic_symbols -> 'b' | 'c' | 'n' | 'o' | 'p' | 's' | 'se' | 'as'
hcount -> 'H' | 'H' DIGIT
charge -> '-' | '-' DIGIT | '-' DIGIT DIGIT | '+' | '+' DIGIT | '+' DIGIT DIGIT
class -> ':' DIGIT | ':' DIGIT DIGIT
bond -> '-' | '=' | '#' | '$' | ':' | '/' | '\\'
ringbond -> DIGIT | bond DIGIT | '%' DIGIT DIGIT | bond '%' DIGIT DIGIT
branched_atom -> atom | atom RB | atom BB | atom RB BB
RB -> RB ringbond | ringbond
BB -> BB branch | branch
branch -> '(' chain ')' | '(' bond chain ')' | '(' dot chain ')'
chain -> branched_atom | chain branched_atom | chain bond branched_atom | chain dot branched_atom
dot -> '.'
"""
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
def getNodes(parent):
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'smiles':



#grammar = grammar.FeatureGrammar.fromstring(p)
gr = nltk.CFG.fromstring(p)
test = 'CN1SC=CC1=O.CN2SC(=CC2=O)C'#l' #'NCC(=O)CP(=O)(O)CC1CCCCC1' #'CC(=O)NCCc1ccccc1'
rd_parser = nltk.ChartParser(gr)
#rd_parser = nltk.ShiftReduceParser(gr)#RecursiveDescentParser(gr)
for tree in rd_parser.parse(list(test)): #list(test)):
    print('going')
    print(tree)
pdb.set_trace()


