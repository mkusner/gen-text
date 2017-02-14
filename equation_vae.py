import nltk
import re

import the_grammar
import molecule_vae
import molecules.model_gr


def tokenize(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn+'(', fn+' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn+'(')
    return s.split()

class EquationGrammarModel(molecule_vae.ZincGrammarModel):
    
    def __init__(self, weights_file, latent_rep_size=2):
        """ Load the (trained) equation encoder/decoder, grammar model. """
        self._grammar = the_grammar
        self._model = molecules.model_gr
        self.MAX_LEN = 15 # TODO: read from elsewhere
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = tokenize
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix
        self.vae = self._model.MoleculeVAE()
        self.vae.load(self._productions, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)

