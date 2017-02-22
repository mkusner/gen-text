import numpy as np
import glob
import gzip
import cPickle as pickle
import sys

if __name__ == '__main__':
    SEED = int(sys.argv[1])
    print "SEED:", SEED

def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret
    
for model in ('grammar', 'character'):

    search = 'bo_optimization_%s/results_gp_multiple_iterations/scores*.seed_%d.dat' % (model, SEED)
    N = len(glob.glob(search))
    best = 10000000
    best_eq = ''
    best_iter = -1
    
    print
    print "***********************"
    print "MODEL: %s" % model
    print "number of iterations: %d" % N
    print "***********************\n\n"
    
    s = 'parse_' if model == 'character' else ''

    for i in xrange(N):
        scores = load_object("bo_optimization_%s/results_gp_multiple_iterations/scores_%s%d.seed_%d.dat" % (model, s, i, SEED))
        eq = load_object("bo_optimization_%s/results_gp_multiple_iterations/valid_eq_%s%d.seed_%d.dat" % (model, s, i, SEED))
        scmin = scores[np.argmin(scores)]
        if best > scmin:
            best = scmin
            best_eq = eq[np.argmin(scores)]
            best_iter = i
        print i, scmin, eq[np.argmin(scores)]
    
    print
    if N > 0:
        print "overall best:", best_eq
        print "score:", best
        print "RMSE estimate:", np.sqrt(np.exp(best)-1)
        print "found on iteration:", best_iter