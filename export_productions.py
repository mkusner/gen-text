
from collections import defaultdict

def export(G):

    order = []

    prod_dict = defaultdict(list)
    for prod in G.productions():
        key = str(prod.lhs())
        if len(order) == 0 or order[-1] != key:
            order.append(key)
        prod_dict[key].append(' '.join(map(str, prod.rhs())))

    res = []
    for key in order:
        res.append(key + ' $\\rightarrow$ ' + ' | '.join(prod_dict[key]))
    return '\n'.join(res)
