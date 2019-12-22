import fractions as f
def get_small_fractions(max_nom, max_denom):
    s=set()
    for n in range(1,max_nom+1):
        for d in range(1,max_denom+1):
            s.add(f.Fraction(n,d))
    return s
def get_simple_aspect_ratios(max_nom, max_denom):
    s=get_small_fractions(max_nom, max_denom)
    return {x if x <= 1 else 1 for x in s}
