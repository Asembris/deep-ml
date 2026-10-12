def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    d={}
    for e in corpus.split():
        d[e]=d.get(e,0)+1
    return d.get(word,0)/len(corpus.split())
