# GSDMM: Short text clustering

This project implements the Gibbs sampling algorithm for a Dirichlet Mixture Model of [Yin and Wang 2014](https://pdfs.semanticscholar.org/058a/d0815ce350f0e7538e00868c762be78fe5ef.pdf) for the 
clustering of short text documents. 
Some advantages of this algorithm:
 - It requires only an upper bound `K` on the number of clusters
 - With good parameter selection, the model converges quickly
 - Space efficient and scalable

This project is an easy to read reference implementation of GSDMM -- I don't plan to maintain it unless there is demand. I am however actively maintaining the much faster Rust version of GSDMM [here](https://github.com/rwalk/gsdmm-rust).

## The Movie Group Process
In their paper, the authors introduce a simple conceptual model for explaining the GSDMM called the Movie Group Process.

Imagine a professor is leading a film class. At the start of the class, the students
are randomly assigned to `K` tables. Before class begins, the students make lists of
their favorite films. The professor repeatedly reads the class role. Each time the student's name is called,
the student must select a new table satisfying one or both of the following conditions:

- The new table has more students than the current table.
- The new table has students with similar lists of favorite movies.

By following these steps consistently, we might expect that the students eventually arrive at an "optimal" table configuration.

## Installation
Enter the directory where `setup.py` stands, and in the command line, use:
```
python setup.py install
```

## Usage
To use a Movie Group Process to cluster short texts, first initialize a [MovieGroupProcess](gsdmm/mgp.py):
```python
from gsdmm import MovieGroupProcess
mgp = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=30)
```
`K` is the largest expected number of clusters in your data, as the algorithm will assign one of the `K` clusters to each doc, and some clusters may not be used.

To fit the model:
```python
docs = [['short','text','A'],
        ['another','short','text'],
        ['others']]
vocab = set(x for doc in docs for x in doc)
y = mgp.fit(docs,len(vocab))
```
`y` is the clustering result. 
Note that: Each doc in `docs` must be a unique list of tokens found in your short text document. And this implementation does not support counting tokens with multiplicity (which generally has little value in short text documents).

To classify a new sample:
```python
doc = ['new','short','text']
label, probability = mgp.choose_best_label(doc)
```

To get the word importance for each topic: _(not in the original version, but my own implemented according to the paper.)_
```python
def cluster_importance(mgp):
    n_z_w = mgp.cluster_word_distribution
    beta, V, K = mgp.beta, mgp.vocab_size, mgp.K
    phi = [{} for i in range(K)]        
    for z in range(K):
        for w in n_z_w[z]:
            phi[z][w] = (n_z_w[z][w]+beta)/(sum(n_z_w[z].values())+V*beta)
    return phi
phi = cluster_importance(mgp)
```
`phi[i][w]` would be the importance of word `w` in topic `i`.
