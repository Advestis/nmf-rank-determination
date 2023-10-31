## On Rank Selection in Non-Negative Matrix Factorization using Concordance</br><sub>Official *GitHub* repository for the original MDPI Mathematics paper</sub>

![Graphical_abstract](images/graphical_abstract.png)
*Credits*: DALL·E 3

**On Rank Selection in Non-Negative Matrix Factorization using Concordance**<br>
Paul Fogel, Christophe Geissler, Nicolas Morizet and George Luta<br>
DOI: xxx

**Abstract**: The choice of the factorization rank of a matrix is critical, e.g.,
in dimensionality reduction, filtering, clustering, deconvolution, etc.,
because selecting a rank that is too high amounts to adjusting the noise,
while selecting a rank that is too low results in oversimplification of the signal.
Numerous methods for selecting the factorization rank of a non-negative matrix
have been proposed. One of them is the cophenetic correlation coefficient (*ccc*),
widely used in data science to evaluate the number of clusters
in a hierarchical clustering, which was first introduced in [*Brunet, 2004*].
In [*Maisog, 2021*] it was shown that *ccc* performs better than other methods
for rank selection in non-negative matrix factorization (NMF) when the underlying structure
of the matrix consists of orthogonal clusters.
In this article, we show that using the ratio of *ccc* to the approximation error
significantly improves the accuracy of the rank selection.
We also propose a new criterion, **concordance**, which, like *ccc*,
benefits from the stochastic nature of NMF;  its accuracy is also improved
by using its ratio-to-error form. Using real and simulated data,
we show that **concordance**, with a CUSUM-based automatic detection algorithm
for its original or ratio-to-error forms, significantly outperforms *ccc*.
It is important to note that the new criterion works with a broader class of matrices,
where the underlying clusters are not assumed to be orthogonal.

**Datasets used in the study**</br>
- Swimmer
- Sausage
- Brunet
- MNIST

**Requirements**
- Python 3.9.0
- scikit-learn 
- pandas 
- numpy 
- scipy 
- tqdm

## Citation
```

```