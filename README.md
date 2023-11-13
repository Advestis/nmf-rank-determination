## On Rank Selection in Non-Negative Matrix Factorization using Concordance</br><sub>Official *GitHub* repository for the original *MDPI Mathematics* paper</sub>

![Graphical_abstract](images/graphical_abstract.png)
*Credits*: DALL·E 3

**On Rank Selection in Non-Negative Matrix Factorization using Concordance**<br>
Paul Fogel, Christophe Geissler, Nicolas Morizet and George Luta<br>
**DOI**: https://doi.org/10.3390/math11224611

This article belongs to the Special Issue [Advances in Applied Probability and Statistical Inference](https://www.mdpi.com/journal/mathematics/special_issues/Advances_Applied_Probability_Statistical_Inference).

**Abstract**: The choice of the factorization rank of a matrix is critical, e.g., in dimensionality reduction,
filtering, clustering, deconvolution, etc., because selecting a rank that is too high amounts to adjusting the noise,
while selecting a rank that is too low results in the oversimplification of the signal.
Numerous methods for selecting the factorization rank of a non-negative matrix have been proposed.
One of them is the cophenetic correlation coefficient (*ccc*), widely used in data science to evaluate the number of
clusters in a hierarchical clustering. In previous work, it was shown that 𝑐𝑐𝑐 performs better than other methods
for rank selection in non-negative matrix factorization (NMF) when the underlying structure of the matrix consists
of orthogonal clusters. In this article, we show that using the ratio of *ccc* to the approximation error significantly
improves the accuracy of the rank selection. We also propose a new criterion, *concordance*, which, like *ccc*,
benefits from the stochastic nature of NMF; its accuracy is also improved by using its ratio-to-error form.
Using real and simulated data, we show that *concordance*, with a CUSUM-based automatic detection algorithm
for its original or ratio-to-error forms, significantly outperforms *ccc*.
It is important to note that the new criterion works for a broader class of matrices,
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
@Article{math11224611,
AUTHOR = {Fogel, Paul and Geissler, Christophe and Morizet, Nicolas and Luta, George},
TITLE = {On Rank Selection in Non-Negative Matrix Factorization Using Concordance},
JOURNAL = {Mathematics},
VOLUME = {11},
YEAR = {2023},
NUMBER = {22},
ARTICLE-NUMBER = {4611},
URL = {https://www.mdpi.com/2227-7390/11/22/4611},
ISSN = {2227-7390},
DOI = {10.3390/math11224611}
}
```