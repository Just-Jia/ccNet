# ccnet

*Ccnet*, cell-cell network, is a single-cell RNA sequencing data analysis package based on non-uniform epsilon-neighborhood network (NEN) [(Bioinformatics, 2022)](https://doi.org/10.1093/bioinformatics/btac114).

## Features

- Different from the traditional analysis of scRNA-seq data, which performs visualization, clustering and trajectory inference using methods based on different theories, *ccnet* accomplishes the three targets in a consistent manner.
- NEN network combines the advantages of both k-neighbors (KNN) and epsilon-neighborhood (EN) to represent the intrinsic manifold of data.


## Installation

Install *ccnet* from pip:

	pip install ccnet

Or, to build and install run from source:

	python setup.py install

## Usage

For the usage of *ccnet*, please refer to the [example](example/example_guo2010.ipynb), where we introduce the relevant analysis steps, including visualization, clustering, pseudotime ordering, finding trajectory-associated genes, etc.

## Contribute

Source Code: [https://github.com/Just-Jia/ccNet.git](https://github.com/Just-Jia/ccNet.git)

## Contacts

My email: junbo_jia@163.com

## Cite
Junbo Jia, Luonan Chen, Single-cell RNA sequencing data analysis based on non-uniform ε−neighborhood network, Bioinformatics, 2022;, btac114,
doi:[https://doi.org/10.1093/bioinformatics/btac114](https://doi.org/10.1093/bioinformatics/btac114)

## License

The project is licensed under the GNU GPLv3 license.
