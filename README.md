# Unsupervised-Non-transferable-Text-Classification

## Overview

We propose a novel unsupervised non-transferable learning method for the text classification task that does not require annotated target domain data. We further introduce a secret key component in our approach for recovering the access to the target domain, where we design both an explicit (prompt secret key) and an implicit method (adapter secret key) for doing so. 

![overview](./overview.pdf)

## Install dependencies

Run the following scripts to install the dependencies.

```shell
pip install -r requirements.txt
```



## Training

Create a directory `outputs` for storing the checkpoints by:

```shell
mkdir outputs
```

Run the scripts to train the UNTL model.

```shell
python UNTL.py
```

As for the secret key based methods, run the following scripts to train the models

* Train the prompt secret key based model

  ```sh
  python UNTL_with_prefix.py
  ```

* Train the adapter secret key based model

  ```sh
  python UNTL_with_adapter.py
  ```



## Evaluatoin

After finishing training, run the following scripts for evaluating the model.

1. Evaluate the UNTL model

   ```shell
   python predict.py
   ```

2. Evaluate the prompt secret key based model

   ```sh
   python predict_prefix.py
   ```

3. Evaluate the adapter secret key based model

   ```shell
   python predict_adapter.py
   ```



## Reference

```
@article{DBLP:journals/corr/abs-2210-12651,
  author    = {Guangtao Zeng and
               Wei Lu},
  title     = {Unsupervised Non-transferable Text Classification},
  journal   = {CoRR},
  volume    = {abs/2210.12651},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2210.12651},
  doi       = {10.48550/arXiv.2210.12651},
  eprinttype = {arXiv},
  eprint    = {2210.12651},
  timestamp = {Fri, 28 Oct 2022 14:21:57 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2210-12651.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

