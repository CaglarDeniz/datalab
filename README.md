# Latent Semantic Analysis for Text Summarization

## Background

This is a repository implementing Latent Semantic Summarization from 
[this](https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis) paper and some form of abstractive summarization using [this](https://towardsdatascience.com/abstractive-summarization-using-pytorch-f5063e67510?gif=true) short guide. You can find a copy of my report summarizing understanding of the implementation and my comparison between the two methods in the repo, and you can acces my video demoing my implementation [here](https://drive.google.com/file/d/1fxAqbIHXi9UkdUKIJLHCdk0PNp6K1faJ/view?usp=sharing)

## Requirements

I've included a YAML file which holds all the packages that I've installed to 
implement these text summarization systems. The file also includes other packages that were installed for convenience and simplicity and may be uninstalled/edited out of 
the YAML file.

To construct a conda environment from these packages, simply install Anaconda 
or miniconda by following the instructions [here](https://docs.anaconda.com/anaconda/install/index.html)

After installing Anaconda or miniconda run 

``` 
conda env create --file=ai.yaml
```

## Running with your text 

To run the python script with your own text segment or document simply edit the 
sample strings provided to hold your own document and run the script with 
```
python3 cross-extraction/lsa-cross.py
```

Please note that the LSA cross method implementation requires no newlines to be present in the string to be able to parse into sentences.

To run the abstractive summarization code again edit the python file to include your own text and run 
```
python3 transformer/transformer.py
```
