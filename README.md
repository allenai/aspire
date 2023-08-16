### Aspire
Repository accompanying paper for modeling fine grained similarity between documents: 

**Title**: "Multi-Vector Models with Textual Guidance for Fine-Grained Scientific Document Similarity"

**Authors**: Sheshera Mysore, Arman Cohan, Tom Hope

**Abstract**: We present a new scientific document similarity model based on matching fine-grained aspects of texts. To train our model, we exploit a naturally-occurring source of supervision: sentences in the full-text of papers that cite multiple papers together (co-citations). Such co-citations not only reflect close paper relatedness, but also provide textual descriptions of how the co-cited papers are related. This novel form of textual supervision is used for learning to match aspects across papers. We develop multi-vector representations where vectors correspond to sentence-level aspects of documents, and present two methods for aspect matching: (1) A fast method that only matches single aspects, and (2) a method that makes sparse multiple matches with an Optimal Transport mechanism that computes an Earth Mover's Distance between aspects. Our approach improves performance on document similarity tasks in four datasets. Further, our fast single-match method achieves competitive results, paving the way for applying fine-grained similarity to large scientific corpora. 

The pre-print can be accessed here: https://arxiv.org/abs/2111.08366

**NEWS:** This work has been accepted to NAACL 2022, stay tuned for the camera-ready paper and additional artifacts.

### Contents
1. [Artifacts](#artifacts)
    1. [HF Models](#models)
    1. [Evaluation Datasets](#evaldata)
1. [Model Usage Instructions](#modelusage)
1. [Repository Contents](#repocontents)
1. [Acknowledgements](#acks)
1. [Citation](#citation)
1. [TODOs](#todos)


### Artifacts <a name="artifacts"></a>

#### Models <a name="models"></a>

Models described in the paper are released as Hugging Face models:

`otAspire`:

- [`allenai/aspire-contextualsentence-multim-compsci`](https://huggingface.co/allenai/aspire-contextualsentence-multim-compsci)
- [`allenai/aspire-contextualsentence-multim-biomed`](https://huggingface.co/allenai/aspire-contextualsentence-multim-biomed)

`tsAspire`: 

- [`allenai/aspire-contextualsentence-singlem-compsci`](https://huggingface.co/allenai/aspire-contextualsentence-singlem-compsci)
- [`allenai/aspire-contextualsentence-singlem-biomed`](https://huggingface.co/allenai/aspire-contextualsentence-singlem-biomed)


`SPECTER-CoCite`: 

- [`allenai/aspire-biencoder-compsci-spec`](https://huggingface.co/allenai/aspire-biencoder-compsci-spec)
- [`allenai/aspire-biencoder-biomed-scib`](https://huggingface.co/allenai/aspire-biencoder-biomed-scib)
- [`allenai/aspire-biencoder-biomed-spec`](https://huggingface.co/allenai/aspire-biencoder-biomed-spec)

`cosentbert`: 

- [`allenai/aspire-sentence-embedder`](https://huggingface.co/allenai/aspire-sentence-embedder)

#### Model Usage Instructions <a name="modelusage"></a>

##### `tsAspire`

The `tsAspire` multi-vector model trained for single matches across documents can be used via the `transformers` library and some additional code to compute contextual sentence vectors as:

```python
from transformers import AutoTokenizer
from examples.ex_aspire_consent import AspireConSent, prepare_abstracts

# Initialize the tokenizer and model.
hf_model_name = 'allenai/aspire-contextualsentence-singlem-compsci'
aspire_tok = AutoTokenizer.from_pretrained(hf_model_name)
aspire_mv_model = AspireConSent(hf_model_name)


# Example input.
ex_abstracts = [
    {'TITLE': "Multi-Vector Models with Textual Guidance for Fine-Grained Scientific"
              " Document Similarity",
     'ABSTRACT': ["We present a new scientific document similarity model based on "
                  "matching fine-grained aspects of texts.",
                  "To train our model, we exploit a naturally-occurring source of "
                  "supervision: sentences in the full-text of papers that cite multiple "
                  "papers together (co-citations)."]},
    {'TITLE': "CSFCube -- A Test Collection of Computer Science Research Articles for "
              "Faceted Query by Example",
     'ABSTRACT': ["Query by Example is a well-known information retrieval task in which"
                  " a document is chosen by the user as the search query and the goal is "
                  "to retrieve relevant documents from a large collection.",
                  "However, a document often covers multiple aspects of a topic.",
                  "To address this scenario we introduce the task of faceted Query by "
                  "Example in which users can also specify a finer grained aspect in "
                  "addition to the input query document. "]}
]

bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=ex_abstracts,
                                                          pt_lm_tokenizer=aspire_tok)
clsreps, contextual_sent_reps = aspire_mv_model.forward(bert_batch=bert_batch,
                                                        abs_lens=abs_lens,
                                                        sent_tok_idxs=sent_token_idxs)
```

##### `otAspire`

The `otAspire` multi-vector model trained for _multiple_ matching across documents can be used via the `transformers` library, and some additional code to compute contextual sentence vectors and to make multiple matches using optimal transport. 

View example usage and sample document matches here: [`examples/demo-contextualsentence-multim.ipynb`](https://github.com/allenai/aspire/blob/main/examples/demo-contextualsentence-multim.ipynb)

##### `SPECTER-CoCite`

The `SPECTER-CoCite` bi-encoder model can be used via the `transformers` library as:

```python
from transformers import AutoModel, AutoTokenizer
aspire_bienc = AutoModel.from_pretrained('allenai/aspire-biencoder-compsci-spec')
aspire_tok = AutoTokenizer.from_pretrained('allenai/aspire-biencoder-compsci-spec')
title = "Multi-Vector Models with Textual Guidance for Fine-Grained Scientific "
        "Document Similarity"
abstract = "We present a new scientific document similarity model based on matching "
           "fine-grained aspects of texts."
d=[title + aspire_tok.sep_token + abstract]
inputs = aspire_tok(d, padding=True, truncation=True, return_tensors="pt", max_length=512)
result = aspire_bienc(**inputs)
clsrep = result.last_hidden_state[:, 0, :]
```

However, note that the Hugging Face models don't have a set of additional scalar-mix parameters to compute a learned weighted sum of the representations from different layers of the transformer encoder. These are used in our paper and are important for performance in some datasets. Obtain the model zip files:

- [`aspire-biencoder-biomed-scib-full`](https://ai2-s2-research.s3.us-west-2.amazonaws.com/aspire/aspire-biencoder-biomed-scib-full.zip)
- [`aspire-biencoder-biomed-spec-full`](https://ai2-s2-research.s3.us-west-2.amazonaws.com/aspire/aspire-biencoder-biomed-spec-full.zip)
- [`aspire-biencoder-compsci-spec-full`](https://ai2-s2-research.s3.us-west-2.amazonaws.com/aspire/aspire-biencoder-compsci-spec-full.zip)

```bash
wget -O aspire-biencoder-compsci-spec-full.zip https://ai2-s2-research.s3.us-west-2.amazonaws.com/aspire/aspire-biencoder-compsci-spec-full.zip
unzip aspire-biencoder-compsci-spec-full.zip
```

Now it may be used as:

```python 
import os, json, codecs, torch
from transformers import AutoTokenizer
from examples.ex_aspire_bienc import AspireBiEnc

# Directory where zipped model was downloaded and unzipped.
model_path = './aspire-biencoder-compsci-spec-full'

# Load hyperparameters from disk.
with codecs.open(os.path.join(model_path, 'run_info.json'), 'r') as fp:
    hparams = json.load(fp)
    model_hparams = hparams['all_hparams']

# Initialize the tokenizer and model.
aspire_tok = AutoTokenizer.from_pretrained(model_hparams['base-pt-layer'])
aspire_bienc = AspireBiEnc(model_hparams)

# Load model parameters from disk.
model_fname = os.path.join(model_path, 'model_cur_best.pt')
aspire_bienc.load_state_dict(torch.load(model_fname))

# Encode example input.
title = "Multi-Vector Models with Textual Guidance for Fine-Grained Scientific "
        "Document Similarity"
abstract = "We present a new scientific document similarity model based on matching "
           "fine-grained aspects of texts."
d = [title + aspire_tok.sep_token + abstract]

inputs = aspire_tok(d, padding=True, truncation=True, return_tensors="pt", max_length=512)
clsrep = aspire_bienc.forward(inputs)

```


#### Evaluation Datasets <a name="evaldata"></a>

The paper uses the following evaluation datasets:

- RELISH was created in [Brown et al. 2019](https://academic.oup.com/database/article/doi/10.1093/database/baz085/5608006?login=true). While I wasn't able to access the link in the publication. I was able to obtain a copy of the dataset from: [link](http://pubannotation.org/projects/RELISH-DB). Dataset splits are created in `pre_proc_relish.py`.

- TRECCOVID presents an ad-hoc search dataset. The versions of the dataset used may be accessed here: [query topics](https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml), [relevance annotations](https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt), and the metadata for papers is obtained from the [CORD-19](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html) dataset in the [2021-06-21](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-06-21/metadata.csv) release. The function `get_qbe_pools` in `pre_proc_treccovid.py`, converts the dataset in its original form to the reformulated form, TRECCOVID-RF, used in the paper. Dataset splits are created in `pre_proc_treccovid.py`.

- SciDocs is obtained from: [link](https://github.com/allenai/scidocs). The dataset splits supplied alongside the original dataset are used as is.

- CSFCube is obtained from: [link](https://github.com/iesl/CSFCube). The dataset splits supplied alongside the original dataset are used as is.

Complete evaluation datasets used in the paper can be downloaded here: [`datasets/datasets.md`](https://github.com/allenai/aspire/blob/main/datasets/datasets.md)


### Repository Contents <a name="repocontents"></a>

    ├── bin
    ├── config
    │             └── models_config
    │                 ├── s2orcbiomed
    │                 ├── s2orccompsci
    │                 └── s2orcscidocs
    ├── scripts
    └── src
        ├── evaluation
        │             ├── utils
        │             │             ├── datasets.py
        │             │             ├── metrics.py
        │             │             ├── models.py
        │             │             └── utils.py
        │             └── evaluate.py
        ├── learning
        │             ├── facetid_models
        │             │             ├── disent_models.py
        │             │             ├── pair_distances.py
        │             │             └── sentsim_models.py
        │             ├── main_fsim.py
        │             ├── batchers.py
        │             └── trainer.py
        └── pre_process
            ├── extract_entities.py
            ├── pp_settings.py
            ├── pre_proc_cocits.py
            ├── pre_proc_gorc.py
            ├── pre_proc_relish.py
            ├── pre_proc_scidocs.py
            ├── pre_proc_treccovid.py
            ├── pp_gen_nearest.py
            └── pre_proc_buildreps.py


**The repository is organized broadly as:**

`src/pre_process/`: Scripts to 1) generate gather and filter co-citations data from the [S2ORC](https://github.com/allenai/s2orc) corpus 2) generate training examples with co-citation data 3) pre-process the evaluation datasets into apt formats for use with models 4) extract NER entities from datasets.

`src/learning/`: Classes for implementing models, training, batching data, and a main script to train and save the model.

`src/evaluation/`: Scripts to evaluate model performances on various evaluation datasets.

`config/models_config`: JSON files with hyper-parameters for models in the paper consumed by code in `src/learning/`. Since we evaluate on datasets in the Biomedical (RELISH, TRECCOVID-RF), Computer Science (CSFCube), and mixed domains (SciDocs) we train separate models for these domains, the sub-directories named `s2orcbiomed`, `s2orccompsci`, and `s2orcscidocs` contain config files for the models trained for each domain.

`bin`: Shell scripts to call the scripts in all the `src` sub-directories with appropriate command line arguments.

`scripts`: Miscellaneous glue code.

**The following files are the main entry points into the repository:**

`src/learning/main_fsim.py`: The main script called from `bin/learning/run_main_fsim-ddp.sh` to initialize and train a model. The models consume json config files in `config/models_config/{<domain>}`. A mapping from the model names/classes/configs in the repository to the models reported in the paper is as follows:

`src/evaluation/evaluate.py`: Contain code to generate rankings over the evaluation datasets. Supports trained & downloaded models, and it is simple to add new models to the flow.
For instructions, read the detailed help snippets of argument parser.

`src/pre_process/pre_proc_gorc.py:` Code to gather full text articles from the [S2ORC](https://github.com/allenai/s2orc) corpus, exclude noisy data, and gather co-citations for different domains used in the paper (biomedical papers and computer science papers). This code assumes the 2019-09-28 release of S2ORC. 

`src/pre_process/pre_proc_cocits.py:` Generate training data for the models reported in the paper. Co-citations are used for training sentence level encoder models and whole abstract models, training data for both these model types are generated from functions in this script. These are the `filter_cocitation_sentences` and `filter_cocitation_papers` functions respectively. Functions listed under `write_examples` generate training positive pairs for various models (negatives are generated with in-batch negative sampling).

`src/pre_process/pre_proc_{relish/scidocs/treccovid}.py`: Pre-process the evaluation datasets (RELISH, TRECCOVID, and SciDocs) into a format consumed by trained models and evaluation scripts. CSFCube data format matches the assumed format. Details about each dataset are as follows:

`src/pre_process/extract_entities.py`:  Use PURE's Entity Model () to extract named entities from abstracts. In some experiments, these are added to the abstract as additional sentences as an augmented input, improving results.
For info on how to run this file see src/pre_process/README_NER 



<div style="margin-left: auto;
            margin-right: auto;
            width: 95%">

| Model name in paper         | Config under `config/models_config/{<domain>}`  | Model class in code   |
|-----------------------------|:-------------------------:|:-----------:|
| cosentbert              |        `cosentbert`        |  `facetid_models.sentsim_models.SentBERTWrapper` |
| ICTSentBert              |        `ictsentbert`        |  `facetid_models.sentsim_models.ICTBERTWrapper` |
| SPECTER-CoCite              |        `hparam_opt/cospecter-best`/`hparam_opt/cospecter-specinit-best`        |  `facetid_models.disent_models.MySPECTER`  |
| tsAspire                    |        `hparam_opt/sbalisentbienc-sup-best`        |        `facetid_models.disent_models.WordSentAbsSupAlignBiEnc`   |
| otAspire                    |        `hparam_opt/miswordbienc-otstuni-best`        |      `facetid_models.disent_models.WordSentAlignBiEnc`   |
| ts+otAspire                 |        `hparam_opt/sbalisentbienc-otuni-best`        |        `facetid_models.disent_models.WordSentAbsSupAlignBiEnc`   |
| maxAspire                 |          `hparam_opt/miswordbienc-l2max-best`      |        `facetid_models.disent_models.WordSentAlignBiEnc` |
| absAspire                 |          `hparam_opt/sbalisentbienc-sup-absali-best`      |        `facetid_models.disent_models.WordSentAbsSupAlignBiEnc`   |
| attAspire                 |          `hparam_opt/miswordbienc-cdatt-best`      |        `facetid_models.disent_models.WordSentAlignBiEnc`   |

</div>


### Acknowledgements <a name="acks"></a>

This work relies on: (1) Data from the [Semantic Scholar Open Research Corpus](https://github.com/allenai/s2orc) (S2ORC) and the evaluation datasets RELISH (kindly shared by [Mariana Neves](https://mariananeves.github.io/)), TRECCOVID, SciDocs, and CSFCube linked above. (2) The pre-trained models of [SPECTER](https://github.com/allenai/specter). (3) The software packages: [GeomLoss](https://www.kernel-operations.io/geomloss/index.html) and [sentence-transformers](https://www.sbert.net/).


### Citation <a name="citation"></a>

Please cite the [Aspire paper](https://arxiv.org/pdf/2004.07180.pdf) as:  

```bibtex
@misc{mysore2021aspire,
      title={Multi-Vector Models with Textual Guidance for Fine-Grained Scientific Document Similarity}, 
      author={Sheshera Mysore and Arman Cohan and Tom Hope},
      year={2021},
      eprint={2111.08366},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


### TODOs <a name="todos"></a>

1. Release trained model parameters. (in-progress)
    - Currently released models are _per-domain_ models for computer science and biomedical papers which were used in the paper. The coming months will also see release of domain independent models trained on data across different scientific domains.
2. Release training training data.
    - Co-citation data used to train the above model will also be released, this is co-citation pairs on the order of a few million pairs of papers.
3. Training code usage instructions.
    - This will be released for reproducibility.