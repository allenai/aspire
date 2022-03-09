### Aspire
Repository accompanying paper for modeling fine grained similarity between documents: 

**Title**: "Multi-Vector Models with Textual Guidance for Fine-Grained Scientific Document Similarity"

**Authors**: Sheshera Mysore, Arman Cohan, Tom Hope

**Abstract**: We present Aspire, a new scientific document similarity model based on matching fine-grained aspects. Our model is trained using co-citation contexts that describe related paper aspects as a novel form of textual supervision. We use multi-vector document representations, recently explored in settings with short query texts but under-explored in the challenging document-document setting. We present a fast method that involves matching only single sentence pairs, and a method that makes sparse multiple matches with optimal transport. Our model improves performance on document similarity tasks across four datasets. Moreover, our fast single-match method achieves competitive results, opening up the possibility of applying fine-grained document similarity models to large-scale scientific corpora. 

The pre-print can be accessed here: https://arxiv.org/abs/2111.08366


### Repository Contents

    ├── bin
    ├── config
    │             └── models_config
    │                 ├── s2orcbiomed
    │                 ├── s2orccompsci
    │                 └── s2orcscidocs
    ├── scripts
    └── src
        ├── evaluation
        │             └── ranking_eval.py
        ├── learning
        │             ├── facetid_models
        │             │             ├── disent_models.py
        │             │             ├── pair_distances.py
        │             │             └── sentsim_models.py
        │             ├── main_fsim.py
        │             ├── batchers.py
        │             └── trainer.py
        └── pre_process
            ├── pp_gen_nearest.py
            ├── pp_settings.py
            ├── pre_proc_buildreps.py
            ├── pre_proc_cocits.py
            ├── pre_proc_gorc.py
            ├── pre_proc_relish.py
            ├── pre_proc_scidocs.py
            └── pre_proc_treccovid.py


**The repository is organized broadly as:**

`src/pre_process/`: Scripts to 1) generate gather and filter co-citations data from the [S2ORC](https://github.com/allenai/s2orc) corpus 2) generate training examples with co-citation data 3) pre-process the evaluation datasets into apt formats for use with models 4) code to generate rankings over evaluation datasets given trained models.

`src/learning/`: Classes for implementing models, training, batching data, and a main script to train and save the model.

`src/evaluation/`: Scripts to evaluate rankings for various evaluation datasets.

`config/models_config`: JSON files with hyper-parameters for models in the paper consumed by code in `src/learning/`. Since we evaluate on datasets in the Biomedical (RELISH, TRECCOVID-RF), Computer Science (CSFCube), and mixed domains (SciDocs) we train separate models for these domains, the sub-directories named `s2orcbiomed`, `s2orccompsci`, and `s2orcscidocs` contain config files for the models trained for each domain.

`bin`: Shell scripts to call the scripts in all the `src` sub-directories with appropriate command line arguments.

`scripts`: Miscellaneous glue code.

**The following files are the main entry points into the repository:**

`src/pre_process/pre_proc_gorc.py:` Code to gather full text articles from the [S2ORC](https://github.com/allenai/s2orc) corpus, exclude noisy data, and gather co-citations for different domains used in the paper (biomedical papers and computer science papers). This code assumes the 2019-09-28 release of S2ORC. 

`src/pre_process/pre_proc_cocits.py:` Generate training data for the models reported in the paper. Co-citations are used for training sentence level encoder models and whole abstract models, training data for both these model types are generated from functions in this script. These are the `filter_cocitation_sentences` and `filter_cocitation_papers` functions respectively. Functions listed under `write_examples` generate training positive pairs for various models (negatives are generated with in-batch negative sampling).

`src/pre_process/pre_proc_{relish/scidocs/treccovid}.py`: Pre-process the evaluation datasets (RELISH, TRECCOVID, and SciDocs) into a format consumed by trained models and evaluation scripts. CSFCube data format matches the assumed format. Details about each dataset are as follows:

- RELISH was created in [Brown et al. 2019](https://academic.oup.com/database/article/doi/10.1093/database/baz085/5608006?login=true). While I wasn't able to access the link in the publication. I was able to obtain a copy of the dataset from: [link](http://pubannotation.org/projects/RELISH-DB). Dataset splits are created in `pre_proc_relish.py`.

- TRECCOVID presents an ad-hoc search dataset. The versions of the dataset used may be accessed here: [query topics](https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml), [relevance annotations](https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt), and the metadata for papers is obtained from the [CORD-19](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html) dataset in the [2021-06-21](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-06-21/metadata.csv) release. The function `get_qbe_pools` in `pre_proc_treccovid.py`, converts the dataset in its original form to the reformulated form, TRECCOVID-RF, used in the paper. Dataset splits are created in `pre_proc_treccovid.py`.

- SciDocs is obtained from: [link](https://github.com/allenai/scidocs). The dataset splits supplied alongside the original dataset are used as is.

- CSFCube is obtained from: [link](https://github.com/iesl/CSFCube). The dataset splits supplied alongside the original dataset are used as is.

`src/pre_process/{pre_proc_buildreps.py/pp_gen_nearest.py}`: Contain code to generate rankings over the evaluation datasets for consumption with a trained model. Most of the results reported in the paper are generated with the `CachingTrainedScoringModel` class in `pp_gen_nearest.py`.

`src/evaluation/ranking_eval.py`: Script for generating eval metrics.

`src/learning/main_fsim.py`: The main script called from `bin/learning/run_main_fsim-ddp.sh` to initialize and train a model. The models consume json config files in `config/models_config/{<domain>}`. A mapping from the model names/classes/configs in the repository to the models reported in the paper is as follows:

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


### Artifacts 

#### Models

Models described in the paper will be released as Hugging Face models.

`cosentbert`: [`allenai/aspire-sentence-embedder`](https://huggingface.co/allenai/aspire-sentence-embedder)

`SPECTER-CoCite`: TODO

`tsAspire`: TODO

`otAspire`: TODO


### Acknowledgements
This work relies on: (1) Data from the [Semantic Scholar Open Research Corpus](https://github.com/allenai/s2orc) (S2ORC) and the evaluation datasets RELISH (kindly shared by [Mariana Neves](https://mariananeves.github.io/)), TRECCOVID, SciDocs, and CSFCube linked above. (2) The pre-trained models of [SPECTER](https://github.com/allenai/specter). (3) The software packages: [GeomLoss](https://www.kernel-operations.io/geomloss/index.html) and [sentence-transformers](https://www.sbert.net/).


### Citation

Please cite the [ASPIRE paper](https://arxiv.org/pdf/2004.07180.pdf) as:  

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


### Todos

1. Training code usage instructions.
2. Release trained model parameters. (in-progress)
3. Release training and evaluation data.