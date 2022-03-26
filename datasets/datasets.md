#### Evaluation datasets

The datasets used for evaluation are released on figshare.com since some of the files are larger than Githubs file size limit of 100MB. Each of the datasets is released in the same format and consists of 4 files:

`abstracts-*.jsonl`: jsonl file containing the paper-id, abstracts, and titles for the queries and candidates which are part of the dataset.

`*-queries-release.csv`: Metadata associated with every query.

`test-pid2anns-*.json`: JSON file with the query paper-id, candidate paper-ids, and relevance annotations for every query paper in the dataset. Use these files in conjunction with `abstracts-*.jsonl` to generate files for use in model evaluation.

`*-evaluation_splits.json`: Paper-ids for the splits to use in reporting evaluation numbers. aspire/src/evaluation/ranking_eval.py included in the github repo accompanying this dataset implements the evaluation protocol and computes evaluation metrics. Please see the paper for descriptions of the experimental protocol we recommend to report evaluation metrics.

CSFCube: https://github.com/iesl/CSFCube

RELISH: https://figshare.com/articles/dataset/RELISH-Aspire/19425506

TRECCOVID-RF: https://figshare.com/articles/dataset/TRECCOVID-RF-Aspire/19425515 

SciDocs: https://figshare.com/articles/dataset/SciDocs-Aspire/19425533
