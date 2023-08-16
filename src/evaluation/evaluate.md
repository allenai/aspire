#### Evaluations

`evaluate.py` is the entry point for evaluation of paper similarity models on datasets.

Given a dataset and a model, the script can:

1. Create (and cache) the model's encodings on the dataset
2. Calculate similarity score for the encoding on the dataset test pool
3. Compute evaluation metrics that test the model's performance.

For more info on running the code, read the help snippets of the argument parser.

`utils/models.py` contains the base class `SimilarityModel`, which is used in this script, as well as implementations of it for trained models, the Aspire model uploaded to allenai, and more. 

Adding your own model to this script is very simple:
1. Create a class that inherits `SimilarityModel`
2. Implement `encode(batch_papers)` which returns batch encodings,
3. Implement `get_similarity(x, y)` which, given 2 encodings, returns a similarity score (higher == closer).
4. Add the model to the factory method `get_model(model_name)`


`utils/datasets.py` contains an implementation of `EvalDataset`, the class used to load and retrieve data from the dataset files. Make sure all files are in place before running the script; see `datasets.md`. Note: if you are intending on using named entities for creating the encodings, run `pre_process/extract_entities.py` first, as explained in `pre_process/extract_entities.md`,  

`utils/metrics.py` contains implementations for metric calculations used in the paper.
