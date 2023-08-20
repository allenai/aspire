#### Extract Entities

`extract_entities.py` uses the "PURE" NER Entity Model (https://github.com/princeton-nlp/PURE)
to extract named entities from papers in a dataset. This allows models to consider the named entities present in a paper absrtact when creating its encoding. 

An interesting use of these entities is as a form of augmented input for existing models, which improve performance without requiring fine-tuning. This can be done, for example, by appending the entities to the end of the abstract, as if they were additional sentences, then feeding the result to a model such as SPECTER or otAspire.


#### Usage Instructions

1. Installing Environment
   1. Clone PURE repo (https://github.com/princeton-nlp/PURE)
   2. Install requirements: ```pip install -r PURE/requirements.txt``` 
   3. Download SciBERT Entity Model

```shell
cd PURE; mkdir scierc_models; cd scierc_models
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip
unzip ent-scib-ctx0.zip; rm -f ent-scib-ctx0.zip
scierc_ent_model=scierc_models/ent-scib-ctx0
```


2. Running script: run `extract_entities.py` with these arguments:

```
--dataset_dir /path/to/dataset_dir/
--dataset_name {name of dataset}
--entity_model_dir /path/to/scibert_model
```

Results are written to {dataset_dir}/{dataset_name}-ner.jsonl

3. Implementation Example in model

In `src.evaluations.utils.models.py` See `AspireNER` as an example of implementation.
Here we take an existing model, override its encode method by appending entities to the end of abstracts, then calling the super() encode method. 

```python
class AspireNER(AspireModel):
    def encode(self, batch_papers: List[Dict]):
        input_batch_with_ner = self._append_entities(batch_papers)
        return super(AspireNER, self).encode(input_batch_with_ner)

    def _append_entities(self, batch_papers):
        # append ners to abstract end as new sentences
        input_batch_with_ner = []
        for sample in batch_papers:
            ner_list = [item for sublist in sample['ENTITIES'] for item in sublist]
            input_sample = {'TITLE': sample['TITLE'],
                            'ABSTRACT': sample['ABSTRACT'] + ner_list
                            }
            input_batch_with_ner.append(input_sample)
        return input_batch_with_ner
```