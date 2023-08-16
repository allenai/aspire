from PURE.shared.const import task_ner_labels, get_labelmap
from PURE.entity.models import EntityModel
import codecs
import json
import os
from scipy.special import softmax
from collections import namedtuple
from tqdm import tqdm
import argparse
from typing import List

### constants ###
TASK_NAME = 'scierc'
NUM_LABELS = len(task_ner_labels[TASK_NAME]) + 1
LABEL2ID, ID2LABEL = get_labelmap(task_ner_labels[TASK_NAME])
MAX_SPAN_LENGTH = 8

def load_entity_model(entity_model_dir: str):
    """
    :param entity_model_dir: path to dir where PURE's entity berty mode was downloaded.
    e.g. /aspire/PURE/scierc_models/ent-scib-ctx0
    :return: loaded entity model
    """
    Args = namedtuple("Args", "model bert_model_dir use_albert max_span_length")
    args = Args(model="allenai/scibert_scivocab_uncased",
                bert_model_dir=entity_model_dir,
                use_albert=False,
                max_span_length=MAX_SPAN_LENGTH)

    return EntityModel(args, num_ner_labels=NUM_LABELS)

def load_dataset(fname: str):
    """
    :param fname: filename for csfcube raw data
    :return: dict of {pid: data}
    """
    dataset = dict()
    with codecs.open(fname, 'r', 'utf-8') as f:
        for jsonline in f:
            data = json.loads(jsonline.strip())
            pid = data['paper_id']
            ret_dict = {
                'TITLE': data['title'],
                'ABSTRACT': data['abstract'],
            }
            dataset[pid] = ret_dict
    return dataset

def prepare_sentence(text: str, max_span_length: int):
    """
    Pre process input data for entity model
    :param text: A single sentence
    :param max_span_length: Maximum number of words we expect in a span
    :return: input data for entity model for this sentence
    """
    sample = {
        'tokens': text,
        'sent_length': len(text),
        'sent_start': 0,
        'send_end': len(text),
        'sent_start_in_doc': 0,
    }
    spans = list()
    for i in range(len(text)):
        for j in range(i, min(len(text), i + max_span_length)):
            spans.append((i, j, j - i + 1))
    span_labels = [0 for _ in range(len(spans))]
    sample['spans'] = spans
    sample['spans_label'] = span_labels
    return sample

def predict_batch(model: EntityModel, batch):
    """
    Runs an input batch through the entity model
    :param model: entity model
    :param batch: input batch
    :return: output
    """
    output_dict = model.run_batch(batch, training=False)
    batch_pred = output_dict['pred_ner']
    batch_probs = output_dict['ner_probs']
    batch_ans = []
    for i, sample in enumerate(batch):
        non_zero_spans = list()
        for span, label, probs in zip(sample['spans'], batch_pred[i], batch_probs[i]):
            if label != 0:
                max_prob = softmax(probs).max(axis=-1)
                non_zero_spans.append((span, max_prob, label))
        batch_ans.append(non_zero_spans)
    return batch_ans

#  https://stackoverflow.com/questions/66232938/how-to-untokenize-bert-tokens
def untokenize(tokens):
    pretok_sent = ""
    for tok in tokens:
        if tok.startswith("##"):
            pretok_sent += tok[2:]
        else:
            pretok_sent += " " + tok
    pretok_sent = pretok_sent[1:]
    return pretok_sent

def extract_ner_spans(sentences: List[str], model: EntityModel) -> List[List[str]]:
    """
    Extracts NER entities from sentences using the entity model provided.
    For each sentence, returns a list of all entities extracted from it,
    given as plain string.
    Entities may appear different in the sentence and as an entity,
    because they are tokenized then untokenized.
    :param model: PURE entity model
    :param sentences: List[str]]
    :return: List of entities for each sentence
    """
    # tokenize and preprocess sentences
    tokens = [model.tokenizer.tokenize(s) for s in sentences]
    inputs = [prepare_sentence(text = t, max_span_length=MAX_SPAN_LENGTH) for t in tokens]

    # run through entity model
    predictions = predict_batch(model, inputs)

    # collect to output shape
    entities = []
    for i, ners in enumerate(predictions):
        sentence_entities = list()
        for ner in ners:
            untokenized_entity = untokenize(tokens[i][ner[0][0]:ner[0][1] + 1])
            sentence_entities.append(untokenized_entity)
        entities.append(sentence_entities)
    return entities

def main(dataset_dir,
         dataset_name,
         entity_model_dir):
    """
    :param dataset_dir: Data path where CSFCube is located
    :param bert_model_dir: Path to Entity model's bert model
    :return:
    """

    # load entity model and dataset
    print("Loading model and dataset")
    model = load_entity_model(entity_model_dir)
    dataset = load_dataset(os.path.join(dataset_dir, f'abstracts-{dataset_name}.jsonl'))

    # find entities for each paper
    print("Extracting entities from abstracts")
    entities = dict()
    for (doc_id, doc) in tqdm(list(dataset.items())[:10]):
        doc_entities = extract_ner_spans(doc['ABSTRACT'], model)
        entities[doc_id] = doc_entities

    # save results
    output_filename = os.path.join(dataset_dir, f'{dataset_name}-ner2.jsonl')
    print(f"Writing output to: {output_filename}")
    with codecs.open(output_filename, 'w', 'utf-8') as fp:
        json.dump(entities, fp)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, help='Name of dataset to extract entities on')
    parser.add_argument('--dataset_dir', required=True, help='Dataset dir. abstracts-{dataset_name}.jsonl should be inside')
    parser.add_argument('--entity_model_dir', required=True, help="Path where PURE Entity model was downloaded to")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(dataset_dir=args.dataset_dir,
         dataset_name=args.dataset_name,
         entity_model_dir=args.entity_model_dir)