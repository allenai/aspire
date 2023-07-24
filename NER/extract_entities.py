from PURE.shared.const import task_ner_labels, get_labelmap
from PURE.entity.models import EntityModel
from eval_datasets import load_dataset
import codecs
import json
import os
from scipy.special import softmax
from collections import namedtuple

### constants ###
TASK_NAME = 'scierc'
NUM_LABELS = len(task_ner_labels[TASK_NAME]) + 1
LABEL2ID, ID2LABEL = get_labelmap(task_ner_labels[TASK_NAME])
MAX_SPAN_LENGTH = 8

def load_entity_model(bert_model_dir):
    Args = namedtuple("Args", "model bert_model_dir use_albert max_span_length")
    args = Args(model="allenai/scibert_scivocab_uncased",
         bert_model_dir=bert_model_dir,
         use_albert=False,
         max_span_length=MAX_SPAN_LENGTH)

    return EntityModel(args, num_ner_labels=NUM_LABELS)

def prepare_sentence(text: str, max_span_length: int):
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

def extract_ner_spans(model: EntityModel, sentences):
    """
    :param model: entity Model
    :param sentences: List[str]]
    :return: List[List[(entity as untokenized string, confidence score, label)]]
    """
    tokens = [model.tokenizer.tokenize(s) for s in sentences]
    inputs = [prepare_sentence(text = t, max_span_length=MAX_SPAN_LENGTH) for t in tokens]
    predictions = predict_batch(model, inputs)
    entities = []
    for i, ners in enumerate(predictions):
        sentence_entities = list()
        for ner in ners:
            untokenized_entity = untokenize(tokens[i][ner[0][0]:ner[0][1] + 1])
            entity_prob = ner[1]
            entity_label = ner[2]
            sentence_entities.append((untokenized_entity, str(entity_prob.round(5)), str(entity_label)))
        entities.append(sentence_entities)
    return entities

def main(data_path, bert_model_dir):
    """
    :param data_path: Data path where CSFCube is located
    :param bert_model_dir: Path to Entity model's bert model
    :return:
    """

    # load entity model and dataset
    model = load_entity_model(bert_model_dir)
    dataset = load_dataset(os.path.join(data_path, 'abstracts-csfcube-preds.jsonl'))

    # find entities for each document
    entities = dict()
    for i, (doc_id, doc) in enumerate(dataset.items()):
        if i % 10 == 0:
            print(f'{i} / {len(dataset)}')
        doc_entities = extract_ner_spans(model, doc['ABSTRACT'])
        entities[doc_id] = doc_entities

    # save results
    output_filename = os.path.join(data_path, 'abstracts-csfcube-preds-ner-probs-labels.jsonl')
    with codecs.open(output_filename, 'w', 'utf-8') as fp:
        json.dump(entities, fp)

if __name__ == '__main__':
    CSFCUBE_DATA_PATH = '/homes/roik/PycharmProjects/aspire/datasets_raw/s2orccompsci/csfcube/'
    BERT_MODEL_DIR = '/homes/roik/PycharmProjects/aspire/PURE/scierc_models/ent-scib-ctx0'
    main(data_path=CSFCUBE_DATA_PATH,
         bert_model_dir=BERT_MODEL_DIR)
