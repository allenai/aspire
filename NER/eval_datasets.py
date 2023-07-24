import codecs
import json
import os

class CSFCube:

    def __init__(self, root_path, entity_threshold=None, entity_blacklist_label=None):
        """
        :param root_path: path where dataset sits
        :param entity_threshold: float in [0,1].
        Only entities with confidence score above this value will be taken.
        :param entity_blacklist_label: List[int]
        Only entities with labels not in this blacklist will be taken.
        See PURE. See extract_entities.ID2LABEL
        """
        self.root_path = root_path
        self.dataset = load_dataset(os.path.join(root_path, 'abstracts-csfcube-preds.jsonl'))
        self.ners = self._read_ners(os.path.join(root_path, 'abstracts-csfcube-preds-ner-probs-labels.jsonl'))
        self.entity_threshold = 0 if entity_threshold is None else entity_threshold
        self.entity_blacklist_label = [] if entity_blacklist_label is None else entity_blacklist_label

    def get(self, pid):
        entities = list()
        for sent_entities in self.ners[pid]:
            entities.append([entity for entity, prob, lab in sent_entities
                             if eval(prob) > self.entity_threshold
                             and eval(lab) not in self.entity_blacklist_label])
        return {**self.dataset[pid],**{'ENTITIES': entities}}

    @staticmethod
    def _read_dataset(fname):
        dataset = dict()
        with codecs.open(fname, 'r', 'utf-8') as f:
            for jsonline in f:
                data = json.loads(jsonline.strip())
                pid = data['paper_id']
                ret_dict = {
                    'TITLE': data['title'],
                    'ABSTRACT': data['abstract'],
                    'FACETS': data['pred_labels']
                }
                dataset[pid] = ret_dict
        return dataset

    @staticmethod
    def _read_ners(fname):
        with codecs.open(fname, 'r', 'utf-8') as ner_f:
            return json.load(ner_f)

    def get_test_pool(self, facet):
        fname = os.path.join(self.root_path, f"test-pid2anns-csfcube-{facet}.json")
        with codecs.open(fname, 'r', 'utf-8') as fp:
            test_pool = json.load(fp)
        return test_pool

def load_dataset(fname):
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
                'FACETS': data['pred_labels']
            }
            dataset[pid] = ret_dict
    return dataset