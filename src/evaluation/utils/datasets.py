import os
import codecs
import json
import pandas as pd
from typing import Dict, Union

class EvalDataset:
    """
    Class for datasets used in evaluation
    """

    def __init__(self, name: str, root_path: str):
        """
        :param name: Name of dataset
        :param root_path: Path where dataset files sit (e.g. abstracts-{name}}.json)
        """
        self.name = name
        self.root_path = root_path
        self.dataset = self._load_dataset(fname=os.path.join(root_path, f'abstracts-{self.name}.jsonl'))
        # load entity data, if exists
        self.ner_data = self._load_ners()

    @staticmethod
    def _load_dataset(fname: str) -> Dict:
        """
        :param fname: File with dataset's papers.
        :return: dictionary of {pid: paper_info},
        with paper_info being a dict with keys ABSTRACT and TITLE.
        If data is CSFcube, also includes FACETS.
        If NER extraction was performed, also includes ENTITIES.
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
                if 'pred_labels' in data:
                    ret_dict['FACETS'] = data['pred_labels']
                dataset[pid] = ret_dict
        return dataset

    def _load_ners(self) -> Union[None, Dict]:
        """
        Attempts to load dictionary with NER information on papers, as dictionary.
        If not found, returns None.
        """
        fname = os.path.join(self.root_path, f'{self.name}-ner.jsonl')
        if os.path.exists(fname):
            with codecs.open(fname, 'r', 'utf-8') as ner_f:
                return json.load(ner_f)
        else:
            return None

    def get(self, pid: str) -> Dict:
        """
        :param pid: paper id
        :return: relevant information for the paper: title, abstract, and if available also facets and entities.
        """
        data = self.dataset[pid]
        if self.ner_data is not None:
            return {**data, **{'ENTITIES': self.ner_data[pid]}}
        else:
            return data

    def get_test_pool(self, facet=None):
        """
        Load the test pool of queries and cadidates.
        If performing faceted search, the test pool depends on the facet.
        :param facet: If cfscube, one of (result, method, background). Else, None.
        :return: test pool
        """
        if facet is not None:
            fname = os.path.join(self.root_path, f"test-pid2anns-{self.name}-{facet}.json")
        else:
            fname = os.path.join(self.root_path, f"test-pid2anns-{self.name}.json")
        with codecs.open(fname, 'r', 'utf-8') as fp:
            test_pool = json.load(fp)
        return test_pool

    def get_gold_test_data(self, facet=None):
        """
        Load the relevancies gold data for the dataset.
        :param facet: If cfscube, one of (result, method, background). Else, None.
        :return: gold data
        """
        # format is {query_id: {candidate_id: relevance_score}}
        gold_fname = f'test-pid2anns-{self.name}-{facet}.json' if facet is not None else f'test-pid2anns-{self.name}.json'
        with codecs.open(os.path.join(self.root_path, gold_fname), 'r', 'utf-8') as fp:
            gold_test_data = {k: dict(zip(v['cands'], v['relevance_adju'])) for k, v in json.load(fp).items()}
        return gold_test_data

    def get_query_metadata(self):
        """
        Load file with metadata on queries in test pool.
        :return:
        """
        metadata_fname = os.path.join(self.root_path, f'{self.name}-queries-release.csv')
        query_metadata = pd.read_csv(metadata_fname, index_col='pid')
        query_metadata.index = query_metadata.index.astype(str)
        return query_metadata

    def get_test_dev_split(self):
        """
        Load file that determines dev/test split for dataset.
        :return:
        """
        if self.name == 'csfcube':
            # entire dataset is test set
            return None
        else:
            with codecs.open(os.path.join(self.root_path, f'{self.name}-evaluation_splits.json'), 'r', 'utf-8') as fp:
                return json.load(fp)

    def get_threshold_grade(self):
        """
        Determines threshold grade of relevancy score.
        Relevancies are scores in range of 0 to 3. If a score is at least as high as this threshold,
        A candidate paper is said to be relevant to the query paper.
        :return:
        """
        return 1 if self.name in {'treccovid', 'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'} else 2

    def __iter__(self):
        return self.dataset.items()
