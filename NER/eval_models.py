from abc import ABCMeta, abstractmethod
from ex_aspire_consent import AspireConSent, AutoTokenizer, prepare_abstracts
from ex_aspire_consent_multimatch import AllPairMaskedWasserstein
from collections import namedtuple
from src.pre_process.pre_proc_buildreps import BertMLM
from scipy.spatial.distance import euclidean
from NER.aspire_contextual_model import AspireConSenContextual

class EvalModel(metaclass=ABCMeta):
    """
    Abstract class for a model that can be evaluated on document similarity.
    """

    @abstractmethod
    def encode(self, input_data, facet=None):
        """
        :param input_data: Paper data, returned from EvalDataset.get(pid)
        :param facet: Facet we wish to encode
        :return: A tensor which is a vector representation of this paper
        """
        raise NotImplementedError()

    @abstractmethod
    def get_similarity(self, x, y):
        """
        :param x: Vector representation for a query paper
        :param y: Vector representation for a candidate paper
        :return: Number, Similarity between the two papers
        """
        raise NotImplementedError()

def ot_distance(x, y):
    """
    :return: Optimal Transport distance between points x and y
    """
    dist_func = AllPairMaskedWasserstein({})
    rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
    xt = rep_len_tup(embed=x[None, :].permute(0, 2, 1), abs_lens=[len(x)])
    yt = rep_len_tup(embed=y[None, :].permute(0, 2, 1), abs_lens=[len(y)])
    ot_dist = dist_func.compute_distance(query=xt, cand=yt).item()
    return ot_dist

class SpecterBase(EvalModel):
    """
    Base class for SPECTER model. Wrapper to fit EvalModel API.
    """
    def __init__(self):
        self.model = BertMLM('specter')

    def _input_data_to_string(self, input_data):
        joined_abstract = ' '.join(input_data['ABSTRACT'])
        return input_data['TITLE']  + ' [SEP] ' + joined_abstract

    def encode(self, input_data, facet=None):
        input_string = self._input_data_to_string(input_data)
        _, rep = self.model.predict([input_string])
        return rep[0]

    def get_similarity(self, x, y):
        return -euclidean(x, y)


class SpecterSentence(SpecterBase):
    """
    Class for SPECTER model, where entities are added as sentences to the end of the abstract.
    """
    def _input_data_to_string(self, input_data):
        title_and_abstract = super(SpecterSentence, self)._input_data_to_string(input_data)
        ner_list = [item for sublist in input_data['ENTITIES'] for item in sublist]
        return title_and_abstract + ' [SEP] ' + ' [SEP] '.join(ner_list)


class AspireBase(EvalModel):
    """
    Base class for ASPIRE model. Wrapper to fit EvalModel API.
    """
    ASPIRE_MODEL_NAME = 'allenai/aspire-contextualsentence-multim-compsci'

    def __init__(self):
        self.model = AspireConSent(AspireBase.ASPIRE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(AspireBase.ASPIRE_MODEL_NAME)

    def encode(self, input_data, facet=None):

        # forward through net
        bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=[input_data],
                                                                  pt_lm_tokenizer=self.tokenizer)
        _, batch_reps_sent = self.model.forward(bert_batch=bert_batch,
                                                             abs_lens=abs_lens,
                                                             sent_tok_idxs=sent_token_idxs)
        batch_reps_sent = batch_reps_sent[0]

        # filter facet sentences
        if facet is not None:
            labels = ['background' if lab == 'objective_label' else lab[:-len('_label')]
                      for lab in input_data['FACETS']]
            abstract_facet_ids = [i for i, k in enumerate(labels) if facet == k]
            batch_reps_sent = batch_reps_sent[abstract_facet_ids]

        return batch_reps_sent

    def get_similarity(self, x, y):
        return -ot_distance(x, y)


class AspireSentence(AspireBase):
    """
    Class for ASPIRE model, where entities are added as sentences to the end of the abstract.
    """

    def encode(self, input_data, facet=None):

        # append ners to abstract end as new sentences
        abstract = input_data['ABSTRACT']
        ner_list = [item for sublist in input_data['ENTITIES'] for item in sublist]

        input_dict = {
            'TITLE': input_data['TITLE'],
            'ABSTRACT': abstract + ner_list
        }

        # forward through net
        bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=[input_dict],
                                                                  pt_lm_tokenizer=self.tokenizer)
        _, batch_reps_sent = self.model.forward(bert_batch=bert_batch,
                                                abs_lens=abs_lens,
                                                sent_tok_idxs=sent_token_idxs)
        batch_reps_sent = batch_reps_sent[0]

        # filter by facet (sentences and their respective entities)
        if facet is not None:
            labels = ['background' if lab == 'objective_label' else lab[:-len('_label')]
                      for lab in input_data['FACETS']]
            abstract_facet_ids = [i for i, k in enumerate(labels) if facet == k]
            ner_cur_id = len(labels)
            ner_facet_ids = []
            for i, sent_ners in enumerate(input_data['ENTITIES']):
                if i in abstract_facet_ids:
                    ner_facet_ids += list(range(ner_cur_id, ner_cur_id + len(sent_ners)))
                ner_cur_id += len(sent_ners)
            batch_reps_sent = batch_reps_sent[abstract_facet_ids + ner_facet_ids]

        return batch_reps_sent

class AspireContextual(EvalModel):
    """
    Class for ASPIRE model, where each entity is represented by the average token embeddings
    for all tokens that are within this entitie's span inside the sentence it appears in.
    Uses aspire_contextual.AspireContextualModel instead of the regular AspireConSent
    """
    def __init__(self):
        self.model = AspireConSenContextual(AspireBase.ASPIRE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(AspireBase.ASPIRE_MODEL_NAME)

    def encode(self, input_data, facet=None):

        # prepare abstracts the normal way
        bert_batch, abs_lens, sent_token_idxs = prepare_abstracts(batch_abs=[input_data],
                                                                  pt_lm_tokenizer=self.tokenizer)
        # get token idxs of ners
        ner_token_idxs = self._get_ner_token_idxs(input_data, sent_token_idxs[0])

        _, batch_reps_sent = self.model.forward(bert_batch=bert_batch,
                                                abs_lens=abs_lens,
                                                sent_tok_idxs=sent_token_idxs,
                                                ner_tok_idxs=[ner_token_idxs])

        batch_reps_sent = batch_reps_sent[0]
        return batch_reps_sent

    def _get_ner_token_idxs(self, input_data, sent_token_idxs):
        sentences = input_data['ABSTRACT']
        sentence_ners = input_data['ENTITIES']
        ner_token_idxs = []
        for ners, sentence, token_idxs in zip(sentence_ners, sentences, sent_token_idxs):
            tokens = self.tokenizer.tokenize(sentence)
            for ner in ners:
                # find the tokens in the sentence that correspond to this entity
                ner_range = self.find_sublist_range(tokens, self.tokenizer.tokenize(ner))
                if ner_range is not None and len(ner_range) > 0:
                    # get all idxs that happen before hitting the max number of tokens
                    ner_idxs = [token_idxs[ner_i] for ner_i in ner_range if ner_i < len(token_idxs)]
                    # take only ners that are completely inside the tokenization
                    if len(ner_range) == len(ner_idxs):
                        ner_token_idxs.append(ner_idxs)
        return ner_token_idxs

    @staticmethod
    def find_sublist_range(suplist, sublist):
        for i in range(len(suplist)):
            subrange = []
            j = 0
            while (i + j) < len(suplist) and j < len(sublist) and suplist[i + j] == sublist[j]:
                subrange.append(i + j)
                j += 1
            if j == len(sublist):
                return subrange
        return None

    def get_similarity(self, x, y):
        return -ot_distance(x, y)


def get_model(model_name) -> EvalModel:
    """
    Factory for eval models.
    :param model_name: str
    :return: Requested Model Object
    """
    if model_name == 'specter_base':
        return SpecterBase()
    if model_name == 'specter_sentence':
        return SpecterSentence()
    elif model_name == 'aspire_base':
        return AspireBase()
    elif model_name == 'aspire_sentence':
        return AspireSentence()
    elif model_name == 'aspire_contextual':
        return AspireContextual()
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")