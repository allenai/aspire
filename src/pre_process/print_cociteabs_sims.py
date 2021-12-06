"""
Read in abstracts and co-citation sentences and print similarities of
abstract sentences to co-citation sentences.
This is a quick script to examine training data placed in pre-process
so the imports work.
"""
import os
import codecs, json
import numpy as np
import scipy
from scipy import special, spatial
import torch
from sklearn import metrics as skmetrics
from sentence_transformers import SentenceTransformer, models
import ot
from . pre_proc_buildreps import TrainedModel


def print_cocite_contextualsentsim(trained_model_path, examples_path):
    """
    Go over the co-cited abstracts and print out their pairwise similarity
    with contextual sentence representations in a bid to understand how
    well multi-instance alignment would work.
    """
    model = TrainedModel(model_name='conswordbienc', trained_model_path=trained_model_path, model_version='init')

    in_triples = codecs.open(os.path.join(examples_path, 'dev-cocitabs.jsonl'), 'r', 'utf-8')
    out_file = codecs.open(os.path.join(examples_path, 'dev-abs_cc-conswordbienc-sims.txt'), 'w', 'utf-8')
    out_file.write(f"Models:\nAbs model: {trained_model_path}\n")
    written_count = 0
    for jsonl in in_triples:
        # Encode sentences for triple.
        ex_dict = json.loads(jsonl.strip())
        qabs = ex_dict['query']['ABSTRACT']
        pos_abs = ex_dict['pos_context']['ABSTRACT']
        _, sent_reps = model.predict([ex_dict['query'], ex_dict['pos_context']])
        qabs_reps = sent_reps[0]
        posabs_reps = sent_reps[1]
        q2pos_abs_sims = np.matmul(qabs_reps, posabs_reps.T)
        q2pos_softmax = special.softmax(q2pos_abs_sims.flatten()/np.sqrt(768))
        q2pos_softmax = q2pos_softmax.reshape(q2pos_abs_sims.shape)
        q2pos_abs_sims = np.array2string(q2pos_softmax, precision=2)
        # Print abstracts and similarities.
        qabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(qabs)])
        out_file.write(f'Query abstract:\n{ex_dict["query"]["TITLE"]}\n{qabs_str}\n')
        out_file.write(q2pos_abs_sims+'\n')
        pabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(pos_abs)])
        out_file.write(f'Positive abstract:\n{ex_dict["pos_context"]["TITLE"]}\n{pabs_str}\n')
        out_file.write('==================================\n')
        written_count += 1
        if written_count > 1000:
            break
    print(f'Wrote: {out_file.name}')
    out_file.close()


def print_cocite_contextualsentsim_contextsent(trained_abs_model_path, trained_sentmodel_path, examples_path):
    """
    Go over the co-cited abstracts and print out their pairwise similarity
    with contextual sentence representations to the sentence context in which
    they occur to understand if the context sentences provide reasonable
    supervision.
    """
    # Init the sentence model.
    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                              max_seq_length=512)
    # Loading local model: https://github.com/huggingface/transformers/issues/2422#issuecomment-571496558
    trained_model_fname = os.path.join(trained_sentmodel_path, 'sent_encoder_cur_best.pt')
    word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Init the abstract model.
    model = TrainedModel(model_name='conswordbienc', trained_model_path=trained_abs_model_path, model_version='init')
    in_triples = codecs.open(os.path.join(examples_path, 'dev-cocitabs.jsonl'), 'r', 'utf-8')
    out_file = codecs.open(os.path.join(examples_path, 'dev-abs2context-conswordbienc-sims.txt'), 'w', 'utf-8')
    out_file.write(f"Models:\nAbs model: {trained_abs_model_path}\nSent model: {trained_abs_model_path}\n")
    written_count = 0
    for jsonl in in_triples:
        # Encode sentences for triple.
        ex_dict = json.loads(jsonl.strip())
        qabs = ex_dict['query']['ABSTRACT']
        pos_abs = ex_dict['pos_context']['ABSTRACT']
        _, sent_reps = model.predict([ex_dict['query'], ex_dict['pos_context']])
        qabs_reps = sent_reps[0]
        posabs_reps = sent_reps[1]
        citing_contexts = ex_dict['citing_contexts']
        _, sent_reps = model.predict([{'TITLE': '', 'ABSTRACT': citing_contexts}])
        cc_reps = sent_reps[0]
        # Get pairwise sims.
        cc2query_abs_sims = np.matmul(qabs_reps, cc_reps.T)
        cc2query_idxs = np.unravel_index(cc2query_abs_sims.argmax(), cc2query_abs_sims.shape)
        cc2query_abs_sims = np.array2string(cc2query_abs_sims, precision=2)
        cc2pos_abs_sims = np.matmul(posabs_reps, cc_reps.T)
        cc2pos_idxs = np.unravel_index(cc2pos_abs_sims.argmax(), cc2pos_abs_sims.shape)
        cc2pos_abs_sims = np.array2string(cc2pos_abs_sims, precision=2)
        q2pos_abs_sims = np.matmul(qabs_reps, posabs_reps.T)
        q2pos_idxs = np.unravel_index(q2pos_abs_sims.argmax(), q2pos_abs_sims.shape)
        # Print abstracts and similarities.
        qabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(qabs)])
        out_file.write(f'Query abstract:\n{ex_dict["query"]["TITLE"]}\n{qabs_str}\n')
        out_file.write(cc2query_abs_sims+'\n')
        contextalign_diff = True if (cc2query_idxs[0], cc2pos_idxs[0]) != (q2pos_idxs[0], q2pos_idxs[1]) else False
        out_file.write(f'cc2q: {cc2query_idxs}; cc2p: {cc2pos_idxs}; q2p: {q2pos_idxs}\n')
        out_file.write(f'contextalign_diff: {contextalign_diff}\n')
        out_file.write('Citing contexts:\n{:}\n'.format('\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(citing_contexts)])))
        out_file.write(cc2pos_abs_sims+'\n')
        pabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(pos_abs)])
        out_file.write(f'Positive abstract:\n{ex_dict["pos_context"]["TITLE"]}\n{pabs_str}\n')
        out_file.write('==================================\n')
        written_count += 1
        if written_count > 1000:
            break
    print(f'Wrote: {out_file.name}')
    out_file.close()


def print_context_abs_sims(trained_model_path, examples_path):
    """
    - Read a triple example,
    - Encode the sentences of the abstract.
    - Encode the citation contexts.
    - Compute pairwise dot products between citation context and the encoded sentences.
    - Do the abstract with a abstract encoder and the sentence with CoSentBert.
    """
    # Init the model.
    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                              max_seq_length=512)
    # Loading local model: https://github.com/huggingface/transformers/issues/2422#issuecomment-571496558
    trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
    word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    in_triples = codecs.open(os.path.join(examples_path, 'dev-cocitabs.jsonl'), 'r', 'utf-8')
    out_file = codecs.open(os.path.join(examples_path, 'dev-abs_cc-sims.txt'), 'w', 'utf-8')
    out_file.write(f'Sentence model: {trained_model_path}\n')
    written_count = 0
    for jsonl in in_triples:
        # Encode sentences for triple.
        ex_dict = json.loads(jsonl.strip())
        qabs = ex_dict['query']['ABSTRACT']
        pos_abs = ex_dict['pos_context']['ABSTRACT']
        citing_contexts = ex_dict['citing_contexts']
        reps = sentbert_model.encode(qabs+pos_abs+citing_contexts)
        qabs_reps = reps[:len(qabs)]
        posabs_reps = reps[len(qabs): len(qabs)+len(pos_abs)]
        cc_reps = reps[len(qabs)+len(pos_abs):]
        # Get pairwise sims.
        cc2query_abs_sims = np.matmul(qabs_reps, cc_reps.T)
        cc2query_idxs = np.unravel_index(cc2query_abs_sims.argmax(), cc2query_abs_sims.shape)
        cc2query_abs_sims = np.array2string(cc2query_abs_sims, precision=2)
        cc2pos_abs_sims = np.matmul(posabs_reps, cc_reps.T)
        cc2pos_idxs = np.unravel_index(cc2pos_abs_sims.argmax(), cc2pos_abs_sims.shape)
        cc2pos_abs_sims = np.array2string(cc2pos_abs_sims, precision=2)
        q2pos_abs_sims = np.matmul(qabs_reps, posabs_reps.T)
        q2pos_idxs = np.unravel_index(q2pos_abs_sims.argmax(), q2pos_abs_sims.shape)
        # Print abstracts and similarities.
        # Print abstracts and similarities.
        qabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(qabs)])
        out_file.write(f'Query abstract:\n{ex_dict["query"]["TITLE"]}\n{qabs_str}\n')
        out_file.write(cc2query_abs_sims+'\n')
        contextalign_diff = True if (cc2query_idxs[0], cc2pos_idxs[0]) != (q2pos_idxs[0], q2pos_idxs[1]) else False
        out_file.write(f'cc2q: {cc2query_idxs}; cc2p: {cc2pos_idxs}; q2p: {q2pos_idxs}\n')
        out_file.write(f'contextalign_diff: {contextalign_diff}\n')
        out_file.write('Citing contexts:\n{:}\n'.format('\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(citing_contexts)])))
        out_file.write(cc2pos_abs_sims+'\n')
        pabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(pos_abs)])
        out_file.write(f'Positive abstract:\n{ex_dict["pos_context"]["TITLE"]}\n{pabs_str}\n')
        out_file.write('==================================\n')
        written_count += 1
        if written_count > 2000:
            break
    print(f'Wrote: {out_file.name}')
    out_file.close()


def print_context_abs_sims_ot(trained_model_path, examples_path):
    """
    - Read a triple example,
    - Encode the sentences of the abstract.
    - Encode the citation contexts.
    - Compute pairwise dot products between citation context and the encoded sentences.
    - Do the abstract with a abstract encoder and the sentence with CoSentBert.
    """
    # Init the model.
    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                              max_seq_length=512)
    # Loading local model: https://github.com/huggingface/transformers/issues/2422#issuecomment-571496558
    trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
    word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    in_triples = codecs.open(os.path.join(examples_path, 'dev-cocitabs.jsonl'), 'r', 'utf-8')
    out_file = codecs.open(os.path.join(examples_path, 'dev-abs2abs-otaligns-sims.txt'), 'w', 'utf-8')
    written_count = 0
    for jsonl in in_triples:
        # Encode sentences for triple.
        ex_dict = json.loads(jsonl.strip())
        qabs = ex_dict['query']['ABSTRACT']
        pos_abs = ex_dict['pos_context']['ABSTRACT']
        citing_contexts = ex_dict['citing_contexts']
        reps = sentbert_model.encode(qabs+pos_abs+citing_contexts)
        qabs_reps = reps[:len(qabs)]
        posabs_reps = reps[len(qabs): len(qabs)+len(pos_abs)]
        cc_reps = reps[len(qabs)+len(pos_abs):]
        # Get pairwise sims.
        cc2query_abs_sims = -1*spatial.distance.cdist(qabs_reps, cc_reps)
        cc2query_idxs = np.unravel_index(cc2query_abs_sims.argmax(), cc2query_abs_sims.shape)
        cc2query_abs_sims = np.array2string(cc2query_abs_sims, precision=2)
        cc2pos_abs_sims = -1*spatial.distance.cdist(posabs_reps, cc_reps)
        cc2pos_idxs = np.unravel_index(cc2pos_abs_sims.argmax(), cc2pos_abs_sims.shape)
        cc2pos_abs_sims = np.array2string(cc2pos_abs_sims, precision=2)
        q2pos_abs_sims = -1*spatial.distance.cdist(qabs_reps, posabs_reps)
        q2pos_idxs = np.unravel_index(q2pos_abs_sims.argmax(), q2pos_abs_sims.shape)
        # quniform = np.array([1.0/len(qabs) for i in range(len(qabs))])
        # cuniform = np.array([1.0/len(pos_abs) for i in range(len(pos_abs))])
        # Consider the sentences importance weighted by their best alignment.
        query_distr = scipy.special.softmax(np.max(q2pos_abs_sims, axis=1))
        cand_distr = scipy.special.softmax(np.max(q2pos_abs_sims, axis=0))
        transport_plan_reg = ot.bregman.sinkhorn_epsilon_scaling(query_distr, cand_distr, -1*q2pos_abs_sims, 0.01)
        transport_plan_reg = np.array2string(np.around(transport_plan_reg, 4), precision=3)
        # Print abstracts and similarities.
        qabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(qabs)])
        out_file.write(f'Query abstract:\n{ex_dict["query"]["TITLE"]}\n{qabs_str}\n')
        # out_file.write(cc2query_abs_sims+'\n')
        # contextalign_diff = True if (cc2query_idxs[0], cc2pos_idxs[0]) != (q2pos_idxs[0], q2pos_idxs[1]) else False
        # out_file.write(f'cc2q: {cc2query_idxs}; cc2p: {cc2pos_idxs}; q2p: {q2pos_idxs}\n')
        # out_file.write(f'contextalign_diff: {contextalign_diff}\n')
        # out_file.write('Citing contexts:\n{:}\n'.format('\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(citing_contexts)])))
        # out_file.write(cc2pos_abs_sims+'\n')
        out_file.write(f'Q_distr:\n{np.array2string(query_distr, precision=3)}\n')
        out_file.write(f'C_distr:\n{np.array2string(cand_distr, precision=3)}\n')
        out_file.write(f'Distances:\n{np.array2string(q2pos_abs_sims, precision=3)}\n')
        out_file.write(f'Transport plan:\n{transport_plan_reg}\n')
        pabs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(pos_abs)])
        out_file.write(f'Positive abstract:\n{ex_dict["pos_context"]["TITLE"]}\n{pabs_str}\n')
        out_file.write('==================================\n')
        written_count += 1
        if written_count > 2000:
            break
    print(f'Wrote: {out_file.name}')
    out_file.close()


def print_abs2abs_contextual_sims(embeddings_path, abstracts_path):
    """
    Read embeddings of the contextualized sentences for csfcube and print out
    pairwise sentence similarities for the same abstract.
    """
    with codecs.open(os.path.join(embeddings_path, 'pid2idx-csfcube-sent.json'), 'r', 'utf-8') as fp:
        pid2idx = json.load(fp)
    embeddings = np.load(os.path.join(embeddings_path, 'csfcube-sent.npy'))
    abs_file = codecs.open(os.path.join(abstracts_path, 'abstracts-csfcube-preds.jsonl'), 'r', 'utf-8')
    out_file = codecs.open(os.path.join(abstracts_path, 'abstracts-csfcube-preds-selfsims-ctxt.txt'), 'w', 'utf-8')
    out_file.write(f'Embeddings with: {embeddings_path}\n')
    for abs_line in abs_file:
        abs_dict = json.loads(abs_line.strip())
        pid = abs_dict['paper_id']
        sent_idx = [pid2idx[f'{pid}-{i}'] for i in range(len(abs_dict['abstract']))]
        sent_reps = embeddings[sent_idx]
        abs_self_sims = skmetrics.pairwise.cosine_similarity(sent_reps, sent_reps)
        abs_self_sims = np.array2string(abs_self_sims, precision=2)
        abs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(abs_dict['abstract'])])
        out_file.write(f'Query abstract:\n{abs_dict["title"]}\n{abs_str}\n')
        out_file.write(abs_self_sims+'\n\n')
    print(f'Wrote: {out_file.name}')
    out_file.close()


def print_abs2abs_nocontext_sims(trained_model_path, abstracts_path):
    """
    Read embeddings of the contextualized sentences for csfcube and print out
    pairwise sentence similarities for the same abstract.
    """
    # Init the model.
    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                              max_seq_length=512)
    # Loading local model: https://github.com/huggingface/transformers/issues/2422#issuecomment-571496558
    trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
    word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    abs_file = codecs.open(os.path.join(abstracts_path, 'abstracts-csfcube-preds.jsonl'), 'r', 'utf-8')
    out_file = codecs.open(os.path.join(abstracts_path, 'abstracts-csfcube-preds-selfsims-noctxt.txt'), 'w', 'utf-8')
    out_file.write(f'Embeddings with: {trained_model_path}\n')
    for abs_line in abs_file:
        abs_dict = json.loads(abs_line.strip())
        sent_reps = sentbert_model.encode(abs_dict['abstract'])
        abs_self_sims = skmetrics.pairwise.cosine_similarity(sent_reps, sent_reps)
        abs_self_sims = np.array2string(abs_self_sims, precision=2)
        abs_str = '\n'.join(['{:d}: {:s}'.format(i, s) for i, s in enumerate(abs_dict['abstract'])])
        out_file.write(f'Query abstract:\n{abs_dict["title"]}\n{abs_str}\n')
        out_file.write(abs_self_sims+'\n\n')
    print(f'Wrote: {out_file.name}')
    out_file.close()

    
if __name__ == '__main__':
    print_context_abs_sims_ot(trained_model_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/model_runs/'
                                              's2orccompsci/cosentbert/cosentbert-2021_07_11-22_58_06-fixeddev',
                           examples_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/datasets_proc/'
                                         's2orccompsci/cospecter/')
    # print_abs2abs_contextual_sims(embeddings_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/datasets_raw/'
    #                                    's2orccompsci/conswordbienc/conswordbienc-2021_07_23-17_46_54-specter_init',
    #                               abstracts_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/datasets_raw/'
    #                                              's2orccompsci')
    # print_abs2abs_nocontext_sims(trained_model_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/model_runs/'
    #                                           's2orccompsci/cosentbert/cosentbert-2021_07_11-22_58_06-fixeddev',
    #                              abstracts_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/datasets_raw/'
    #                                             's2orccompsci')
    # print_cocite_contextualsentsim(trained_model_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/model_runs/'
    #                                                   's2orccompsci/conswordbienc/'
    #                                                   'conswordbienc-2021_07_23-17_46_54-specter_init',
    #                                examples_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/datasets_proc/'
    #                                              's2orccompsci/cospecter/')
    # print_cocite_contextualsentsim_contextsent(
    #     trained_abs_model_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/model_runs/'
    #                            's2orccompsci/conswordbienc/'
    #                            'conswordbienc-2021_07_23-17_46_54-specter_init',
    #     trained_sentmodel_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/model_runs/'
    #                            's2orccompsci/cosentbert/cosentbert-2021_07_11-22_58_06-fixeddev',
    #     examples_path='/mnt/nfs/work1/mccallum/smysore/2021-ai2-scisim/datasets_proc/'
    #                   's2orccompsci/cospecter/')
