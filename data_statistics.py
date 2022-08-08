import sys
import os
import statistics
sys.path.append('../..')

from my_sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult

from resources import BERT_TYPE_PATH_DIC, SENT_TRANSFORMER_TYPE_PATH_DIC
import config
import json


def data_statistics_collector(year, ref_metric, eval_level='summary',
                              sent_transformer_type='bert_large_nli_stsb_mean_tokens',
                              add_gold_refs_to_sys_summs=False,
                              sent_represnt_type='max_all',
                              pacsum_beta=0.0,
                              pacsum_lambda1=2.0,
                              pacsum_lambda2=1.0,
                              device='cpu',
                              log_file=None):
    print('year: {}, ref_metric: {}'.format(year, ref_metric))
    log_file.write('year: {}, ref_metric: {}\n'.format(year, ref_metric))
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR, year)
    if year == 'cnndm':
        human_cnndm_overall = tacData.getHumanScores(eval_level, 'overall')
        human_cnndm_grammar = tacData.getHumanScores(eval_level, 'grammar')
        human_cnndm_redundancy = tacData.getHumanScores(eval_level, 'redundancy')
    else:
        human_pyramid = tacData.getHumanScores(eval_level, 'pyramid')  # responsiveness or pyramid
        human_respns = tacData.getHumanScores(eval_level, 'responsiveness')  # responsiveness or pyramid
    # assert sent_transformer_type == 'bert_large_nli_stsb_mean_tokens'
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC[sent_transformer_type]
    bert_model = SentenceTransformer(sent_transformer_path, device=device)  # 'bert-large-nli-stsb-mean-tokens')

    statistics_dict = {}
    # use mover-score or bertscore to compute scores
    topic_idx = -1
    doc_set_num = []
    doc_sent_num = []
    doc_word_num = []
    sys_num = []
    sys_summ_sent_num = []
    sys_summ_word_num = []
    for topic, docs, models in corpus_reader(year):
        if '.B' in topic: continue
        topic_idx += 1
        doc_set_num.append(len(docs))
        print('\n=====Topic{}: {}====='.format(topic_idx, topic))
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(models, bert_model,
                                                                                              ref_metric,
                                                                                              sent_represnt_type)
        else:
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(docs, bert_model,
                                                                                              ref_metric,
                                                                                              sent_represnt_type,
                                                                                              pacsum_beta=pacsum_beta,
                                                                                              pacsum_lambda1=pacsum_lambda1,
                                                                                              pacsum_lambda2=pacsum_lambda2)
        ref_dic = {k: sent_info_dic[k] for k in sent_info_dic if sents_weights[k] > 0.0}  # wchen: '>=0.1' -> '> 0.0'
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        ref_sources = sorted(list(ref_sources))

        # build reference list
        references = {doc_name: [] for doc_name in ref_sources}
        ref_sent_word_num = {doc_name: [] for doc_name in ref_sources}
        for k in sorted(ref_dic.keys()):
            for doc_name in references:
                if ref_dic[k]['doc'] == doc_name:
                    # the sentence order is consistent with the order of k
                    # therefore, no reordering is required
                    references[doc_name].append(ref_dic[k]['text'])
                    ref_sent_word_num[doc_name].append(len(all_tokens[k]) - 2)
                    break

        for doc_name in references:
            doc_sent_num.append(len(references[doc_name]))
            doc_word_num.append(sum(ref_sent_word_num[doc_name]))

        sys_num.append(len(peer_summaries[topic]))
        for ss_idx, ss in enumerate(peer_summaries[topic]):
            if len(ss[1]) != 0:
                one_summ_sents_vecs, one_summ_tokens_vecs, one_summ_tokens = bert_model.encode(ss[1], sent_represnt_type)
                ss_sent_num = len(one_summ_tokens)
                ss_word_num = sum([len(one_ss_sent) - 2 for one_ss_sent in one_summ_tokens])
                sys_summ_sent_num.append(ss_sent_num)
                sys_summ_word_num.append(ss_word_num)
            else:
                svv, vv, tt, tweights, sweights = None, None, None, None, None
    keys = []
    statistics_dict['# of topic'] = topic_idx + 1
    keys.append('# of topic')
    statistics_dict['# of doc set'] = round(sum(doc_set_num) / len(doc_set_num), 1)
    keys.append('# of doc set')
    statistics_dict['# of sent per doc'] = round(sum(doc_sent_num) / len(doc_sent_num), 1)
    keys.append('# of sent per doc')
    statistics_dict['# of word per doc'] = round(sum(doc_word_num) / len(doc_word_num), 1)
    keys.append('# of word per doc')
    statistics_dict['# of systems'] = round(sum(sys_num) / len(sys_num), 1)
    keys.append('# of systems')
    statistics_dict['# of sent per summ'] = round(sum(sys_summ_sent_num) / len(sys_summ_sent_num), 1)
    keys.append('# of sent per summ')
    statistics_dict['# of word per summ'] = round(sum(sys_summ_word_num) / len(sys_summ_word_num), 1)
    keys.append('# of word per summ')
    statistics_dict['stdv of word per summ'] = round(statistics.stdev(sys_summ_word_num))
    keys.append('stdv of word per summ')

    for key in keys:
        line = key + ': {}'.format(statistics_dict[key]) + '\n'
        print(line)
        if log_file is not None:
            log_file.write(line)

    print('\n\n\n')
    log_file.write('\n\n\n')


if __name__ == '__main__':
    # get the general configuration
    parser = config.ArgumentParser("pseudo_ref_file_generator.py")
    config.pseudo_ref_file_generator_args(parser)
    config.my_metrics_args(parser)
    opt = parser.parse_args()
    print("\nMetric: pseudo_ref_file_generator.py")
    print("Configurations:", opt)
    # '08', '09', '2010', '2011', 'cnndm'
    year = opt.year
    ref_summ = opt.ref_summ
    ref_metric = opt.ref_metric
    eval_level = opt.evaluation_level
    sent_transformer_type = opt.sent_transformer_type
    add_gold_refs_to_sys_summs = opt.add_gold_refs_to_sys_summs
    sent_represnt_type = opt.sent_represnt_type
    pacsum_beta = opt.pacsum_beta
    pacsum_lambda1 = opt.pacsum_lambda1
    pacsum_lambda2 = opt.pacsum_lambda2
    device = opt.device

    # save the statistics into a log file
    folder_name = os.path.join('logs', 'data_statistics')
    os.makedirs(folder_name, exist_ok=True)
    log_file = open(os.path.join(folder_name, 'data_statistics.txt'), 'w', encoding='utf-8')

    ref_metric = 'full_doc'
    for year in ['2010', '2011', '09', '08', 'cnndm']:
        data_statistics_collector(year=year, ref_metric=ref_metric, eval_level=eval_level,
                                  sent_transformer_type=sent_transformer_type,
                                  add_gold_refs_to_sys_summs=add_gold_refs_to_sys_summs,
                                  sent_represnt_type=sent_represnt_type,
                                  pacsum_beta=pacsum_beta,
                                  pacsum_lambda1=pacsum_lambda1,
                                  pacsum_lambda2=pacsum_lambda2,
                                  device=device, log_file=log_file)
