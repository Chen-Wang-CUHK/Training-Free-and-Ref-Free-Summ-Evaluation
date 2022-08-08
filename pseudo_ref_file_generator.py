import sys
import os
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

def moverscore_style_pseudo_ref_gen(year, ref_metric, eval_level='summary',
                                    sent_transformer_type='bert_large_nli_stsb_mean_tokens',
                                    add_gold_refs_to_sys_summs=False,
                                    sent_represnt_type='max_all',
                                    pacsum_beta=0.0,
                                    pacsum_lambda1=2.0,
                                    pacsum_lambda2=1.0,
                                    device='cpu'):
    '''
    the format of moverscore style input dataset file

    {
	'D0939':{
		'references':[{'text':['sum_sent0', 'sum_sent1',...,'sum_sentN1'],'id':'D0939-A.M.100.H.F'}*4]
		'annotations':[{
			'responsiveness':3.0,
			'pyr_mod_score':0.364, //references have no this key
			'text':['sum_sent0', 'sum_sent1',...,'sum_sentN2'],
			'pyr_score':0.364,
			'topic_id':'D0939-A',
			'summ_id':1
		    }*(55 systems + 4 references)]
	    }
    }
    '''
    print('year: {}, ref_metric: {}'.format(year,ref_metric))
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR,year)
    if year == 'cnndm':
        human_cnndm_overall = tacData.getHumanScores(eval_level, 'overall')
        human_cnndm_grammar = tacData.getHumanScores(eval_level, 'grammar')
        human_cnndm_redundancy = tacData.getHumanScores(eval_level, 'redundancy')
    else:
        human_pyramid = tacData.getHumanScores(eval_level, 'pyramid') # responsiveness or pyramid
        human_respns = tacData.getHumanScores(eval_level, 'responsiveness') # responsiveness or pyramid
    # assert sent_transformer_type == 'bert_large_nli_stsb_mean_tokens'
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC[sent_transformer_type]
    bert_model = SentenceTransformer(sent_transformer_path, device=device)  # 'bert-large-nli-stsb-mean-tokens')

    moverscore_dataset = {}
    # use mover-score or bertscore to compute scores
    topic_idx = -1
    for topic,docs,models in corpus_reader(year):
        topic_idx += 1
        if '.B' in topic: continue
        print('\n=====Topic{}: {}====='.format(topic_idx, topic))
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(models,bert_model,ref_metric,sent_represnt_type)
        else:
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(docs,bert_model,ref_metric,sent_represnt_type,
                                                                                              pacsum_beta=pacsum_beta, pacsum_lambda1=pacsum_lambda1, pacsum_lambda2=pacsum_lambda2)
        ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k] > 0.0} # wchen: '>=0.1' -> '> 0.0'
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        ref_sources = sorted(list(ref_sources))

        # build a moversocre-style data instance
        topic_name = topic.split('.')[0]
        # build reference list
        references = [{'text': [], 'sents_weights': [], 'id': rs} for rs in ref_sources]
        for k in sorted(ref_dic.keys()):
            for idx in range(len(references)):
                if ref_dic[k]['doc'] == references[idx]['id']:
                    # the sentence order is consistent with the order of k
                    # therefore, no reordering is required
                    references[idx]['text'].append(ref_dic[k]['text'])
                    references[idx]['sents_weights'].append(sents_weights[k])
                    break

        # build annotation list
        topic_id = topic.replace('.', '-')
        annotations = []
        current_system_summs = peer_summaries[topic]
        for sys_tuple in current_system_summs:
            one_annot = {'topic_id': topic_id}
            file_name = os.path.basename(sys_tuple[0])
            summ_id = file_name.split('.')[-1]
            one_annot['summ_id'] = summ_id
            if year == 'cnndm':
                # 'overall'
                sys_overall = human_cnndm_overall['topic{}_sum_{}'.format(topic_id, summ_id)]
                one_annot['overall'] = sys_overall
                # 'grammar'
                sys_grammar = human_cnndm_grammar['topic{}_sum_{}'.format(topic_id, summ_id)]
                one_annot['grammar'] = sys_grammar
                # 'redundancy'
                sys_redundancy = human_cnndm_redundancy['topic{}_sum_{}'.format(topic_id, summ_id)]
                one_annot['redundancy'] = sys_redundancy
            else:
                # 'responsiveness'
                sys_respns = human_respns['topic{}_sum{}'.format(topic_id, summ_id)]
                one_annot['responsiveness'] = sys_respns
                # 'pyr_score'
                sys_pyramid = human_pyramid['topic{}_sum{}'.format(topic_id, summ_id)]
                one_annot['pyr_score'] = sys_pyramid
            # 'text'
            sys_text = sys_tuple[1]
            one_annot['text'] = sys_text

            annotations.append(one_annot)

        if year != 'cnndm':
            annotations = sorted(annotations, key=lambda i: float(i['summ_id']))

        if add_gold_refs_to_sys_summs:
            gold_refs_annots = []
            for true_ref_tuple in models:
                one_annot = {'topic_id': topic_id}
                if year != 'cnndm':
                    summ_id = true_ref_tuple[0].split('.')[-1]
                    one_annot['summ_id'] = summ_id
                    # 'responsiveness'
                    true_ref_respns = human_respns['topic{}_blockmodel_sum{}'.format(topic_id, summ_id)]
                    one_annot['responsiveness'] = true_ref_respns
                    # 'pyr_score'
                    true_ref_pyramid = human_pyramid['topic{}_blockmodel_sum{}'.format(topic_id, summ_id)]
                    one_annot['pyr_score'] = true_ref_pyramid
                    # 'text'
                    true_ref_text = sent_tokenize(' '.join(true_ref_tuple[1]))
                    one_annot['text'] = true_ref_text
                else:
                    summ_id = 'reference'
                    one_annot['summ_id'] = summ_id
                    # 'overall'
                    true_ref_overall = human_cnndm_overall['topic{}_sum_{}'.format(topic_id, summ_id)]
                    one_annot['overall'] = true_ref_overall
                    # 'grammar'
                    true_ref_grammar = human_cnndm_grammar['topic{}_sum_{}'.format(topic_id, summ_id)]
                    one_annot['grammar'] = true_ref_grammar
                    # 'redundancy'
                    true_ref_redundancy = human_cnndm_redundancy['topic{}_sum_{}'.format(topic_id, summ_id)]
                    one_annot['redundancy'] = true_ref_redundancy

                gold_refs_annots.append(one_annot)
            if year != 'cnndm':
                gold_refs_annots = sorted(gold_refs_annots, key=lambda i: i['summ_id'])
            annotations = annotations + gold_refs_annots

        # add the data instance
        moverscore_dataset[topic_name] = {'references': references, 'annotations': annotations}

    # save the built moverscore style dataset file
    folder_name = os.path.join('data', 'moverscore_style_files')
    os.makedirs(folder_name, exist_ok=True)
    # whether contain true reference annotations in the system annotations needed to evaluate
    with_true_refs = 'withTrueRefsInSys.' if add_gold_refs_to_sys_summs else ''
    dataset_name = 'tac'
    if ref_metric == 'true_ref':
        file_name = os.path.join(folder_name, '{}.{}.trueRef.{}mds.gen.resp-pyr'.format(dataset_name, year, with_true_refs))
    else:
        file_name = os.path.join(folder_name, '{}.{}.psdRef.{}.{}withSentsW.mds.gen.resp-pyr'.format(dataset_name, year, ref_metric, with_true_refs))
    json.dump(moverscore_dataset, open(file_name, 'w'))


def s_wms_style_pseudo_ref_gen(year, ref_metric, eval_level='summary',
                               sent_transformer_type='bert_large_nli_stsb_mean_tokens',
                               add_gold_refs_to_sys_summs=False,
                               sent_represnt_type='max_all',
                               pacsum_beta=0.0,
                               pacsum_lambda1=2.0,
                               pacsum_lambda2=1.0,
                               device='cpu'):
    """
    The input file format of the S+WMS:
        ref11 \t sys_summ1
        ref12 \t sys_summ1
        ...
        ref11 \t sys_summ2
        ref12 \t sys_summ2
        ...
    """
    # save the built s+wms style dataset file
    folder_name = os.path.join('data', 's_wms_style_files')
    os.makedirs(folder_name, exist_ok=True)
    text_pair_file = open(os.path.join(folder_name, 'sms_tac_{}_text_pair.txt'.format(year)), 'w', encoding='utf-8')
    scores_file = open(os.path.join(folder_name, 'sms_tac_{}_scores.jsonl'.format(year)), 'w', encoding='utf-8')

    print('year: {}, ref_metric: {}'.format(year, ref_metric))
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

    moverscore_dataset = {}
    # use mover-score or bertscore to compute scores
    topic_idx = -1
    line_cnt = 0
    for topic, docs, models in corpus_reader(year):
        topic_idx += 1
        if '.B' in topic: continue
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

        # build a sms-style data instance
        topic_name = topic.split('.')[0]
        # build reference list
        references = {doc_name: [] for doc_name in ref_sources}
        for k in sorted(ref_dic.keys()):
            for doc_name in references:
                if ref_dic[k]['doc'] == doc_name:
                    # the sentence order is consistent with the order of k
                    # therefore, no reordering is required
                    references[doc_name].append(ref_dic[k]['text'])
                    break
        # build each ref sys_summ pair and human-annotated scores
        current_system_summs = peer_summaries[topic]
        for sys_tuple in current_system_summs:
            summ_file_name = os.path.basename(sys_tuple[0])
            summ_txt = sys_tuple[1]
            if len(summ_txt) == 0 or (len(summ_txt) == 1 and summ_txt[0].strip() == ''):
                continue
            for doc_name in references:
                # psdRef-sys_summs pair
                one_pair = ' '.join(references[doc_name]) + '\t' + ' '.join(summ_txt) + '\n'
                if year != 'cnndm':
                    topic_id = topic.replace('.', '-')
                    summ_id = summ_file_name.split('.')[-1]
                    score_key = 'topic{}_sum{}'.format(topic_id, summ_id)
                    scores = {'pyr_score': human_pyramid[score_key],
                              'responsiveness': human_respns[score_key]}
                else:
                    topic_id = topic
                    summ_id = summ_file_name
                    score_key = 'topic{}_sum_{}'.format(topic_id, summ_id)
                    scores = {'overall': human_cnndm_overall[score_key],
                              'grammar': human_cnndm_grammar[score_key],
                              'redundancy': human_cnndm_redundancy[score_key]}

                scores_info = {'id': 'topic{}_doc_name_{}_summ_{}'.format(topic_id, doc_name, summ_id),
                               'human_scores': scores}
                scores_info = json.dumps(scores_info) + '\n'

                # write one_pair and scores_info into the files
                text_pair_file.write(one_pair)
                scores_file.write(scores_info)
                line_cnt = line_cnt + 1
    print("\nTotally {} psdRef-summ pairs".format(line_cnt))


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
    if opt.style_type == 'moverscore':
        moverscore_style_pseudo_ref_gen(year=year, ref_metric=ref_metric, eval_level=eval_level,
                                        sent_transformer_type=sent_transformer_type,
                                        add_gold_refs_to_sys_summs=add_gold_refs_to_sys_summs,
                                        sent_represnt_type=sent_represnt_type,
                                        pacsum_beta=pacsum_beta,
                                        pacsum_lambda1=pacsum_lambda1,
                                        pacsum_lambda2=pacsum_lambda2,
                                        device=device)
    elif opt.style_type == 'sms':
        s_wms_style_pseudo_ref_gen(year=year, ref_metric=ref_metric, eval_level=eval_level,
                                   sent_transformer_type=sent_transformer_type,
                                   add_gold_refs_to_sys_summs=add_gold_refs_to_sys_summs,
                                   sent_represnt_type=sent_represnt_type,
                                   pacsum_beta=pacsum_beta,
                                   pacsum_lambda1=pacsum_lambda1,
                                   pacsum_lambda2=pacsum_lambda2,
                                   device=device)
