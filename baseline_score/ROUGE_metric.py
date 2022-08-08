import sys
import os
sys.path.append('../')

from my_sentence_transformers import SentenceTransformer
from resources import BASE_DIR, LANGUAGE, ROUGE_DIR
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score, get_token_vecs, get_sent_vecs, filter_remain_sents_vecs
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult

from resources import BERT_TYPE_PATH_DIC, SENT_TRANSFORMER_TYPE_PATH_DIC
import config
from pyrouge import Rouge155
from nltk.tokenize import word_tokenize
import string
import numpy as np


def run_rouge_score_metrics(year, ref_metric, eval_level, human_metric, rouge_metric,
                            sent_transformer_type='bert_large_nli_stsb_mean_tokens', sent_represnt_type='max_all',
                            pacsum_beta=0.0, pacsum_lambda1=2.0, pacsum_lambda2=1.0, device='cpu'):
    print('year: {}, ref_metric: {}, human_metric: {}, rouge_metric: {}'.format(year, ref_metric, human_metric, rouge_metric))
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR, year)
    human = tacData.getHumanScores(eval_level, human_metric)  # responsiveness or pyramid
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC[sent_transformer_type]
    bert_model = SentenceTransformer(sent_transformer_path, device=device)  # 'bert-large-nli-stsb-mean-tokens')
    all_results = {}

    total_hss = []
    total_pss = []
    # use bertscore or mover-score to compute scores
    topic_idx = -1
    for topic, docs, models in corpus_reader(year):
        topic_idx += 1
        if '.B' in topic: continue
        # read human scores
        hss = [get_human_score(topic, os.path.basename(ss[0]), human, year) for ss in peer_summaries[topic]]
        print('\n=====Topic{}: {}====='.format(topic_idx, topic))
        # prepare the pseudo ref
        if ref_metric == 'true_ref':
            psd_ref = []
            for one_ref in models:
                one_ref_sents = one_ref[1]
                one_ref_sents = [' '.join(word_tokenize(sent.lower())) + '\n' for sent in one_ref_sents]
                psd_ref.append(one_ref_sents)
        else:
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(docs, bert_model, ref_metric,
                                                                                              sent_represnt_type,
                                                                                              pacsum_beta=pacsum_beta,
                                                                                              pacsum_lambda1=pacsum_lambda1,
                                                                                              pacsum_lambda2=pacsum_lambda2)
            ref_dic = {k: sent_info_dic[k] for k in sent_info_dic if sents_weights[k] > 0.0}  # wchen: '>=0.1' -> '> 0.0'
            # for debug
            # print('extracted sent ratio', len(ref_dic)*1./len(sent_info_dic))
            # nstpwd_sents_num = len([1 for svec in sent_vecs if svec is None])
            # nstpwd_sents_ratio = nstpwd_sents_num * 1.0 / len(sent_vecs)
            # print('All nstpwd sents num:{}, ratio:{}'.format(nstpwd_sents_num, nstpwd_sents_ratio))
            ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
            ref_sources = sorted(list(ref_sources))
            # get sents in ref/doc
            ref_sents = []
            sorted_ref_dic_keys = sorted(ref_dic.keys())

            for rs in ref_sources:
                ref_sents.append([ref_dic[k]['text'] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
            # ===========original part=========
            psd_ref = []
            for one_doc_sents in ref_sents:
                one_doc_sents = [' '.join(word_tokenize(sent.lower())) + '\n' for sent in one_doc_sents]
                psd_ref.append(one_doc_sents)
        # prepar sys summs and compute pss scores
        pss = []
        for ss_idx, ss in enumerate(peer_summaries[topic]):
            ss_sents = ss[1]
            if len(ss_sents) == 0:
                pss.append(None)
            else:
                ss_sents = [' '.join(word_tokenize(sent.lower())) + '\n' for sent in ss_sents]
                # prepare tmp folder and files
                tmp_model_folder = os.path.join(BASE_DIR, 'data', 'rouge_tmp', year, topic, 'sys_summs_{}'.format(ss_idx), 'models')
                os.makedirs(tmp_model_folder, exist_ok=True)
                # write model summaries
                for ref_idx, one_ref_sents in enumerate(psd_ref):
                    # 'some_name.[A-Z].#ID#.txt'
                    ref_file_name = os.path.join(tmp_model_folder, topic + '.' + string.ascii_uppercase[ref_idx] + '.' + '{}.txt'.format(ss_idx))
                    with open(ref_file_name, 'w', encoding='utf-8') as fw:
                        fw.writelines(one_ref_sents)
                # write system summaries
                tmp_sys_folder = os.path.join(BASE_DIR, 'data', 'rouge_tmp', year, topic, 'sys_summs_{}'.format(ss_idx), 'systems')
                os.makedirs(tmp_sys_folder, exist_ok=True)
                # 'some_name.(\d+).txt'
                summ_file_name = os.path.join(tmp_sys_folder, topic + '.' + '{}.txt'.format(ss_idx))
                with open(summ_file_name, 'w', encoding='utf-8') as fw:
                    fw.writelines(ss_sents)
                print('rouge_metric: {}'.format(rouge_metric))
                # compute rouge scores
                r = Rouge155()
                r.system_dir = tmp_sys_folder
                r.model_dir = tmp_model_folder
                r.system_filename_pattern = '{}.(\d+).txt'.format(topic)
                r.model_filename_pattern = '{}.[A-Z].#ID#.txt'.format(topic)

                output = r.convert_and_evaluate()
                # print(output)
                # keys={'rouge_Ntype_Stype', 'rouge_Ntype_Stype_cb', 'rouge_Ntype_Stype_ce'}
                # Ntype = {'1', '2', '3', '4', 'l', 'w_1.2', 's*', 'su*'}
                # Stype = {'recall', 'precision', 'f_score'}
                output_dict = r.output_to_dict(output)
                select_rg_score = output_dict['{}_f_score'.format(rouge_metric)]
                pss.append(select_rg_score)
        assert len(pss) == len(hss)
        pseudo_scores, human_scores = [], []
        for i in range(len(pss)):
            if hss[i] is not None and pss[i] is not None:
                pseudo_scores.append(pss[i])
                human_scores.append(hss[i])
        assert len(human_scores) == len(pseudo_scores)
        if len(human_scores) < 2: continue
        total_hss.extend(human_scores)
        total_pss.extend(pseudo_scores)
        if not (np.array(human_scores) == human_scores[0]).all():
            results = evaluateReward(pseudo_scores, human_scores)
            addResult(all_results, results)
            for kk in results:
                print('{}:\t{}'.format(kk, results[kk]))

    print('\n=====ALL Macro RESULTS=====')
    print('year: {}, ref_metric: {}, rouge_metric: {}'.format(year, ref_metric, rouge_metric))
    for kk in all_results:
        if kk.startswith('p_'): continue
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}, significant {} out of {}'.format(kk, np.max(
            all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk]), len(
            [p for p in all_results['p_{}'.format(kk)] if p < 0.05]), len(all_results[kk])))

    assert len(total_hss) == len(total_pss)
    print('\n=====ALL Micro RESULTS=====')
    results = evaluateReward(total_pss, total_hss)
    for kk in results:
        print('{}:\t{}'.format(kk, results[kk]))


if __name__ == '__main__':
    # get the general configuration
    parser = config.ArgumentParser("ROUGW_metric.py")
    config.general_args(parser)
    config.pseudo_ref_sim_metrics_args(parser)
    config.rouge_metric_args(parser)
    config.my_metrics_args(parser)
    opt = parser.parse_args()
    print("\nMetric: sbert_score_metrics.py")
    print("Configurations:", opt)
    # '08', '09', '2010', '2011', 'cnndm'
    year = opt.year
    human_metric = opt.human_metric
    ref_metric = opt.ref_metric
    eval_level = opt.evaluation_level
    rouge_metric = opt.rouge_metric
    sent_transformer_type = opt.sent_transformer_type
    sent_represnt_type = opt.sent_represnt_type
    pacsum_beta = opt.pacsum_beta
    pacsum_lambda1 = opt.pacsum_lambda1
    pacsum_lambda2 = opt.pacsum_lambda2
    device = opt.device
    run_rouge_score_metrics(year=year,
                            ref_metric=ref_metric,
                            eval_level=eval_level,
                            human_metric=human_metric,
                            rouge_metric=rouge_metric,
                            sent_transformer_type=sent_transformer_type,
                            sent_represnt_type=sent_represnt_type,
                            pacsum_beta=pacsum_beta,
                            pacsum_lambda1=pacsum_lambda1,
                            pacsum_lambda2=pacsum_lambda2,
                            device=device)
