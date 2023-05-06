import sys
import os
import copy
sys.path.append('../..')

from my_sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
import string

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score, get_token_vecs, get_sent_vecs, filter_remain_sents_vecs
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult

from resources import BERT_TYPE_PATH_DIC, SENT_TRANSFORMER_TYPE_PATH_DIC
import config
import json


from torch import cuda


def get_idf(doc_token_list):
    df_dic = {}
    for i,doc_tokens in enumerate(doc_token_list):
        if doc_tokens is None: continue
        for tk in doc_tokens:
            if tk in df_dic: df_dic[tk].append(i)
            else: df_dic[tk] = [i]

    doc_num = len(doc_token_list)
    idf_list = []
    for i,doc_tokens in enumerate(doc_token_list):
        if doc_tokens is None:
            idf_list.append(None)
            continue
        idf = []
        for tk in doc_tokens: idf.append(-1.*np.log( (len(set(df_dic[tk]))+0.5)/(doc_num+0.5)))
        idf_list.append(np.array(idf))
    return idf_list


def mrg_tokens_sents(tokens_vecs, token_weights, sents_vecs, sents_weights):
    mrgd_vecs = []
    mrgd_weights = []
    for tvec, tweights, svec, sweights in zip(tokens_vecs, token_weights, sents_vecs, sents_weights):
        if svec is not None and tvec is not None:
            mrgd_vecs.append(np.concatenate([tvec, svec], axis=0))  # (nstpw_num + sent_num, dim)
            mrgd_weights.append(np.concatenate([tweights, sweights]))
        elif svec is not None:
            # if tvec is None, sve may not be None
            mrgd_vecs.append(svec)  # (sent_num, dim)
            mrgd_weights.append(sweights)
        elif tvec is not None:
            # if svec is None, tve may not be None
            mrgd_vecs.append(tvec)  # (nstpw_num, dim)
            mrgd_weights.append(tweights)
        else:
            mrgd_vecs.append(None)
            mrgd_weights.append(None)
    return mrgd_vecs, mrgd_weights


def get_my_score(ref_vecs, ref_weights, ref_tokens, summ_vecs, summ_weights, summ_tokens, wmd_score_type, wmd_weight_type, mask_self=False, beta_gamma=2):
    recall_list = []
    precision_list = []
    f1_list = []
    empty_summs_ids = []

    if mask_self:
        assert wmd_weight_type == 'none'
        assert wmd_score_type == 'recall'

    if 'idf' in wmd_weight_type:
        final_ref_weights = get_idf(ref_tokens)
        final_summ_weights = get_idf(summ_tokens)
    elif 'graph_weighted' in wmd_weight_type:
        final_ref_weights = ref_weights
        final_summ_weights = summ_weights

    if 'renormalize' in wmd_weight_type:
        final_ref_weights = [final_ref_weights[i] / final_ref_weights[i].sum() if final_ref_weights[i] is not None else None for i in range(len(final_ref_weights))]
        final_summ_weights = [final_summ_weights[i] / final_summ_weights[i].sum() if final_summ_weights[i] is not None else None for i in range(len(final_summ_weights))]

    for i,rvecs in enumerate(ref_vecs):
        r_recall_list = []
        r_precision_list = []
        r_f1_list = []
        for j,svecs in enumerate(summ_vecs):
            if svecs is None or len(svecs) == 0:
                empty_summs_ids.append(j)
                r_recall_list.append(None)
                r_precision_list.append(None)
                r_f1_list.append(None)
                continue
            if mask_self:
                # only token level information is utilized
                assert rvecs.shape[0] == len(ref_tokens[i])
                assert svecs.shape[0] == len(summ_tokens[j])
                # the matrix should be square matrix
                assert rvecs.shape[0] == svecs.shape[0]
            sim_matrix = cosine_similarity(rvecs,svecs)
            if mask_self:
                np.fill_diagonal(sim_matrix, 0)
            beta_square = 1
            if wmd_score_type == 'f1_beta':
                beta_square = (rvecs.shape[0] / svecs.shape[0]) ** (1/beta_gamma)
                beta_square = 2 if beta_square > 2 else beta_square
                beta_square = 1 if beta_square < 1 else beta_square
            if 'idf' in wmd_weight_type or 'graph_weighted' in wmd_weight_type:
                weighted_recall = np.dot(np.max(sim_matrix, axis=1), final_ref_weights[i])
                weighted_precision = np.dot(np.max(sim_matrix, axis=0), final_summ_weights[j])
                weighted_f1 = (1. + beta_square) * weighted_recall * weighted_precision / (weighted_recall + beta_square * weighted_precision)
                r_recall_list.append(weighted_recall)
                r_precision_list.append(weighted_precision)
                r_f1_list.append(weighted_f1)
            else:
                recall = np.mean(np.max(sim_matrix, axis=1))
                precision = np.mean(np.max(sim_matrix, axis=0))
                f1 = (1. + beta_square) * recall * precision / (recall + beta_square * precision)
                r_recall_list.append(recall)
                r_precision_list.append(precision)
                r_f1_list.append(f1)
        recall_list.append(r_recall_list)
        precision_list.append(r_precision_list)
        f1_list.append(r_f1_list)
    empty_summs_ids = list(set(empty_summs_ids))
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    f1_list = np.array(f1_list)
    if 'recall' in wmd_score_type:
        scores = []
        for i in range(len(summ_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(recall_list[:,i]))
        return scores
        #return np.mean(np.array(recall_list), axis=0)
    elif 'precision' in wmd_score_type:
        scores = []
        for i in range(len(summ_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(precision_list[:,i]))
        return scores
        #return np.mean(np.array(precision_list), axis=0)
    else:
        assert 'f1' in wmd_score_type
        scores = []
        for i in range(len(summ_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(f1_list[:,i]))
        return scores
        #return np.mean(np.mean(f1_list),axis=0)


def run_my_score_metrics(year, ref_metric, wmd_score_type, wmd_weight_type,
                         eval_level='summary', human_metric='pyramid',
                         sent_transformer_type='bert_large_nli_stsb_mean_tokens', map_type='t2t',
                         sent_represnt_type='mean_all', ref_st_mrg_type='wAll_sAll', sim_th=0.0,
                         lambda_redund=0.0, pacsum_beta=0.0, pacsum_lambda1=2.0, pacsum_lambda2=1.0, beta_gamma=2,
                         summ_sys_num_limit=-1, doc_num_limit=-1, score_saved_file=None, device='cpu'):
    print('year: {}, ref_metric: {}, wmd_score_type: sbert-{}, lambda_redund: {}'.format(year,ref_metric,wmd_score_type,lambda_redund))
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    # select the generated summaries
    if summ_sys_num_limit > 0:
        for topic in peer_summaries:
            peer_summaries[topic] = peer_summaries[topic][:summ_sys_num_limit]
    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores(eval_level, human_metric) # responsiveness or pyramid
    # assert sent_transformer_type == 'bert_large_nli_stsb_mean_tokens'
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC[sent_transformer_type]
    bert_model = SentenceTransformer(sent_transformer_path, device=device)  # 'bert-large-nli-stsb-mean-tokens')
    all_results = {}

    total_hss = []
    total_pss = []
    # use bertscore or mover-score to compute scores
    topic_idx = -1
    if score_saved_file is not None:
        score_saved_file = open(score_saved_file, 'w', encoding='utf-8')

    for topic,docs,models in corpus_reader(year):
        topic_idx += 1
        # if topic not in ['001992', '004237']: continue # for debug
        if '.B' in topic: continue
        # select docs
        if doc_num_limit > 0:
            docs = docs[:doc_num_limit]
        # read human scores
        hss = [get_human_score(topic, os.path.basename(ss[0]), human, year) for ss in peer_summaries[topic]]
        # # When use Macro-averaging, filter out the instance which has the same hss for all systems
        # if year in ['cnndm'] and (np.array(hss) == hss[0]).all():
        #     continue

        print('\n=====Topic{}: {}====='.format(topic_idx, topic))
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(models,bert_model,ref_metric,sent_represnt_type)
        else:
            sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(docs,bert_model,ref_metric,sent_represnt_type,
                                                                                              pacsum_beta=pacsum_beta, pacsum_lambda1=pacsum_lambda1, pacsum_lambda2=pacsum_lambda2)
        ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k] > 0.0} # wchen: '>=0.1' -> '> 0.0'
        #for debug
        #print('extracted sent ratio', len(ref_dic)*1./len(sent_info_dic))
        #nstpwd_sents_num = len([1 for svec in sent_vecs if svec is None])
        #nstpwd_sents_ratio = nstpwd_sents_num * 1.0 / len(sent_vecs)
        #print('All nstpwd sents num:{}, ratio:{}'.format(nstpwd_sents_num, nstpwd_sents_ratio))
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        ref_sources = sorted(list(ref_sources))
        # get sents in ref/doc
        ref_sents = []
        ref_sents_vecs = []
        ref_sents_weights = []
        ref_tokens_vecs = []
        ref_tokens = []
        sorted_ref_dic_keys = sorted(ref_dic.keys())

        for rs in ref_sources:
            ref_sents.append([ref_dic[k]['text'] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
            ref_sents_vecs.append([sent_vecs[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
            ref_sents_weights.append([sents_weights[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
            ref_tokens_vecs.append([token_vecs[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
            ref_tokens.append([all_tokens[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])

        # get the filtered vecs fro ref
        filtered_ref_tokens_vecs = []
        filtered_ref_tokens = []
        filtered_ref_token_weights = []

        if ref_st_mrg_type.startswith('wTop'):
            wTopNum = int(ref_st_mrg_type.split('_')[0].strip()[4:])
            token_level_idx_list = []
            sent_level_idx_list = []
            for doc_weights in ref_sents_weights:
                doc_weights = np.array(doc_weights)
                if doc_weights.all():
                    idxs_list = [k for k in range(len(doc_weights))]
                else:
                    idxs_list = doc_weights.argsort().tolist()
                    idxs_list = idxs_list[::-1]
                token_level_idx_list.append(idxs_list[:wTopNum])
                if ref_st_mrg_type.endswith('sBottom'):
                    sent_level_idx_list.append(idxs_list[wTopNum:])
                else:
                    assert ref_st_mrg_type.endswith('sAll')
                    sent_level_idx_list.append(idxs_list)
        else:
            token_level_idx_list = [[k for k in range(len(doc_weights))] for doc_weights in ref_sents_weights]
            sent_level_idx_list = token_level_idx_list

        for doc_idx in range(len(ref_sents)):
            tvecs_in = [ref_tokens_vecs[doc_idx][k] for k in token_level_idx_list[doc_idx]]
            tokens_in = [ref_tokens[doc_idx][k] for k in token_level_idx_list[doc_idx]]
            # we use sent weight as the weight of each token
            weights_in = [np.array([ref_sents_weights[doc_idx][k]]*len(ref_tokens[doc_idx][k])) for k in token_level_idx_list[doc_idx]]
            vv, tt, ww = get_token_vecs(vecs=tvecs_in, tokens=tokens_in, weights=weights_in)
            filtered_ref_tokens_vecs.append(vv)
            filtered_ref_tokens.append(tt)
            filtered_ref_token_weights.append(ww)

        assert sim_th == 0.0, "The sim_th argument is abandoned"
        # if wTopNum is not None and sim_th > 0.:
        #     ref_sents_vecs = filter_remain_sents_vecs(ref_sents_vecs, wTopNum, sim_th)

        filtered_ref_sents_vecs = []
        filtered_ref_sent_weights = []
        for svec_list, sweights, sent_level_idxs in zip(ref_sents_vecs, ref_sents_weights, sent_level_idx_list):
            remain_svecs = None
            remain_sweights = None
            if len(sent_level_idxs) > 0:
                # remain_svecs = [svec for svec in svec_list[wTopNum:] if svec is not None]
                remain_svecs = [svec_list[k] for k in sent_level_idxs if svec_list[k] is not None]
                remain_sweights = [sweights[k] for k in sent_level_idxs if svec_list[k] is not None]
                if len(remain_svecs) > 0:
                    remain_svecs = np.stack(remain_svecs)
                    remain_sweights = np.array(remain_sweights)
                else:
                    remain_svecs = None
                    remain_sweights = None
            filtered_ref_sents_vecs.append(remain_svecs)
            filtered_ref_sent_weights.append(remain_sweights)

        # get sents in system summaries
        filtered_summ_tokens_vecs = []
        filtered_summ_tokens = []
        filtered_summ_token_weights = []
        filtered_summ_sents_vecs = []
        filtered_summ_sent_weights = []

        # all_summ_sents = []
        # all_sys_idx = []
        # for ss_idx, ss in enumerate(peer_summaries[topic]):
        #     all_summ_sents.extend(ss[1])
        #     all_sys_idx.extend([ss_idx]*len(ss[1]))
        # assert len(all_summ_sents) == len(all_sys_idx)
        # all_summ_sents_vecs, all_summ_tokens_vecs, all_summ_tokens = bert_model.encode(all_summ_sents)

        for ss_idx, ss in enumerate(peer_summaries[topic]):
            if len(ss[1]) != 0:
                one_summ_sents_vecs, one_summ_tokens_vecs, one_summ_tokens = bert_model.encode(ss[1], sent_represnt_type)
                # print('summary length: {}'.format(sum([len(one_ss_sent) for one_ss_sent in one_summ_tokens]))) # for debug
                vv, tt, _ = get_token_vecs(vecs=one_summ_tokens_vecs, tokens=one_summ_tokens)
                svv = np.stack([svec for svec in one_summ_sents_vecs if svec is not None])
                tweights = np.ones(tt.shape[0])
                sweights = np.ones(svv.shape[0])
            else:
                svv, vv, tt, tweights, sweights = None, None, None, None, None
            filtered_summ_tokens_vecs.append(vv)
            filtered_summ_tokens.append(tt)
            filtered_summ_token_weights.append(tweights)
            filtered_summ_sents_vecs.append(svv)
            filtered_summ_sent_weights.append(sweights)

        # get the merged sent and token representations of references
        filtered_ref_mrgd_vecs, filtered_ref_mrgd_weights = mrg_tokens_sents(filtered_ref_tokens_vecs,
                                                                             filtered_ref_token_weights,
                                                                             filtered_ref_sents_vecs,
                                                                             filtered_ref_sent_weights)
        # get the merged sent and token representations of summs
        filtered_summ_mrgd_vecs, filtered_summ_mrgd_weights = mrg_tokens_sents(filtered_summ_tokens_vecs,
                                                                               filtered_summ_token_weights,
                                                                               filtered_summ_sents_vecs,
                                                                               filtered_summ_sent_weights)
        # filtered_ref_mrgd_vecs = []
        # filtered_ref_mrgd_weights = []
        # for tvec, tweights, svec, sweights in zip(filtered_ref_tokens_vecs, filtered_ref_token_weights, filtered_ref_sents_vecs, filtered_ref_sents_weights):
        #     if svec is not None and tvec is not None:
        #         filtered_ref_mrgd_vecs.append(np.concatenate([tvec, svec], axis=0)) # (ref_nstpw_num + ref_sent_num, dim)
        #         filtered_ref_mrgd_weights.append(np.concatenate([tweights, sweights]))
        #     elif svec is not None:
        #         # if tvec is None, sve may not be None
        #         filtered_ref_mrgd_vecs.append(svec)  # (ref_sent_num, dim)
        #         filtered_ref_mrgd_weights.append(sweights)
        #     elif tvec is not None:
        #         # if svec is None, tve may not be None
        #         filtered_ref_mrgd_vecs.append(tvec)  # (ref_nstpw_num, dim)
        #         filtered_ref_mrgd_weights.append(tweights)
        #     else:
        #         filtered_ref_mrgd_vecs.append(None)
        #         filtered_ref_mrgd_weights.append(None)

        # # get the merged sent and token representations of summs
        # filtered_summ_mrgd_vecs = []
        # for tvec, svec in zip(filtered_summ_tokens_vecs, filtered_summ_sents_vecs):
        #     if svec is not None or tvec is not None:
        #         filtered_summ_mrgd_vecs.append(np.concatenate([tvec, svec], axis=0))  # (summ_nstpw_num + summ_sent_num, dim)
        #     elif svec is not None:
        #         filtered_summ_mrgd_vecs.append(svec) # (summ_sent_num, dim)
        #     elif tvec is not None:
        #         filtered_summ_mrgd_vecs.append(tvec) # (summ_nstpwd_num, dim)
        #     else:
        #         filtered_summ_mrgd_vecs.append(None)

        # get the final input vectors
        assert '2' in map_type
        map_type_ref, map_type_summ = map_type.split('2')
        map_type_ref = map_type_ref.strip()
        map_type_summ = map_type_summ.strip()
        # for ref
        if map_type_ref == 't':
            # token2* mapping
            assert ref_st_mrg_type == 'wAll_sAll'
            final_ref_vecs = filtered_ref_tokens_vecs
            final_ref_weights = filtered_ref_token_weights
        elif map_type_ref == 's':
            # sent2* mapping
            assert 'idf' not in wmd_score_type
            final_ref_vecs = filtered_ref_sents_vecs
            final_ref_weights = filtered_ref_sent_weights
        else:
            # (sent+token)2* mapping
            assert 'idf' not in wmd_score_type
            assert map_type_ref == 'st'
            final_ref_vecs = filtered_ref_mrgd_vecs
            final_ref_weights = filtered_ref_mrgd_weights

        # for summ
        if map_type_summ == 't':
            # *2token mapping
            final_summ_vecs = filtered_summ_tokens_vecs
            final_summ_weights = filtered_summ_token_weights
        elif map_type_summ == 's':
            # *2sent mapping
            assert 'idf' not in wmd_score_type
            final_summ_vecs = filtered_summ_sents_vecs
            final_summ_weights = filtered_summ_sent_weights
        else:
            # *2(sent+token) mapping
            assert 'idf' not in wmd_score_type
            assert map_type_summ == 'st'
            final_summ_vecs = filtered_summ_mrgd_vecs
            final_summ_weights = filtered_summ_mrgd_weights

        # relevance/informativeness score
        relevance_score = get_my_score(final_ref_vecs, final_ref_weights, filtered_ref_tokens,
                                       final_summ_vecs, final_summ_weights, filtered_summ_tokens,
                                       wmd_score_type, wmd_weight_type, beta_gamma=beta_gamma)
        # redundancy score
        redund_score = []
        for i in range(len(filtered_summ_tokens_vecs)):
            redund_score_i = get_my_score([filtered_summ_tokens_vecs[i]], [filtered_summ_token_weights[i]], [filtered_summ_tokens[i]],
                                          [filtered_summ_tokens_vecs[i]], [filtered_summ_token_weights[i]], [filtered_summ_tokens[i]],
                                          wmd_score_type='recall', wmd_weight_type='none', mask_self=True)
            redund_score.append(redund_score_i[0])
        assert len(relevance_score) == len(redund_score)

        # final score
        pss = []
        for i in range(len(relevance_score)):
            if relevance_score[i] is not None and redund_score[i] is not None:
                pss.append((relevance_score[i] - lambda_redund * redund_score[i]) / (1 + lambda_redund))
            else:
                assert relevance_score[i] is None and redund_score[i] is None
                pss.append(None)
        # compute correlation
        # (topic,ss[0].split('/')[-1],human)
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
        if score_saved_file is not None:
            saved_dict = {'topic':topic, 'human_scores': human_scores, 'pseudo_scores': pseudo_scores}
            saved_line = json.dumps(saved_dict) + '\n'
            score_saved_file.write(saved_line)
        if not (np.array(human_scores) == human_scores[0]).all():
            results = evaluateReward(pseudo_scores,human_scores)
            addResult(all_results,results)
            for kk in results:
                print('{}:\t{}'.format(kk,results[kk]))

    print('\n=====ALL Macro RESULTS=====')
    print('year: {}, ref_metric: {}, wmd_score_type: sbert-{}'.format(year,ref_metric,wmd_score_type))
    for kk in all_results:
        if kk.startswith('p_'): continue
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}, significant {} out of {}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk]), len([p for p in all_results['p_{}'.format(kk)] if p<0.05]), len(all_results[kk])))
    if sent_transformer_type != 'bert_large_nli_stsb_mean_tokens':
        print("WARNING: the sentence-transformer is not bert_large_nli_stsb_mean_tokens but {}".format(sent_transformer_type))

    assert len(total_hss) == len(total_pss)
    print('\n=====ALL Micro RESULTS=====')
    results = evaluateReward(total_pss, total_hss)
    for kk in results:
        print('{}:\t{}'.format(kk,results[kk]))

######################################################
#
#
#
#
#   Input: [str, str, str]                      <--- 2 of them, 1 pred, 1 ref 
#   Output: {'score': [float, float, float]}    
#
#####################################################
def score_function(predictions, references):


    # Some variables 
    # These variables directly come from bash input, so i dont know what they mean

    sent_represnt_type = 'mean_all'     # this is fixed dont change
    
    map_type_ref = 'st'                 # IDK what this variable is in the first place,
                                        # choices are t, s, st maybe stands for token, sentence, sentence + token
    map_type_summ = 'st'                # same as above
    
    wmd_score_type = 'f1'               # defaults to f1
                                        # choices are 'f1_beta', 'idf', and 'f1'
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    # Load the Model
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC['bert_large_nli_stsb_mean_tokens']
    bert_model = SentenceTransformer(sent_transformer_path, device=device)      #This SentenceTransformer isnt the default library, this guy modified it
    
    
    # I think this part is getting reference vectors and all that
    #####################################################################################################################################
    
    sent_info_dic, sent_vecs, sents_weights, token_vecs, all_tokens = parse_documents(references,bert_model,'','mean_all')
    ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k] > 0.0}
    ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
    ref_sources = sorted(list(ref_sources))
    
    # get sents in ref/doc
    ref_sents = []
    ref_sents_vecs = []
    ref_sents_weights = []
    ref_tokens_vecs = []
    ref_tokens = []
    sorted_ref_dic_keys = sorted(ref_dic.keys())

    for rs in ref_sources:
        ref_sents.append([ref_dic[k]['text'] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_sents_vecs.append([sent_vecs[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_sents_weights.append([sents_weights[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_tokens_vecs.append([token_vecs[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
        ref_tokens.append([all_tokens[k] for k in sorted_ref_dic_keys if ref_dic[k]['doc'] == rs])
    
    # get the filtered vecs fro ref
    filtered_ref_tokens_vecs = []
    filtered_ref_tokens = []
    filtered_ref_token_weights = []

    # I have no idea what wTop means
    # Assuming no wTop
    ref_st_mrg_type = ''
    if ref_st_mrg_type.startswith('wTop'):
        wTopNum = int(ref_st_mrg_type.split('_')[0].strip()[4:])
        token_level_idx_list = []
        sent_level_idx_list = []
        for doc_weights in ref_sents_weights:
            doc_weights = np.array(doc_weights)
            if doc_weights.all():
                idxs_list = [k for k in range(len(doc_weights))]
            else:
                idxs_list = doc_weights.argsort().tolist()
                idxs_list = idxs_list[::-1]
            token_level_idx_list.append(idxs_list[:wTopNum])
            if ref_st_mrg_type.endswith('sBottom'):
                sent_level_idx_list.append(idxs_list[wTopNum:])
            else:
                assert ref_st_mrg_type.endswith('sAll')
                sent_level_idx_list.append(idxs_list)
    else:
        token_level_idx_list = [[k for k in range(len(doc_weights))] for doc_weights in ref_sents_weights]
        sent_level_idx_list = token_level_idx_list

    for doc_idx in range(len(ref_sents)):
        tvecs_in = [ref_tokens_vecs[doc_idx][k] for k in token_level_idx_list[doc_idx]]
        tokens_in = [ref_tokens[doc_idx][k] for k in token_level_idx_list[doc_idx]]
        # we use sent weight as the weight of each token
        weights_in = [np.array([ref_sents_weights[doc_idx][k]]*len(ref_tokens[doc_idx][k])) for k in token_level_idx_list[doc_idx]]
        vv, tt, ww = get_token_vecs(vecs=tvecs_in, tokens=tokens_in, weights=weights_in)
        filtered_ref_tokens_vecs.append(vv)
        filtered_ref_tokens.append(tt)
        filtered_ref_token_weights.append(ww)
    
    filtered_ref_sents_vecs = []
    filtered_ref_sent_weights = []
    for svec_list, sweights, sent_level_idxs in zip(ref_sents_vecs, ref_sents_weights, sent_level_idx_list):
        remain_svecs = None
        remain_sweights = None
        if len(sent_level_idxs) > 0:
            # remain_svecs = [svec for svec in svec_list[wTopNum:] if svec is not None]
            remain_svecs = [svec_list[k] for k in sent_level_idxs if svec_list[k] is not None]
            remain_sweights = [sweights[k] for k in sent_level_idxs if svec_list[k] is not None]
            if len(remain_svecs) > 0:
                remain_svecs = np.stack(remain_svecs)
                remain_sweights = np.array(remain_sweights)
            else:
                remain_svecs = None
                remain_sweights = None
        filtered_ref_sents_vecs.append(remain_svecs)
        filtered_ref_sent_weights.append(remain_sweights)
    
    #####################################################################################################################################


    # Getting the summary vectors 
    #####################################################################################################################################
    

    

    # get sents in system summaries
    filtered_summ_tokens_vecs = []
    filtered_summ_tokens = []
    filtered_summ_token_weights = []
    filtered_summ_sents_vecs = []
    filtered_summ_sent_weights = []

    # I am assuming bert_model.encode can encode a batch of data at once
    # [pred1, pred2, pred3] where pred are str
    one_summ_sents_vecs, one_summ_tokens_vecs, one_summ_tokens = bert_model.encode(predictions, sent_represnt_type)
    vv, tt, _ = get_token_vecs(vecs=one_summ_tokens_vecs, tokens=one_summ_tokens)
    svv = np.stack([svec for svec in one_summ_sents_vecs if svec is not None])
    tweights = np.ones(tt.shape[0])
    sweights = np.ones(svv.shape[0])

    filtered_summ_tokens_vecs.append(vv)
    filtered_summ_tokens.append(tt)
    filtered_summ_token_weights.append(tweights)
    filtered_summ_sents_vecs.append(svv)
    filtered_summ_sent_weights.append(sweights)

    #####################################################################################################################################

    # get the merged sent and token representations of references
    filtered_ref_mrgd_vecs, filtered_ref_mrgd_weights = mrg_tokens_sents(filtered_ref_tokens_vecs,
                                                                            filtered_ref_token_weights,
                                                                            filtered_ref_sents_vecs,
                                                                            filtered_ref_sent_weights)
    # get the merged sent and token representations of summs
    filtered_summ_mrgd_vecs, filtered_summ_mrgd_weights = mrg_tokens_sents(filtered_summ_tokens_vecs,
                                                                            filtered_summ_token_weights,
                                                                            filtered_summ_sents_vecs,
                                                                            filtered_summ_sent_weights)

    
    if map_type_ref == 't':
        # token2* mapping
        assert ref_st_mrg_type == 'wAll_sAll'
        final_ref_vecs = filtered_ref_tokens_vecs
        final_ref_weights = filtered_ref_token_weights
    elif map_type_ref == 's':
        # sent2* mapping
        assert 'idf' not in wmd_score_type
        final_ref_vecs = filtered_ref_sents_vecs
        final_ref_weights = filtered_ref_sent_weights
    else:
        # (sent+token)2* mapping
        assert 'idf' not in wmd_score_type
        assert map_type_ref == 'st'
        final_ref_vecs = filtered_ref_mrgd_vecs
        final_ref_weights = filtered_ref_mrgd_weights

    # for summ
    if map_type_summ == 't':
        # *2token mapping
        final_summ_vecs = filtered_summ_tokens_vecs
        final_summ_weights = filtered_summ_token_weights
    elif map_type_summ == 's':
        # *2sent mapping
        assert 'idf' not in wmd_score_type
        final_summ_vecs = filtered_summ_sents_vecs
        final_summ_weights = filtered_summ_sent_weights
    else:
        # *2(sent+token) mapping
        assert 'idf' not in wmd_score_type
        assert map_type_summ == 'st'
        final_summ_vecs = filtered_summ_mrgd_vecs
        final_summ_weights = filtered_summ_mrgd_weights



    # relevance/informativeness score
    relevance_score = get_my_score(final_ref_vecs, final_ref_weights, filtered_ref_tokens,
                                    final_summ_vecs, final_summ_weights, filtered_summ_tokens,
                                    wmd_score_type, wmd_weight_type, beta_gamma=beta_gamma)
    # redundancy score
    redund_score = []
    for i in range(len(filtered_summ_tokens_vecs)):
        redund_score_i = get_my_score([filtered_summ_tokens_vecs[i]], [filtered_summ_token_weights[i]], [filtered_summ_tokens[i]],
                                        [filtered_summ_tokens_vecs[i]], [filtered_summ_token_weights[i]], [filtered_summ_tokens[i]],
                                        wmd_score_type='recall', wmd_weight_type='none', mask_self=True)
        redund_score.append(redund_score_i[0])


    # PSS is the final result
    pss = []
    for i in range(len(relevance_score)):
        if relevance_score[i] is not None and redund_score[i] is not None:
            pss.append((relevance_score[i] - lambda_redund * redund_score[i]) / (1 + lambda_redund))


    # Return EVAL_Base type data

    return {'score': pss}
    




if __name__ == '__main__':
    # get the general configuration
    parser = config.ArgumentParser("my_score_metrics.py")
    config.general_args(parser)
    config.pseudo_ref_sim_metrics_args(parser)
    config.pseudo_ref_wmd_metrics_args(parser)
    config.my_metrics_args(parser)
    opt = parser.parse_args()
    print("\nMetric: sbert_score_metrics.py")
    print("Configurations:", opt)
    # '08', '09', '2010', '2011'
    year = opt.year
    ref_summ = opt.ref_summ
    human_metric = opt.human_metric
    ref_metric = opt.ref_metric
    eval_level = opt.evaluation_level
    sent_transformer_type = opt.sent_transformer_type
    bert_type = opt.bert_type
    device = opt.device
    wmd_score_type = opt.wmd_score_type
    wmd_weight_type = opt.wmd_weight_type
    map_type = opt.map_type
    sent_represnt_type = opt.sent_represnt_type
    ref_st_mrg_type = opt.ref_st_mrg_type
    sim_th = opt.sim_th
    lambda_redund = opt.lambda_redund
    pacsum_beta = opt.pacsum_beta
    pacsum_lambda1 = opt.pacsum_lambda1
    pacsum_lambda2 = opt.pacsum_lambda2
    beta_gamma = opt.beta_gamma
    summ_sys_num_limit = opt.summ_sys_num_limit
    doc_num_limit = opt.doc_num_limit
    score_saved_file = opt.score_saved_file
    run_my_score_metrics(year=year, ref_metric=ref_metric,
                         wmd_score_type=wmd_score_type,
                         wmd_weight_type=wmd_weight_type,
                         eval_level=eval_level,
                         human_metric=human_metric,
                         sent_transformer_type=sent_transformer_type,
                         map_type=map_type,
                         sent_represnt_type=sent_represnt_type,
                         ref_st_mrg_type=ref_st_mrg_type,
                         sim_th=sim_th,
                         lambda_redund=lambda_redund,
                         pacsum_beta=pacsum_beta,
                         pacsum_lambda1=pacsum_lambda1,
                         pacsum_lambda2=pacsum_lambda2,
                         beta_gamma=beta_gamma,
                         summ_sys_num_limit=summ_sys_num_limit,
                         doc_num_limit=doc_num_limit,
                         score_saved_file=score_saved_file,
                         device=device)