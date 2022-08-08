import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

import string
import copy
from nltk.corpus import stopwords
from resources import LANGUAGE


def get_human_score(topic, summ_name, human, year):
    if year != 'cnndm':
        block = summ_name.split('-')[1].split('.')[0]
        id = summ_name.split('.')[-1]
        key = 'topic{}-{}_sum{}'.format(topic.split('.')[0],block,id)
    else:
        key = 'topic{}_sum_{}'.format(topic, summ_name)
    if key not in human:
        return None
    else:
        return human[key]

def get_idf_weights(ref_vecs):
    sim_matrix = cosine_similarity(ref_vecs,ref_vecs)
    #dfs = [np.sort(sim_matrix[i])[-2] for i in range(len(ref_vecs))]
    dfs = [np.sum(sim_matrix[i])-1. for i in range(len(ref_vecs))]
    dfs = [1.*d/(len(ref_vecs)-1) for d in dfs]
    dfs = [(d+1.)/2. for d in dfs]
    idf = [-1.*np.log(df) for df in dfs]
    return idf


def get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic):
    ref_dic = {}
    docs = set([info_dic[k]['doc'] for k in info_dic])
    for dd in docs:
        ref_dic[dd] = [i for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>=0.1 and info_dic[i]['doc']==dd]
    vecs = []
    for dd in ref_dic:
        allv = np.array(doc_sent_vecs)[ref_dic[dd]]
        meanv = np.mean(allv,axis=0)
        vecs.append(meanv)
    return vecs


def get_sim_metric(summ_vec_list, doc_sent_vecs, doc_sent_weights, info_dic, method='1'):
    #print('weights', doc_sent_weights)
    # method 1: get the avg doc vec, then cosine
    if method == '1':
        summ_vec = np.mean(np.array(summ_vec_list),axis=0)
        dvec = np.matmul(np.array(doc_sent_weights).reshape(1,-1),  np.array(doc_sent_vecs))
        return cosine_similarity(dvec,summ_vec.reshape(1,-1))[0][0]
        # below: good performance with true_ref, poorer performance with other pseduo-refs
        #ref_vecs = get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic)
        #sims = cosine_similarity(np.array(ref_vecs), np.array(summ_vec).reshape(1,-1))
        #return np.mean(sims)

    # method 2: cosine between each doc and the summ, then avg
    elif method == '2':
        summ_vec = np.mean(np.array(summ_vec_list),axis=0)
        sim_matrix = cosine_similarity(np.array(doc_sent_vecs),summ_vec.reshape(1,-1))
        mm = np.matmul(np.array(sim_matrix).reshape(1,-1),np.array(doc_sent_weights)).reshape(-1,1)[0][0]
        return mm

    else:
        ref_vecs = [doc_sent_vecs[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        #ref_vecs = get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic)
        weights = [doc_sent_weights[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        idf_weights = get_idf_weights(ref_vecs)
        sim_matrix = cosine_similarity(np.array(ref_vecs),np.array(summ_vec_list))
        recall = np.mean(np.max(sim_matrix,axis=1))
        idf_recall = np.dot(np.max(sim_matrix,axis=1),idf_weights)/np.sum(idf_weights)
        precision = np.mean(np.max(sim_matrix,axis=0))
        if recall+precision == 0:
            f1 = None
        else:
            f1 = 2.*recall*precision/(recall+precision)
            idf_f1 = 2.*idf_recall*precision/(idf_recall+precision)
        if method.lower().startswith('f'): return f1
        elif method.lower().startswith('r'): return recall
        elif method.lower().startswith('p'): return precision
        elif method.lower().startswith('idf'):
            if 'recall' in method: return idf_recall
            elif 'f1' in method: return idf_f1
            else: return None
        elif method.lower().startswith('w'): return np.dot(np.array(np.max(sim_matrix,axis=1)),np.array(weights))/np.sum(weights)
        else: return None


def parse_docs(docs,bert_model,sent_represnt_type):
    all_sents = []
    sent_index = {}
    cnt = 0
    for dd in docs:
        # changed by wchen to adopt to both Linux and Windows machine
        # dname = dd[0].split('/')[-1]
        dname = os.path.basename(dd[0])

        doc_len = len(dd[1])
        for i, sent in enumerate(dd[1]):
            sent_index[cnt] = {'doc': dname, 'text': sent, 'inside_doc_idx': i, 'doc_len': doc_len,
                               'inside_doc_position_ration': i * 1. / doc_len}
            cnt += 1
            all_sents.append(sent)
    all_sent_vecs = None
    all_token_vecs = None
    all_tokens = None
    if bert_model is not None:
        all_sent_vecs, all_token_vecs, all_tokens = bert_model.encode(all_sents,sent_represnt_type)
    return sent_index, all_sent_vecs, all_token_vecs, all_tokens


def parse_refs(refs,bert_model,sent_represnt_type):
    all_sents = []
    sent_index = {}
    cnt = 0
    for i,rr in enumerate(refs):
        if len(rr[1]) == 1: # one piece of text
            ref_sents = sent_tokenize(rr[1][0])
        else: # a list of sentences
            ref_sents = rr[1]
        ref_name = 'ref{}'.format(i)
        for j, sent in enumerate(ref_sents):
            sent_index[cnt] = {'doc': ref_name, 'text': sent, 'inside_doc_idx': j, 'doc_len': len(ref_sents),
                               'inside_doc_position_ration': j * 1. / len(ref_sents)}
            cnt += 1
            all_sents.append(sent)
    all_sent_vecs = None
    all_token_vecs = None
    all_tokens = None
    if bert_model is not None:
        all_sent_vecs, all_token_vecs, all_tokens = bert_model.encode(all_sents,sent_represnt_type)
    return sent_index, all_sent_vecs, all_token_vecs, all_tokens


def replace_xml_special_tokens_and_preprocess(fpath, line):
    line = line.strip().replace('&lt;', '<').replace('&gt;', '>').replace('&apos;', "'")
    line = line.replace('&amp;amp;', '&').replace('&amp;', '&').replace('&amp ;', '&')
    line = line.replace('&quot;', '"').replace('&slash;', '/')
    special_tokens = re.search("&[a-z]*?;", line)
    assert special_tokens == None, "\nFile path: {}\nThis line contains special tokens {}:\n{}".format(
        fpath, special_tokens, line)
    line = preprocess_one_line(line)
    return line


def preprocess_one_line(line):
    line = line.strip()
    orig_words = line.split()
    new_words = []
    for w in orig_words:
        # 1. remove "___"
        if len(w) >= 2 and all([c == '_' for c in w]):
            continue
        # 2. "insurgent.Four" --> " insurgent. Four"
        posis = [p.start() for p in re.finditer("\.",w)]
        if len(posis) == 1 and w[-1] != '.' and w[posis[0]+1].isupper() and len(w) > 3:
            w = w.replace('.', '. ')
        new_words.append(w)
    new_line = ' '.join(new_words)
    return new_line


def get_token_vecs(vecs, tokens, weights=None, remove_stopwords=True):
    if len(tokens) == 0: return None, None, None
    full_weights = None
    for i, rtv in enumerate(vecs):
        if i==0:
            full_vec = rtv
            full_token = copy.deepcopy(tokens[i])
        else:
            full_vec = np.row_stack((full_vec, rtv))
            full_token.extend(tokens[i])
    assert len(full_token) == full_vec.shape[0]
    if weights is not None:
        full_weights = np.concatenate(weights)
        assert len(full_token) == full_weights.shape[0]

    if remove_stopwords:
        mystopwords = list(set(stopwords.words(LANGUAGE)))
        mystopwords.extend(list(string.punctuation))
        mystopwords.extend(['[cls]','[sep]'])
        wanted_idx = [j for j,tk in enumerate(full_token) if tk.lower() not in mystopwords and '##' not in tk]
    else:
        wanted_idx = [k for k in range(len(full_token))]
    selected_full_weights = full_weights[wanted_idx] if full_weights is not None else None
    return full_vec[wanted_idx], np.array(full_token)[wanted_idx], selected_full_weights


def get_sent_vecs(tokens_vecs, tokens, sent_represnt_type='mean_all', no_stack=False):
    pool_method = np.mean if 'mean' in sent_represnt_type else np.max
    filtered_sents_vecs = []
    for doc_idx, one_doc_tvecs in enumerate(tokens_vecs):
        if len(one_doc_tvecs) == 0:
            filtered_sents_vecs.append(None)
            continue
        if sent_represnt_type == 'mean_all' or sent_represnt_type == 'max_all':
            filtered_one_doc_svecs = [pool_method(tvec, axis=0) for tvec in one_doc_tvecs]
        elif sent_represnt_type == 'mean_words' or sent_represnt_type == 'max_words':
            # do not consider the CLS and SEP tokens
            filtered_one_doc_svecs = [pool_method(tvec[1:-1], axis=0) for tvec in one_doc_tvecs]
        elif sent_represnt_type == 'mean_nstpwd' or sent_represnt_type == 'max_nstpwd':
            # only consider the non-stop-words
            filtered_one_doc_svecs = []
            for sent_idx, sent_tvecs in enumerate(one_doc_tvecs):
                vv, tt, _ = get_token_vecs(vecs=[one_doc_tvecs[sent_idx]], tokens=[tokens[doc_idx][sent_idx]])
                if vv.shape[0] != 0:
                    filtered_one_doc_svecs.append(pool_method(vv, axis=0))
                elif no_stack:
                    filtered_one_doc_svecs.append(None)
        elif sent_represnt_type == 'CLS':
            # only consider the CLS tokens
            filtered_one_doc_svecs = [tvec[0] for tvec in one_doc_tvecs]
        elif sent_represnt_type == 'SEP':
            # only consider the SEP tokens
            filtered_one_doc_svecs = [tvec[-1] for tvec in one_doc_tvecs]

        if no_stack:
            filtered_sents_vecs.append(filtered_one_doc_svecs)
        else:
            filtered_sents_vecs.append(np.stack(filtered_one_doc_svecs))
    return filtered_sents_vecs


def filter_remain_sents_vecs(docs_sents_vecs, wTopW, sim_th):
    filtered_docs_sents_vecs = []
    filtered_cnt = 0
    total_remain_sent_num = 0
    for doc_idx, sents_vecs in enumerate(docs_sents_vecs):
        total_remain_sent_num += len(sents_vecs[wTopW:])
        if len(sents_vecs) > wTopW:
            for s_idx in range(wTopW, len(sents_vecs)):
                if sents_vecs[s_idx] is None: continue
                # filter if it is too similar to its former sentences
                ref_sents_vecs_list = [svec for svec in sents_vecs[:s_idx] if svec is not None]
                filter_condition = (cosine_similarity([sents_vecs[s_idx]], ref_sents_vecs_list) >= sim_th).any()
                if filter_condition:
                    sents_vecs[s_idx] = None
                    filtered_cnt += 1
        filtered_docs_sents_vecs.append(sents_vecs)
    # for debug
    filtered_ratio = filtered_cnt/total_remain_sent_num if total_remain_sent_num != 0 else 0
    print('Filtered document sentence number: {}, ratio: {:.3f}, sim_th: {}'.format(filtered_cnt, filtered_ratio, sim_th))
    return filtered_docs_sents_vecs


