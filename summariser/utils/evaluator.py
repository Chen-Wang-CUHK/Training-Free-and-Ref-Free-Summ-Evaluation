import sys
sys.path.append('../')

# from rouge.rouge import Rouge
from resources import *
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr, kendalltau


def add_result(all_dic,result):
    for metric in result:
        if metric in all_dic:
            all_dic[metric].append(result[metric])
        else:
            all_dic[metric] = [result[metric]]


# def evaluate_summary_rouge(cand,model,max_sum_len=100):
#     rouge_scorer = Rouge(ROUGE_DIR,BASE_DIR,True)
#     r1, r2, rl, rsu4 = rouge_scorer(cand,[model],max_sum_len)
#     rouge_scorer.clean()
#     dic = OrderedDict()
#     dic['ROUGE-1'] = r1
#     dic['ROUGE-2'] = r2
#     dic['ROUGE-L'] = rl
#     dic['ROUGE-SU4'] = rsu4
#     return dic


def addResult(all_dic,result):
    for metric in result:
        if metric in all_dic:
            all_dic[metric].append(result[metric])
        else:
            all_dic[metric] = [result[metric]]


def evaluateReward(learnt_scores, human_scores):
    """
    Create by wchen to run the code smoothly
    :param learnt_scores:
    :param human_scores:
    :return:
    """
    results = {}
    # compute the Pearson's correlation coefficients
    pearson_r, p_r = pearsonr(learnt_scores, human_scores)
    results['pearson_r'] = pearson_r
    results['p_pearson_r'] = p_r

    # compute the Spearman's correlation coefficients
    spearman_rho, p_rho = spearmanr(learnt_scores, human_scores)
    results['spearman_rho'] = spearman_rho
    results['p_spearman_rho'] = p_rho

    # compute the Kendall's correlation coefficients
    kendall_tau, p_tau = kendalltau(learnt_scores, human_scores)
    results['kendall_tau'] = kendall_tau
    results['p_kendall_tau'] = p_tau
    return results