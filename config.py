import configargparse as cfargparse

ArgumentParser = cfargparse.ArgumentParser


def general_args(parser):
    """
    add some general options
    """
    group = parser.add_argument_group("general_args")
    group.add('--year', '-year', type=str, default='2010',
              choices=['08', '09', '2010', '2011', 'cnndm'],
              help="The year of the TAC data")
    group.add('--human_metric', '-human_metric', type=str, default='pyramid',
              choices=['scu','pyramid','lingustic','responsiveness', 'overall', 'grammar', 'redundancy'],
              help="The selected human metric which is used to computer correlations with the designed auto metrics")
    group.add('--ref_summ', '-ref_summ', action='store_true',
              help="use the gold references")
    group.add('--evaluation_level', '-evaluation_level', type=str, default='summary',
              choices=['summary', 'micro', 'system', 'macro'],
              help="The level to evaluate the summarization systems")
    group.add('--device', '-device', type=str, default='cpu', choices=["cpu", "cuda"],
              help="The selected device to run the code")


def pseudo_ref_sim_metrics_args(parser):
    '''
    add some options for bert_metrics, elmo_metrics, js_metrics, sbert_score_metrics
    '''
    group = parser.add_argument_group("pseudo_ref_sim_metrics_args")
    group.add('--ref_metric', '-ref_metric', type=str, default='top12',
              help="TopN_th: Top 'N' sentences are selected as the reference and 'th' is the threshold. If no 'th', then directly use top N sents.")
    group.add('--bert_type', '-bert_type', type=str, default='bert',
              choices=['bert', 'albert', 'roberta_large_mnli', 'roberta_large_openai_detector', 'roberta_large'],
              help="The pretrained model used to encoding the tokens of summaries.")
    group.add('--sent_transformer_type', '-sent_transformer_type', type=str, default='bert_large_nli_stsb_mean_tokens',
              choices=['bert_base_nli_stsb_mean_tokens', 'bert_large_nli_stsb_mean_tokens',
                       'roberta_base_nli_stsb_mean_tokens', 'roberta_large_nli_stsb_mean_tokens',
                       'distilbert_base_nli_stsb_mean_tokens'],
              help="The pretrained sentence transformer used to encoding the sentences of the documents.")


def pseudo_ref_wmd_metrics_args(parser):
    '''
    add some options for bert_metrics, elmo_metrics, js_metrics, sbert_score_metrics
    '''
    group = parser.add_argument_group("pseudo_ref_wmd_metrics_args")
    group.add('--wmd_score_type', '-wmd_score_type', type=str, default='f1', choices=['f1', 'precision', 'recall', 'f1_beta'],
              help="The type of word-mover-distance score")
    group.add('--wmd_weight_type', '-wmd_weight_type', type=str, default='none', choices=['global_idf_renormalize', 'idf_renormalize', 'none', 'graph_weighted_renormalize'],
              help="The type of word-mover-distance weight")


def pseudo_ref_file_generator_args(parser):
    '''
    add some options for building the reference file
    '''
    group = parser.add_argument_group("pseudo_ref_file_generator_args")
    group.add('--style_type', '-style_type', type=str, default='moverscore',
              choices=['moverscore', 'sms'],
              help='The style type of the generated pseudo reference file.')
    group.add('--year', '-year', type=str, default='2010',
              choices=['08', '09', '2010', '2011', 'cnndm'],
              help="The year of the TAC data")
    group.add('--ref_metric', '-ref_metric', type=str, default='top12',
              help="topN_th: Top 'N' sentences are selected as the reference and 'th' is the threshold. If no 'th', then directly use top N sents."
                   "'true_ref': use the gold summaries as the references. 'full_doc': use full doc as the pseudo ref."
                   "'full_doc_wtopN_th': use full doc as the pseudo ref but the topN_th sentences use word-level representations and the others use sentence level representation")
    group.add('--ref_summ', '-ref_summ', action='store_true',
              help="Ignore the ref_metric and use the gold references")
    group.add('--sent_transformer_type', '-sent_transformer_type', type=str, default='bert_large_nli_stsb_mean_tokens',
              choices=['bert_base_nli_stsb_mean_tokens', 'bert_large_nli_stsb_mean_tokens',
                       'roberta_base_nli_stsb_mean_tokens', 'roberta_large_nli_stsb_mean_tokens',
                       'distilbert_base_nli_stsb_mean_tokens'],
              help="The pretrained sentence transformer used to encoding the sentences of the documents.")
    group.add('--evaluation_level', '-evaluation_level', type=str, default='summary',
              choices=['summary', 'micro', 'system', 'macro'],
              help="The level to evaluate the summarization systems")
    group.add('--device', '-device', type=str, default='cpu', choices=["cpu", "cuda"],
              help="The selected device to run the code")
    group.add('--add_gold_refs_to_sys_summs', '-add_gold_refs_to_sys_summs', action='store_true',
              help='Add the gold refs as part of the system summaries')


def my_metrics_args(parser):
    '''
    add some options for my_metrics
    '''
    group = parser.add_argument_group("my_metrics_args")
    group.add('--map_type', '-map_type', type=str, default='t2t', choices=['t2t', 't2s',  't2st', 's2t', 's2s', 's2st', 'st2t', 'st2s', 'st2st'],
              help="The type of the mapping type between the source input and sys summs")
    group.add('--sent_represnt_type', '-sent_represnt_type', type=str, default='max_all', choices=['mean_all', 'mean_words', 'mean_nstpwd', 'max_all', 'max_words', 'max_nstpwd', 'CLS', 'SEP'],
              help='The type of the sentence representation.')
    group.add('--ref_st_mrg_type', '-ref_st_mrg_type', type=str, default='wAll_sAll',
              help="'wAll_sAll': merge all the word-level representations and sentence-level representations."
                   "'wTopN_sBottom': the top N sentences use word-level representations, the remaining sentences use sent-level representations."
                   "'wTopN_sAll': the top N sentences use word-level representations, all the sentences use sent-level representations.")
    group.add('--sim_th', '-sim_th', type=float, default=0.0,
              help='The similarity threshold. Follow SUPERT, we use 0.75 as the defualt.')
    group.add('--lambda_redund', '-lambda_redund', type=float, default=0.0,
              help='The weight of the redundancy score.')
    group.add('--pacsum_beta', '-pacsum_beta', type=float, default=0.0,
              help='The beta of pacsum')
    group.add('--pacsum_lambda1', '-pacsum_lambda1', type=float, default=2.0,
              help='The pacsum_lambda1 of pacsum')
    group.add('--pacsum_lambda2', '-pacsum_lambda2', type=float, default=1.0,
              help='The pacsum_lambda2 of pacsum')
    group.add('--beta_gamma', '-beta_gamma', type=int, default=2,
              help='The gamma hyperparameter for the value of beta square.')
    group.add('--summ_sys_num_limit', '-summ_sys_num_limit', type=int, default=-1,
              help='The limit of the number of the summarization system.')
    group.add('--doc_num_limit', '-doc_num_limit', type=int, default=-1,
              help='The limit of the number of the document for each topic.')
    group.add('--score_saved_file', '-score_saved_file', type=str, default=None,
              help='The file to save human scores and predicted scores.')



def rouge_metric_args(parser):
    '''
    add some options for rouge metric
    '''
    group = parser.add_argument_group("rouge_metric_args")
    group.add('--rouge_metric', '-rouge_metric', type=str, default='rouge_1',
              choices=['rouge_1', 'rouge_2', 'rouge_3', 'rouge_4', 'rouge_l', 'rouge_w_1.2', 'rouge_s*', 'rouge_su*'],
              help="The selected rouge metric which is used to computer correlations with the human metrics")
