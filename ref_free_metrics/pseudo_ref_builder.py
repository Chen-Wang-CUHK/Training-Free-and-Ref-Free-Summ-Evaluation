from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation


def get_doc_simtop(sim_matrix, max_sim_value):
    nn = sim_matrix.shape[0]
    for i in range(1,sim_matrix.shape[0]):
        if np.max(sim_matrix[i][:i])>max_sim_value: 
            nn = i
            break
    return nn


def get_top_sim_weights(sent_info, full_vec_list, max_sim_value):
    doc_names = set([sent_info[k]['doc'] for k in sent_info])
    weights = [0.]*len(sent_info)
    for dn in doc_names:
        doc_idx = [k for k in sent_info if sent_info[k]['doc']==dn]
        sim_matrix = cosine_similarity(np.array(full_vec_list)[doc_idx], np.array(full_vec_list)[doc_idx])
        nn = get_doc_simtop(sim_matrix, max_sim_value)
        for i in range(np.min(doc_idx),np.min(doc_idx)+nn): weights[i] = 1.
    return weights


def get_top_weights(sent_index, topn):
    weights = []
    for i in range(len(sent_index)):
        if sent_index[i]['inside_doc_idx'] < topn:
            weights.append(1.)
        else:
            weights.append(0.)
    return weights


def get_subgraph(sim_matrix, threshold):
    gg = nx.Graph()
    for i in range(0,sim_matrix.shape[0]-1):
        for j in range(i+1,sim_matrix.shape[0]):
            if sim_matrix[i][j] >= threshold:
                gg.add_node(i)
                gg.add_node(j)
                gg.add_edge(i,j)
    subgraph = list(nx.connected_component_subgraphs(gg))
    subgraph_nodes = [list(sg._node.keys()) for sg in subgraph]
    return list(subgraph_nodes)


def get_other_weights(full_vec_list, sent_index, weights, thres):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    subgraphs = get_subgraph(similarity_matrix, thres)
    '''
    top_sent_idx = [i for i in range(len(weights)) if weights[i]>0.9]
    for sg in subgraphs:
        if len(set([sent_index[n]['doc'] for n in sg])) < 2: continue #must appear in multiple documents
        for n in sg: weights[n]=1./len(sg)
    '''
    for sg in subgraphs:
        if any(weights[n]>=0.9 for n in sg): continue #ignore the subgraph similar to a top sentence
        if len(set([sent_index[n]['doc'] for n in sg])) < 2: continue #must appear in multiple documents
        for n in sg: weights[n]=1./len(sg)
        #print(sg,'added to weights')


def graph_centrality_weight(similarity_matrix):
    weights_list = [np.sum(similarity_matrix[i])-1. for i in range(similarity_matrix.shape[0])]
    return weights_list

# some code is borrowed from PacSum
def pacsum_compute_scores(similarity_matrix, edge_threshold):
    forward_scores = [0 for _ in range(len(similarity_matrix))]
    backward_scores = [0 for _ in range(len(similarity_matrix))]
    edges = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix[i])):
            edge_score = similarity_matrix[i][j]
            if edge_score > edge_threshold:
                forward_scores[j] += edge_score
                backward_scores[i] += edge_score
                edges.append((i,j,edge_score))
    return np.asarray(forward_scores), np.asarray(backward_scores), edges


def graph_pacsum_centrality_weights(similarity_matrix, pacsum_beta=0.0, pacsum_lambda1=2.0, pacsum_lambda2=1.0):
    min_score = similarity_matrix.min()
    max_score = similarity_matrix.max()
    edge_threshold = min_score + pacsum_beta * (max_score - min_score)
    new_edge_scores = similarity_matrix - edge_threshold
    forward_scores, backward_scores, _ = pacsum_compute_scores(new_edge_scores, 0)
    forward_scores = 0 - forward_scores

    node_scores = []
    for node in range(len(forward_scores)):
        node_scores.append(pacsum_lambda1 * forward_scores[node] + pacsum_lambda2 * backward_scores[node])
    return node_scores


def graph_weights(full_vec_list):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    weights_list = graph_centrality_weight(similarity_matrix)
    return weights_list


def graph_pacsum_weights(full_vecs, pacsum_beta=0.0, pacsum_lambda1=2.0, pacsum_lambda2=1.0):
    similarity_matrix = cosine_similarity(full_vecs, full_vecs)
    # full_vecs = np.array(full_vecs)
    # similarity_matrix = np.matmul(full_vecs, full_vecs.T)
    weights_list = graph_pacsum_centrality_weights(similarity_matrix, pacsum_beta=pacsum_beta, pacsum_lambda1=pacsum_lambda1, pacsum_lambda2=pacsum_lambda2)
    return weights_list


def get_indep_pacsum_weights(sent_info_dic, sent_vecs, num=0, pacsum_beta=0.0, pacsum_lambda1=2.0, pacsum_lambda2=1.0):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])
    doc_names = sorted(list(doc_names))
    # set the none element as zero vector
    for svec in sent_vecs:
        if svec is not None:
            svec_shape = svec.shape
            break
    sent_vecs = [svec if svec is not None else np.zeros(svec_shape) for svec in sent_vecs]
    weights = np.zeros(len(sent_vecs))
    for dname in doc_names:
        ids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc'] == dname])
        if len(ids) > num:
            doc_weights = np.array(graph_pacsum_weights(np.array(sent_vecs)[ids], pacsum_beta=pacsum_beta, pacsum_lambda1=pacsum_lambda1, pacsum_lambda2=pacsum_lambda2))
            sorted_idxs = doc_weights.argsort()
            sorted_idxs = sorted_idxs[::-1]
            if num == 0:
                selected_idxs = sorted_idxs
            else:
                selected_idxs = sorted_idxs[:num]
            selected_doc_weights = doc_weights[selected_idxs]
            # normalize the weights within selected document sents, if directly use the following selected_doc_weights, the minimum one will be filtered
            assert (selected_doc_weights.max() - selected_doc_weights.min()) > 0
            selected_doc_weights = (selected_doc_weights - selected_doc_weights.min()) / (selected_doc_weights.max() - selected_doc_weights.min())
            # method1 divide the interval
            # selected_doc_weights = selected_doc_weights / selected_doc_weights.sum()
            # method2 softmax
            e_weights = np.exp(selected_doc_weights)
            selected_doc_weights = e_weights / e_weights.sum()

            wanted_id = ids[selected_idxs]
            assert (selected_doc_weights > 0).all()
            weights[wanted_id] = selected_doc_weights
        else:
            weights[ids] = 1.0 / len(ids)
    return weights


def get_indep_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_ratio):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])

    # add by wchen for none svec
    # set the none element as zero vector
    for svec in sent_vecs:
        if svec is not None:
            svec_shape = svec.shape
            break
    sent_vecs = [svec if svec is not None else np.zeros(svec_shape) for svec in sent_vecs]

    wanted_id = []
    for dname in doc_names:
        ids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc']==dname])
        doc_weights = np.array(graph_weights(np.array(sent_vecs)[ids]))
        if top_n is not None: 
            for j in range(top_n): 
                if j>=len(doc_weights): break
                doc_weights[j] *= extra_ratio
        wanted_id.extend(list(ids[doc_weights.argsort()[-num:]]))
    weights = [0.]*len(sent_vecs)
    for ii in wanted_id: weights[ii] = 1.
    return weights


def get_global_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_ratio):
    # add by wchen for none svec
    # set the none element as zero vector
    for svec in sent_vecs:
        if svec is not None:
            svec_shape = svec.shape
            break
    sent_vecs = [svec if svec is not None else np.zeros(svec_shape) for svec in sent_vecs]

    raw_weights = graph_weights(sent_vecs)
    if top_n is not None:
        top_ids = [i for i in sent_info_dic if sent_info_dic[i]['inside_doc_idx']<top_n]
        adjusted_weights = [w*extra_ratio if j in top_ids else w for j,w in enumerate(raw_weights) ]
    else:
        adjusted_weights = raw_weights
    wanted_id = np.array(adjusted_weights).argsort()[-num:]
    weights = [0.] * len(sent_vecs)
    for ii in wanted_id: weights[ii] = 1.
    return weights


def get_indep_cluster_weights(sent_info_dic, sent_vecs):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])
    sums = [np.sum(sv) for sv in sent_vecs]
    wanted_ids = []
    for dname in doc_names:
        sids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc']==dname])
        clustering = AffinityPropagation().fit(np.array(sent_vecs)[sids])
        centers = clustering.cluster_centers_
        for cc in centers: wanted_ids.append(sums.index(np.sum(cc)))
    print('indep cluster, pseudo-ref sent num', len(wanted_ids))
    weights = [1. if i in wanted_ids else 0. for i in range(len(sent_vecs))]
    return weights


def get_global_cluster_weights(sent_vecs):
    clustering = AffinityPropagation().fit(sent_vecs)
    centers = clustering.cluster_centers_
    print('global cluster, pseudo-ref sent num', len(centers))
    sums = [np.sum(sv) for sv in sent_vecs]
    ids = []
    for cc in centers: ids.append(sums.index(np.sum(cc)))
    assert len(ids) == len(centers)
    weights = [1. if i in ids else 0. for i in range(len(sent_vecs))]
    return weights

'''
def give_top_extra_weights(weights, sent_info_dic, top_n, extra_ratio):
    for ii in sent_info_dic:
        if sent_info_dic[ii]['inside_doc_idx']<top_n: weights[ii]*extra_ratio
'''





