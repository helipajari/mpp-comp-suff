
import itertools
import functools
import logging
import warnings
from typing import Callable, List

import numpy as np
import scipy
import sklearn
import sklearn.metrics
# from graph_attribution import graphs as graph_utils


def silent_nan_np(f):
    """Decorator that silences np errors and returns nan if undefined.
    The np.nanmax and other numpy functions will log RuntimeErrors when the
    input is only nan (e.g. All-NaN axis encountered). This decorator silences
    these messages.
    Args:
      f: function to decorate.
    Returns:
      Variant of function that will be silent to invalid numpy errors, and with
      np.nan when metric is undefined.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                return f(*args, **kwargs)
            except (ValueError, sklearn.exceptions.UndefinedMetricWarning):
                return np.nan
    return wrapper


accuracy_score = sklearn.metrics.accuracy_score
nanmax = silent_nan_np(np.nanmax)
nanmean = silent_nan_np(np.nanmean)
nan_auroc_score = silent_nan_np(sklearn.metrics.roc_auc_score)
nan_precision_score = silent_nan_np(sklearn.metrics.precision_score)
nan_f1_score = silent_nan_np(sklearn.metrics.f1_score)


def nodewise_metric(f):
    """Wrapper to apply a metric computation to each nodes vector in a heatmap.
    For example, given a function `auroc` that computes AUROC over a pair
    (y_true, y_pred) for a binary classification task,
    '_heatmapwise_metric(auroc)' will compute a AUROC for each nodes heatmap in
    a list.
    Args:
      f: A function taking 1-D arrays `y_true` and `y_pred` of shape
        [num_examples] and returning some metric value.
    Returns:
      A function taking 2-D arrays `y_true` and `y_pred` of shape [num_examples,
      num_output_classes], returning an array of shape [num_output_classes].
    """

    def vectorized_f(y_true, y_pred, *args, **kwargs):
        n = len(y_true)
        values = [
            f(y_true[i].nodes, y_pred[i].nodes, *args, **kwargs) for i in range(n)
        ]
        return np.array(values)

    return vectorized_f


def _validate_attribution_inputs(y_true,
                                 y_pred):
    """Helper function to validate that attribution metric inputs are good."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f'Expected same number of graphs in y_true and y_pred, found {len(y_true)} and {len(y_pred)}'
        )
    # for att_true in y_true:
    #     node_shape = att_true.shape
    #     if len(node_shape) != 2:
    #         raise ValueError(
    #             f'Expecting 2D nodes for true attribution, found at least one with shape {node_shape}'
    #         )


def attribution_metric(f):
    """Wrapper to apply a 'attribution' style metric computation to each graph.
    For example, given a function `auroc` that computes AUROC over a pair
    (y_true, y_pred) for a binary classification task,
    '_attribution_metric(auroc)' will compute a AUROC for each graph in
    a list.
    Args:
      f: A function taking 1-D arrays `y_true` and `y_pred` of shape
        [num_examples] and returning some metric value.
    Returns:
      A function taking 2-D arrays `y_true` and `y_pred` of shape [num_examples,
      num_output_classes], returning an array of shape [num_output_classes].
    """

    def vectorized_f(y_true, y_pred, *args, **kwargs):
        _validate_attribution_inputs(y_true, y_pred)
        values = []
        for att_true, att_pred in zip(y_true, y_pred):
            values.append(f(att_true, att_pred, *args, **kwargs))
        return np.array(values)

    return vectorized_f


def kendall_tau_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's tau rank correlation, used for relative orderings."""
    return scipy.stats.kendalltau(y_true, y_pred).correlation


def pearson_r_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson's r for linear correlation."""
    r, _ = scipy.stats.pearsonr(y_true, y_pred)
    return r[0] if hasattr(r, 'ndim') and r.ndim == 1 else r


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))


nodewise_f1_score = nodewise_metric(nan_f1_score)
nodewise_kendall_tau_score = nodewise_metric(kendall_tau_score)
nodewise_pearson_r_score = nodewise_metric(pearson_r_score)

attribution_auroc = attribution_metric(nan_auroc_score)
attribution_accuracy = attribution_metric(accuracy_score)
attribution_f1 = attribution_metric(nan_f1_score)
attribution_precision = attribution_metric(nan_precision_score)



def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))

def attribution_accuracy_mean(y_true, y_pred):
    _validate_attribution_inputs(y_true, y_pred)
    y_true = itertools_chain(y_true)
    y_pred = itertools_chain(y_pred)
    return accuracy_score(y_true, y_pred)

def attribution_auroc_mean(y_true, y_pred):
    _validate_attribution_inputs(y_true, y_pred)
    y_true = itertools_chain(y_true)
    y_pred = itertools_chain(y_pred)
    return nan_auroc_score(y_true, y_pred)


def get_optimal_threshold(y_true,
                          y_prob,
                          grid_spacing=0.01,
                          verbose=False,
                          multi=False):
    """For probabilities, find optimal threshold according to f1 score.
    For a set of groud truth labels and predicted probabilities of these labels,
    performs a grid search over several probability thresholds. For each threshold
    f1_score is computed and the threshold that maximizes said metric is returned.
    If multiple maxmium scores are possible, we pick the median of these
    thresholds.
    Arguments:
      y_true (np.array): 1D array with true labels.
      y_prob (np.array): 1D array with predicted probabilities.
      grid_spacing (float): controls the spacing for the grid search, should be a
        positive value lower than 1.0 . Defaults to 0.01.
      verbose (bool): flag to print values.
    Returns:
      p_threshold (float): Probability threshold.
    """
    thresholds = np.arange(grid_spacing, 1.0, grid_spacing)
    scores = []
    for t in thresholds:
        if multi:
            y_preds = [np.array([1 if att>t else -1 if att<(-t) else 0 for att in att_prob]) for att_prob in y_prob]
        else:
            y_preds = [np.array([1 if att>t else 0 for att in att_prob]) for att_prob in y_prob]
        # scores.append(np.nanmean(attribution_precision(y_true, y_preds)))
        # scores.append(np.nanmean(nan_f1_score(y_true, y_preds)))
        scores.append(np.nanmean(attribution_accuracy(y_true, y_preds)))
    scores = np.array(scores)
    max_thresholds = thresholds[scores == nanmax(scores)]
    p_threshold = np.median(max_thresholds)
    if verbose:
        logging.info('Optimal p_threshold is %.2f', p_threshold)
    return p_threshold


# def get_opt_binary_attributions(atts_true,
#                                 atts_pred,
#                                 metric=nodewise_f1_score,
#                                 n_steps=20):
#     """Binarize attributions according to a threshold."""

#     thresholds = np.linspace(0, 1, num=n_steps)
#     scores = []
#     for thres in thresholds:
#         atts = [graph_utils.binarize_np_nodes(g, thres) for g in atts_pred]
#         scores.append(nanmean(metric(atts_true, atts)))
#     scores = np.array(scores)
#     max_thresholds = thresholds[scores == nanmax(scores)]
#     opt_threshold = np.median(max_thresholds)
#     atts_pred = [
#         graph_utils.binarize_np_nodes(g, opt_threshold) for g in atts_pred
#     ]
#     return atts_pred


### metrics used in bagel benchmark
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import APPNP
from scipy.stats import entropy

def subgraph(model, node_idx, feature_matrix, edge_index, **kwargs):
    num_nodes, num_edges = feature_matrix.size(0), edge_index.size(1)

    flow = 'source_to_target'
    for module in model.modules():
        if isinstance(module, MessagePassing):
            flow = module.flow
            break

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if isinstance(module, APPNP):
                num_hops += module.K
            else:
                num_hops += 1

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=num_nodes, flow=flow)

    feature_matrix = feature_matrix[subset]
    for key, item in kwargs:
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]
        kwargs[key] = item

    return feature_matrix, edge_index, mapping, edge_mask, kwargs


def fidelity(model,  # is a must
             node_idx,  # is a must
             full_feature_matrix,  # must
             edge_index=None,  # the whole, so data.edge_index
             node_mask=None,  # at least one of these three: node, feature, edge
             feature_mask=None,
             edge_mask=None,
             samples=100,
             random_seed=12345,
             device="cpu"
             ):
    """
    Distortion/Fidelity (for Node Classification)
    :param model: GNN model which is explained
    :param node_idx: The node which is explained
    :param full_feature_matrix: The feature matrix from the Graph (X)
    :param edge_index: All edges
    :param node_mask: Is a (binary) tensor with 1/0 for each node in the computational graph
    => 1 means the features of this node will be fixed
    => 0 means the features of this node will be pertubed/randomized
    if not available torch.ones((1, num_computation_graph_nodes))
    :param feature_mask: Is a (binary) tensor with 1/0 for each feature
    => 1 means this features is fixed for all nodes with 1
    => 0 means this feature is randomized for all nodes
    if not available torch.ones((1, number_of_features))
    :param edge_mask:
    :param samples:
    :param random_seed:
    :param device:
    :param validity:
    :return:
    """
    if edge_mask is None and feature_mask is None and node_mask is None:
        raise ValueError("At least supply one mask")

    computation_graph_feature_matrix, computation_graph_edge_index, mapping, hard_edge_mask, kwargs = \
        subgraph(model, node_idx, full_feature_matrix, edge_index)

    # get predicted label
    log_logits = model(computation_graph_feature_matrix,
                       computation_graph_edge_index)
    predicted_labels = log_logits.argmax(dim=-1)

    predicted_label = predicted_labels[mapping]

    # fill missing masks
    if feature_mask is None:
        (num_nodes, num_features) = full_feature_matrix.size()
        feature_mask = torch.ones((1, num_features), device=device)

    num_computation_graph_nodes = computation_graph_feature_matrix.size(0)
    if node_mask is None:
        # all nodes selected
        node_mask = torch.ones((1, num_computation_graph_nodes), device=device)

    # set edge mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = edge_mask

    (num_nodes, num_features) = full_feature_matrix.size()

    num_nodes_computation_graph = computation_graph_feature_matrix.size(0)

    # retrieve complete mask as matrix
    mask = node_mask.T.matmul(feature_mask)

    correct = 0.0

    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    random_indices = torch.randint(num_nodes, (samples, num_nodes_computation_graph, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)

    for i in range(samples):
        random_features = torch.gather(full_feature_matrix,
                                       dim=0,
                                       index=random_indices[i, :, :])

        randomized_features = mask * computation_graph_feature_matrix + (1 - mask) * random_features

        log_logits = model(x=randomized_features, edge_index=computation_graph_edge_index)
        distorted_labels = log_logits.argmax(dim=-1)

        if distorted_labels[mapping] == predicted_label:
            correct += 1

    # reset mask
    if edge_mask is not None:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

    return correct / samples


#### entropy based sparsity,for further deatils read the documentation https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

def sparsity(feature_sparsity, node_sparsity, feature_mask=None, node_mask=None):
    if feature_sparsity:
        return entropy(feature_mask)
    else:
        return entropy(node_mask)


def f1(pre, rec):
    if pre == 0 or rec == 0:
        return 0
    return 2 * pre * rec / (pre + rec)


######## defining the correctness of the explanation ################

def correctness(node_mask, ground_truth, mask_type, topk):
    gt_positive = 0
    true_positive = 0
    pred_positive = 0

    if mask_type == "soft":
        values, explanation_nodes = node_mask.topk(topk)  ## make sure node_mask is a tensor, we recommend to use topk=20
    else:
        explanation_nodes = node_mask.nonzero()  ## if masks are hard then simply retrieve the indices where the mask=1

    gt_positive = gt_positive + len(ground_truth)
    pred_positive = pred_positive + len(explanation_nodes)

    for ex_node in explanation_nodes:
        if ex_node in ground_truth:
            true_positive = true_positive + 1
    recall = true_positive / gt_positive
    precision = true_positive / pred_positive
    f1_score = f1(precision, recall)

    return precision, recall, f1_score


def aopc_overall(aopc, thresholds, key):
    instance_scores = []
    for t in thresholds:
        klass = aopc[t]['classification']
        beta_0_list = aopc[t]['classification_scores']
        beta_0 = beta_0_list[0][klass]
        beta_k_list = aopc[t][key]
        beta_k = beta_k_list[0][klass]
        delta = beta_0 - beta_k
        instance_scores.append(delta.item())

    return np.mean(instance_scores)


############# the sufficiency and comprehensiveness are only for soft masks explanations
# from dataset.create_movie_reviews import to_eraser_dict
def expand_weights(weights, indices, text_len):
    expanded_weights = [0.] * text_len
    if len(weights) == len(indices):
        for idx, weight in zip(indices, weights):
            expanded_weights[idx] = weight
    else:
        assert len(indices) == text_len, f"{len(indices)} vs. {text_len}"
        for t_idx, w_idx in enumerate(indices):
            if w_idx == -1:
                continue
            expanded_weights[t_idx] = weights[w_idx]
    return expanded_weights


def _tensor_to_str_dict(t):
    t = t.view(-1)
    assert len(t) == 2, f"tensor needs to be of length 2, but has shape {t.shape} and length {len(t)}"
    return {"NEG": float(t[0]), "POS": float(t[1])}

device = 'cuda'


def to_eraser_dict(dataset, idx, weights, model=None, odd=False, device=device, k=None):
    txt, indices = dataset.get_text(idx)
    txt_len = len(txt.split(' '))
    annotation_id = dataset.movies_list[idx]["annotation_id"]
    docid = annotation_id

    if not len(weights) == txt_len:
        expanded_weights = expand_weights(weights, indices, txt_len)
    else:
        expanded_weights = weights

    rational_weights = [float(w) for w in expanded_weights]
    rationale_obj = {"annotation_id": annotation_id, "rationales": [{"docid": docid, "soft_rationale_predictions": rational_weights}]}
    aopc_dic = {}


    if model is not None:
        # do comprehensiveness & sufficiency
        data = dataset[idx].to(device)
        data.batch = torch.zeros(data.x.shape[0], device=device).long()
        assert len(weights) == data.x.shape[0]
        # get top k_d nodes
        top_k_node_mask = torch.zeros(data.x.shape[0], device=device)
        k_d = int((data.x.shape[0]*k))
        if k_d <=1:
            k_d=3
        top_k_edge_index = []
        non_top_k_edge_index = []
        for _ in range(k_d):
            max_weight = 0.
            max_idx = -1
            for i, w in enumerate(weights):
                if (max_idx == -1 or w > max_weight) and top_k_node_mask[i] == 0:
                    max_weight = w
                    max_idx = i
            top_k_node_mask[max_idx] = 1

        # construct subgraphs with only top k_d nodes and without top k_d nodes

        top_k_node_map = list(range(data.x.shape[0]))
        non_top_k_node_map = list(range(data.x.shape[0]))
        for i, b in enumerate(top_k_node_mask):
            if b == 1:
                non_top_k_node_map.remove(i)
            else:
                top_k_node_map.remove(i)
        top_k_node_map = {j: i for i, j in enumerate(top_k_node_map)}
        non_top_k_node_map = {j: i for i, j in enumerate(non_top_k_node_map)}
        for i, edge in enumerate(data.edge_index.T):
            if top_k_node_mask[edge[0].item()] == top_k_node_mask[edge[1].item()]:
                if top_k_node_mask[edge[0].item()] == 0:
                    non_top_k_edge_index.append([non_top_k_node_map[edge[0].item()], non_top_k_node_map[edge[1].item()]])
                else:
                    top_k_edge_index.append([top_k_node_map[edge[0].item()], top_k_node_map[edge[1].item()]])
        if len(top_k_edge_index) == 0:
            top_k_edge_index = [[0, 0]]
        top_k_data = Data(x=data.x[top_k_node_mask.bool()], edge_index=torch.tensor(top_k_edge_index, device=device).long().T)
        non_top_k_data = Data(x=data.x[~top_k_node_mask.bool()], edge_index=torch.tensor(non_top_k_edge_index, device=device).long().T)
        top_k_data.batch = torch.zeros(top_k_data.x.shape[0], device=device).long()
        non_top_k_data.batch = torch.zeros(non_top_k_data.x.shape[0], device=device).long()

        # get model predictions of all 3 graphs
        data.to(device)
        top_k_data.to(device)
        top_k_data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch)
            top_k_pred = model(top_k_data.x, top_k_data.edge_index, top_k_data.batch)
            non_top_k_pred = model(non_top_k_data.x, non_top_k_data.edge_index, non_top_k_data.batch)
        rationale_obj["classification"] = "NEG" if pred.argmax() == 0 else "POS"
        rationale_obj["classification_scores"] = _tensor_to_str_dict(pred)
        rationale_obj["comprehensiveness_classification_scores"] = _tensor_to_str_dict(top_k_pred)
        rationale_obj["sufficiency_classification_scores"] = _tensor_to_str_dict(non_top_k_pred)
        aopc_dic["classification"] = 0 if pred.argmax() == 0 else 1
        aopc_dic["classification_scores"] = pred
        aopc_dic["comprehensiveness_classification_scores"] = top_k_pred
        aopc_dic["sufficiency_classification_scores"] = non_top_k_pred
        aopc_dic["threshold"] = k


    return rationale_obj, aopc_dic


def suff_and_comp(idx, model, explanation, test_dataset):
    ''' We follow the definition of suff and comp from eraser benchmark https://arxiv.org/abs/1911.03429

        idx: this is the id of the graph from test dataset
        model: trained GNN model
        explanation: the soft mask explanations generated by explainers
        test_dataset: this is the test dataset for which we are computing sufficiency and comprehensiveness
    '''

    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

    aopc_dict_all = {}
    for t in thresholds:
        dict_obj, aopc_dict_ = to_eraser_dict(test_dataset, idx, explanation, model=model, k=t)
        aopc_dict_all[t] = aopc_dict_
    comp = aopc_overall(aopc_dict_all, thresholds, key='comprehensiveness_classification_scores')
    suff = aopc_overall(aopc_dict_all, thresholds, key='sufficiency_classification_scores')

    return suff, comp
