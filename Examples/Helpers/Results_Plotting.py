from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score


def roc_results(y_labels: List, y_preds: List, multiclass_num_classes: int = 3):

    if isinstance(y_preds[0], list) and isinstance(y_preds[0][0], list):

        y_preds = torch.softmax(torch.FloatTensor(y_preds), dim=1)
        y_labels = torch.nn.functional.one_hot(torch.LongTensor(y_labels), multiclass_num_classes)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(multiclass_num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_labels[:, :, i], y_preds[:, :,  i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(multiclass_num_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(multiclass_num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= multiclass_num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        fpr, tpr, thresholds = roc_curve(np.array(y_labels).ravel(), np.array(y_preds).ravel())

        roc_auc = auc(fpr, tpr)

    else:
        if isinstance(y_preds[0], list):
            y_preds_array = [sublist[0] for sublist in y_preds]
        elif isinstance(y_preds[0], torch.Tensor):
            y_preds_array = np.array([tensor.numpy() for tensor in y_preds])
        else:
            y_preds_array = y_preds
        # Step 5: Calculate the ROC curve and AUC score
        fpr, tpr, thresholds = roc_curve(y_labels, y_preds_array)
        roc_auc = roc_auc_score(y_labels, y_preds_array)




    # Step 6: Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()