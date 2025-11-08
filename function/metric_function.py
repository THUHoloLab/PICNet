import numpy as np

def rmse(prediction, target):

    return np.sqrt(np.mean((prediction - target) ** 2))


def calculate_pcc(predicted, ground_truth):
    """
    Calculate the Pearson Correlation Coefficient (PCC) between
    the predicted values and the ground truth values.

    Parameters:
    predicted (np.ndarray): Predicted values (flattened or 2D array).
    ground_truth (np.ndarray): Ground truth values (same shape as predicted).

    Returns:
    float: PCC value between the two sets of values.
    """


    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()

    mean_pred = np.mean(predicted)
    mean_gt = np.mean(ground_truth)

    covariance = np.sum((predicted - mean_pred) * (ground_truth - mean_gt))

    std_pred = np.sqrt(np.sum((predicted - mean_pred) ** 2))
    std_gt = np.sqrt(np.sum((ground_truth - mean_gt) ** 2))

    pcc_value = covariance / (std_pred * std_gt + 1e-8)

    return pcc_value
