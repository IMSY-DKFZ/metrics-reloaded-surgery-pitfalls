import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
import concurrent.futures
from scipy.stats import bootstrap
from sklearn.utils import resample
from torchmetrics import AveragePrecision as AP
import seaborn as sns
import matplotlib.pyplot as plt
from torcheval.metrics import TopKMultilabelAccuracy

# TODO add paths
path = "" # working directory
data_path = ""
df = pd.read_csv(data_path + 'cholectriplet_official_CV_split.csv')

class CFG:
    device =  "cpu"
    metric_tsize = 100  # Number of predictions/triplets
    col0 = "tri0"  # First triplet column name

########################################################
####### Compute boostrap CIs for micro mAP
########################################################

def compute_mAP_naive(data, CFG=CFG):
    tri0_idx = int(data.columns.get_loc(CFG.col0))
    pred0_idx = int(data.columns.get_loc("0"))

    # AP is computed as mutlilabel problem with micro averaging method
    torch_ap = AP(
        task="multilabel",
        num_labels=CFG.metric_tsize,
        average="micro"
    ).to(CFG.device)

    predictions = torch.tensor(data.iloc[:, pred0_idx: pred0_idx + CFG.metric_tsize].values,
                               dtype=torch.float32, device=CFG.device)
    sigmoid_preds = predictions.sigmoid()
    ground_truth = torch.tensor(
        data.iloc[:, tri0_idx: tri0_idx + CFG.metric_tsize].values,
        dtype=torch.long,
        device=CFG.device
    )

    classwise = torch_ap(sigmoid_preds, ground_truth)
    return classwise


def bootstrap_ap(data, n_iterations=1000, random_state=None):
    """
    Compute the Confidence Interval for AP using bootstrap sampling with scipy for percentiles.
    """
    np.random.seed(random_state)
    ap_scores = []

    # Perform bootstrap sampling
    for i in range(n_iterations):
        # Resample with replacement
        resample_df = resample(data)

        # Compute AP for the bootstrapped sample
        ap = compute_mAP_naive(resample_df)
        ap_scores.append(ap)

    ci_lower = np.percentile(ap_scores, 2.5)  # 2.5th percentile
    ci_upper = np.percentile(ap_scores, 97.5)  # 97.5th percentile

    return ci_lower, ci_upper


def bootstrap_hierarchique_one_level(data, n_iterations):
    bootstrap_samples = []

    videos = data['video'].unique()

    for i in range(n_iterations):
        bootstrap_data = []
        # First resample over the videos
        sampled_video = np.random.choice(videos, size=len(videos), replace=True)

        for vid in sampled_video:
            vid_data = data[data['video'] == vid]
            # Second resample over the images on the sampled videos
            sampled_ap = resample(vid_data)

            bootstrap_data.append(compute_mAP_naive(sampled_ap))

        # average mAP on the images for each video
        bootstrap_samples.append(np.mean(bootstrap_data))

    return bootstrap_samples


# Compute hierarchical mean mAP (average mAP across videos)
def hier_mean(data):
    videos = data['video'].unique()
    mean_array = []
    for vid in videos:
        vid_data = data[data['video'] == vid]
        mean = compute_mAP_naive(vid_data)
        mean_array.append(mean)
    return np.mean(mean_array)


def process_fold(fold):
    """Function to process one fold in parallel."""
    fold_df = df[df.fold == fold].copy().reset_index()
    t = bootstrap_hierarchique_one_level(fold_df, 1000)
    hierarchical_lower, hierarchical_upper = np.percentile(t, [2.5, 97.5])
    naive_lower, naive_upper = bootstrap_ap(fold_df)
    naive_mean = compute_mAP_naive(fold_df)
    hierar_mean = hier_mean(fold_df)

    return [
        {'fold': fold, 'type': 'bootstrap_hierarchical', 'CI_lower': hierarchical_lower, "CI_upper": hierarchical_upper,
         "CI_width": hierarchical_upper - hierarchical_lower, 'mean': hierar_mean},
        {'fold': fold, 'type': 'bootstrap_naive', 'CI_lower': naive_lower, "CI_upper": naive_upper,
         "CI_width": naive_upper - naive_lower, 'mean': naive_mean}
    ]


by_fold_micro = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_fold, range(5)))

by_fold_micro = [entry for result in results for entry in result]
print(by_fold_micro)

df_results_micro = pd.DataFrame(by_fold_micro)
df_results_micro["mean"] = df_results_micro["mean"].apply(float)

print(df_results_micro)
df_results_micro.to_csv(path + "results_CI_comparison_mAP.csv", index=False)


### Plot CI widths across folds
plt.figure(figsize=(8, 5))

sns.barplot(data=df_results_micro, x="fold", y="CI_width", hue="type", palette={"bootstrap_hierarchical": "red", "bootstrap_naive": "blue"})

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("Confidence Interval Width")
plt.ylim([0,0.3])
plt.title("Comparison of CI Width for micro mAP")
plt.legend(title="Bootstrap Method")

# Show the plot
plt.show()

# Compute ratio (hierarchical / naive) per fold
ratio_df_mAP = (
    df_results_micro.pivot_table(index="fold", columns="type", values="CI_width")
    .reset_index()
)
ratio_df_mAP["ratio"] = ratio_df_mAP["bootstrap_hierarchical"] / ratio_df_mAP["bootstrap_naive"]

df_results_micro = pd.merge(df_results_micro, ratio_df_mAP[["fold", "ratio"]], on="fold", how="left")

# Compute median summary over folds
df_results_medianfolds_mAP = (
    df_results_micro.groupby("type")
    .agg(
        CI_lower_median=("CI_lower", "median"),
        CI_upper_median=("CI_upper", "median"),
        CI_width_median=("CI_width", "median"),
        mean_median=("mean", "median"),
        ratio=("ratio", "median")
    )
    .reset_index()
)

print(df_results_medianfolds_mAP)

### Plot median CIs
df_results_medianfolds_mAP = df_results_medianfolds_mAP.sort_values("type", ascending=False).reset_index(drop=True)
plt.figure(figsize=(4, 5))

x = [0, 1]  # x positions for naive and hierarchical
plt.vlines(x, df_results_medianfolds_mAP["CI_lower_median"], df_results_medianfolds_mAP["CI_upper_median"], color="black", lw=2)
plt.scatter(x, df_results_medianfolds_mAP["mean_median"], color="black", s=50)

plt.xticks(x, df_results_medianfolds_mAP["type"], rotation=15)
plt.ylabel("mAP")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

########################################################
####### Compute boostrap CIs for weighted mAP
########################################################
def compute_mAP_naive(data, CFG=CFG):
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(data.columns.get_loc(CFG.col0))
    pred0_idx = int(data.columns.get_loc("0"))

    # Empty list to stack the [100 predictions] of each video

    # Get the fold's corresponding videos

    results = []
    torch_ap = AP(
        task="multilabel",
        num_labels=CFG.metric_tsize,
        average="weighted"
    ).to(CFG.device)
    # Do this once before the loop

    predictions = torch.tensor(data.iloc[:, pred0_idx: pred0_idx + CFG.metric_tsize].values,
                               dtype=torch.float32, device=CFG.device)
    sigmoid_preds = predictions.sigmoid()
    # Precompute sigmoid
    ground_truth = torch.tensor(
        data.iloc[:, tri0_idx: tri0_idx + CFG.metric_tsize].values,
        dtype=torch.long,
        device=CFG.device
    )

    classwise = torch_ap(sigmoid_preds, ground_truth)
    return classwise


def compute_mAP_naive_multiclass(data, CFG=CFG):
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(data.columns.get_loc(CFG.col0))
    pred0_idx = int(data.columns.get_loc("0"))

    # Empty list to stack the [100 predictions] of each video

    # Get the fold's corresponding videos

    results = []
    torch_ap = AP(
        task="multiclass",
        num_classes=CFG.metric_tsize,
        average="weighted"
    ).to(CFG.device)
    # Do this once before the loop

    predictions = torch.tensor(data.iloc[:, pred0_idx: pred0_idx + CFG.metric_tsize].values,
                               dtype=torch.float32, device=CFG.device)
    sigmoid_preds = predictions.sigmoid()
    # Precompute sigmoid
    ground_truth = torch.tensor(
        data.iloc[:, tri0_idx: tri0_idx + CFG.metric_tsize].values,
        dtype=torch.long,
        device=CFG.device
    )
    targets = torch.argmax(ground_truth, dim=1)
    classwise = torch_ap(sigmoid_preds, targets)
    return classwise


def bootstrap_ap(data, n_iterations=1000, random_state=None):
    """
    Compute the Confidence Interval for AP using bootstrap sampling with scipy for percentiles.

    Parameters:
    - df: DataFrame containing the true labels and predicted scores.
    - tri0_idx: The index of the first column containing true labels.
    - pred0_idx: The index of the first column containing predicted probabilities.
    - n_iterations: The number of bootstrap samples to generate.
    - random_state: Random state for reproducibility.

    Returns:
    - ci_lower: Lower bound of the CI.
    - ci_upper: Upper bound of the CI.
    """
    np.random.seed(random_state)
    ap_scores = []

    # Get the true labels and predicted probabilities

    # Perform bootstrap sampling
    for i in range(n_iterations):
        # Resample with replacement
        resample_df = resample(data)

        # Compute AP for the bootstrapped sample
        ap = compute_mAP_naive_multiclass(resample_df)
        ap_scores.append(ap)

    # Compute the Confidence Interval (CI) for AP using scipy
    ci_lower = np.percentile(ap_scores, 2.5)  # 2.5th percentile
    ci_upper = np.percentile(ap_scores, 97.5)  # 97.5th percentile

    return ci_lower, ci_upper


def bootstrap_hierarchique_one_level(data, n_iterations):
    # Initialisation de la liste pour stocker les échantillons bootstrap
    bootstrap_samples = []

    videos = data['video'].unique()

    # Effectuer le bootstrap sur n_iterations
    for i in range(n_iterations):
        bootstrap_data = []
        print(i)
        sampled_video = np.random.choice(videos, size=len(videos), replace=True)

        for vid in sampled_video:
            vid_data = data[data['video'] == vid]

            sampled_ap = resample(vid_data)

            bootstrap_data.append(compute_mAP_naive_multiclass(sampled_ap))

        bootstrap_samples.append(np.mean(bootstrap_data))

    return bootstrap_samples


def hier_mean(data):
    videos = data['video'].unique()
    mean_array = []
    for vid in videos:
        vid_data = data[data['video'] == vid]
        mean = compute_mAP_naive_multiclass(vid_data)
        mean_array.append(mean)
    return np.mean(mean_array)


def process_fold(fold):
    print(f"processing {fold}")
    """Function to process one fold in parallel."""
    fold_df = df[df.fold == fold].copy().reset_index()
    t = bootstrap_hierarchique_one_level(fold_df, 1000)
    hierarchical_lower, hierarchical_upper = np.percentile(t, [2.5, 97.5])
    naive_lower, naive_upper = bootstrap_ap(fold_df)
    naive_mean = compute_mAP_naive_multiclass(fold_df)
    hierar_mean = hier_mean(fold_df)

    return [
        {'fold': fold, 'type': 'bootstrap_hierarchical', 'CI_lower': hierarchical_lower, "CI_upper": hierarchical_upper,
         "CI_width": hierarchical_upper - hierarchical_lower, 'mean': hierar_mean},
        {'fold': fold, 'type': 'bootstrap_naive', 'CI_lower': naive_lower, "CI_upper": naive_upper,
         "CI_width": naive_upper - naive_lower, 'mean': naive_mean}
    ]


by_fold_weigthed_mAP = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_fold, range(5)))
# Flatten results
by_fold_weigthed_mAP = [entry for result in results for entry in result]
print(by_fold_weigthed_mAP)

# Parallel execution
df_results_mAP_weighted = pd.DataFrame(by_fold_weigthed_mAP)
df_results_mAP_weighted["mean"] = df_results_mAP_weighted["mean"].apply(float)

print(df_results_mAP_weighted)
df_results_mAP_weighted.to_csv(path + "results_CI_comparison_mAP_weighted.csv", index=False)

### Plot CI widths across folds
plt.figure(figsize=(8, 5))

sns.barplot(data=df_results_mAP_weighted, x="fold", y="CI_width", hue="type", palette={"bootstrap_hierarchical": "red", "bootstrap_naive": "blue"})

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("Confidence Interval Width")
plt.ylim([0,0.3])
plt.title("Comparison of CI Width for weighted mAP")
plt.legend(title="Bootstrap Method")

# Show the plot
plt.show()

# Compute ratio (hierarchical / naive) per fold
ratio_df_mAP_weighted = (
    df_results_mAP_weighted.pivot_table(index="fold", columns="type", values="CI_width")
    .reset_index()
)
ratio_df_mAP_weighted["ratio"] = ratio_df_mAP_weighted["bootstrap_hierarchical"] / ratio_df_mAP_weighted["bootstrap_naive"]

df_results_mAP_weighted = pd.merge(df_results_mAP_weighted, ratio_df_mAP_weighted[["fold", "ratio"]], on="fold", how="left")

# Compute median summary over folds
df_results_medianfolds_mAP_weighted = (
    df_results_mAP_weighted.groupby("type")
    .agg(
        CI_lower_median=("CI_lower", "median"),
        CI_upper_median=("CI_upper", "median"),
        CI_width_median=("CI_width", "median"),
        mean_median=("mean", "median"),
        ratio=("ratio", "median")
    )
    .reset_index()
)

print(df_results_medianfolds_mAP_weighted)

### Plot median CIs
df_results_medianfolds_mAP_weighted = df_results_medianfolds_mAP_weighted.sort_values("type", ascending=False).reset_index(drop=True)
plt.figure(figsize=(4, 5))

x = [0, 1]  # x positions for naive and hierarchical
plt.vlines(x, df_results_medianfolds_mAP_weighted["CI_lower_median"], df_results_medianfolds_mAP_weighted["CI_upper_median"], color="black", lw=2)
plt.scatter(x, df_results_medianfolds_mAP_weighted["mean_median"], color="black", s=50)

plt.xticks(x, df_results_medianfolds_mAP_weighted["type"], rotation=15)
plt.ylabel("Weighted mAP")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

########################################################
####### Compute boostrap CIs for top k accuracy
########################################################
def compute_top_k_accuracy(data, k, CFG=CFG):
    """
    Compute the Top-k accuracy for a dataset.

    Parameters:
    - data: The input DataFrame with predictions and true labels.
    - k: The value of k for Top-k accuracy.
    - CFG: Configuration object containing necessary information.

    Returns:
    - top_k_accuracy: The Top-k accuracy.
    """
    metric = TopKMultilabelAccuracy(criteria="overlap", k=k)
    tri0_idx = int(data.columns.get_loc(CFG.col0))  # Get true label index
    pred0_idx = int(data.columns.get_loc("0"))  # Get prediction column index

    # Get predictions and ground truth
    predictions = torch.tensor(data.iloc[:, pred0_idx: pred0_idx + CFG.metric_tsize].values,
                               dtype=torch.float32, device=CFG.device)
    ground_truth = torch.tensor(data.iloc[:, tri0_idx: tri0_idx + CFG.metric_tsize].values,
                                dtype=torch.float32, device=CFG.device)  # Use float32 for multi-label

    # Get the top-k predictions

    metric.update(predictions, ground_truth)
    metric.compute()

    # Check if the ground truth is present in the top-k predictions
    # Top-k predictions are the indices of the top-k predicted labels, so we check if
    # any of the top-k predictions match the ground truth class labels.

    return metric.compute()


compute_top_k_accuracy(df[df.fold == 1], 5)

t = np.array([[1, 2, 3, 4], [2, 1, 4, 3], [4, 3, 2, 1]])  # Multiple rows for t
truth = np.array([[1, 0, 0, 0, 0, 0, 1],  # Corresponding truth rows
                  [0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0, 0]])

# Step 1: Find indices where truth is 1
for row_t, row_truth in zip(t, truth):
    # Step 1: Find indices where truth is 1 for the current row
    truth_indices = np.where(row_truth == 1)[0]

    # Step 2: Check if any element in row_t corresponds to an index where truth is 1
    matching_indices = [x for x in row_t if x in truth_indices]

    # Result: matching indices where truth is 1 for this row
    print(f"Matching indices for row {row_t}: {matching_indices}")


def bootstrap_top_k_accuracy(data, k, n_iterations=1000, random_state=None):
    """
    Compute the Confidence Interval for Top-k accuracy using bootstrap sampling.

    Parameters:
    - data: DataFrame containing the true labels and predicted scores.
    - k: The value of k for Top-k accuracy.
    - n_iterations: The number of bootstrap samples to generate.
    - random_state: Random state for reproducibility.

    Returns:
    - ci_lower: Lower bound of the CI.
    - ci_upper: Upper bound of the CI.
    """
    np.random.seed(random_state)
    top_k_accuracies = []

    # Perform bootstrap sampling
    for i in range(n_iterations):
        resample_df = resample(data)
        top_k_accuracy = compute_top_k_accuracy(resample_df, k)
        top_k_accuracies.append(top_k_accuracy)

    # Compute Confidence Interval (CI)
    ci_lower = np.percentile(top_k_accuracies, 2.5)
    ci_upper = np.percentile(top_k_accuracies, 97.5)

    return ci_lower, ci_upper


def bootstrap_hierarchique_top_k(data, k, n_iterations):
    """
    Perform hierarchical bootstrap sampling to compute Top-k accuracy.

    Parameters:
    - data: DataFrame with the dataset.
    - k: The value of k for Top-k accuracy.
    - n_iterations: The number of bootstrap samples.

    Returns:
    - bootstrap_samples: List of Top-k accuracies for each bootstrap iteration.
    """
    bootstrap_samples = []
    videos = data['video'].unique()

    for i in range(n_iterations):
        bootstrap_data = []
        print(i)
        sampled_video = np.random.choice(videos, size=len(videos), replace=True)

        for vid in sampled_video:
            vid_data = data[data['video'] == vid]
            sampled_accuracy = resample(vid_data)
            bootstrap_data.append(compute_top_k_accuracy(sampled_accuracy, k))

        bootstrap_samples.append(np.mean(bootstrap_data))

    return bootstrap_samples


def hier_mean_top_k(data, k):
    """
    Compute hierarchical mean Top-k accuracy.

    Parameters:
    - data: DataFrame containing the dataset.
    - k: The value of k for Top-k accuracy.

    Returns:
    - The mean Top-k accuracy across videos.
    """
    videos = data['video'].unique()
    mean_array = []

    for vid in videos:
        vid_data = data[data['video'] == vid]
        mean = compute_top_k_accuracy(vid_data, k)
        mean_array.append(mean)

    return np.mean(mean_array)


def process_fold(fold, k):
    """
    Process one fold in parallel to compute Top-k accuracy and confidence intervals.

    Parameters:
    - fold: The fold number.
    - k: The value of k for Top-k accuracy.

    Returns:
    - Results with confidence intervals and means for Top-k accuracy.
    """
    print(f"Processing fold {fold}")
    fold_df = df[df.fold == fold].copy().reset_index()

    # Hierarchical bootstrap
    t = bootstrap_hierarchique_top_k(fold_df, k, 1000)
    hierarchical_lower, hierarchical_upper = np.percentile(t, [2.5, 97.5])

    # Naive bootstrap
    naive_lower, naive_upper = bootstrap_top_k_accuracy(fold_df, k)

    # Mean Top-k accuracy
    naive_mean = compute_top_k_accuracy(fold_df, k)
    hierar_mean = hier_mean_top_k(fold_df, k)

    return [
        {'fold': fold, 'type': 'bootstrap_hierarchical', 'CI_lower': hierarchical_lower, 'CI_upper': hierarchical_upper,
         'CI_width': hierarchical_upper - hierarchical_lower, 'mean': hierar_mean},
        {'fold': fold, 'type': 'bootstrap_naive', 'CI_lower': naive_lower, 'CI_upper': naive_upper,
         'CI_width': naive_upper - naive_lower, 'mean': naive_mean}
    ]


# Parallel execution for Top-k accuracy
by_fold_overlap_Acc = []
k = 5  # Change k to the desired value for top-k accuracy (e.g., 1 for Top-1, 5 for Top-5)
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda fold: process_fold(fold, k), range(5)))

# Flatten results
by_fold_overlap_Acc = [entry for result in results for entry in result]
print(by_fold_overlap_Acc)

df_results_Acc = pd.DataFrame(by_fold_overlap_Acc)
df_results_Acc["mean"] = df_results_Acc["mean"].apply(float)
print(df_results_Acc)
df_results_Acc.to_csv(path + "results_CI_comparison_Accuracy.csv", index=False)



# Set up the figure
plt.figure(figsize=(8, 5))

# Bar plot comparing CI widths across folds
sns.barplot(data=df_results_Acc, x="fold", y="CI_width", hue="type", palette={"bootstrap_hierarchical": "red", "bootstrap_naive": "blue"})

# Add labels and title
plt.xlabel("Fold")
plt.ylabel("Confidence Interval Width")
plt.ylim([0,0.3])
plt.title("Comparison of CI Width for top-k Accuracy")
plt.legend(title="Bootstrap Method")

# Show the plot
plt.show()

# Compute ratio (hierarchical / naive) per fold
ratio_df_Acc = (
    df_results_Acc.pivot_table(index="fold", columns="type", values="CI_width")
    .reset_index()
)
ratio_df_Acc["ratio"] = ratio_df_Acc["bootstrap_hierarchical"] / ratio_df_Acc["bootstrap_naive"]

df_results_Acc = pd.merge(df_results_Acc, ratio_df_Acc[["fold", "ratio"]], on="fold", how="left")

# Compute median summary over folds
df_results_medianfolds_Acc = (
    df_results_Acc.groupby("type")
    .agg(
        CI_lower_median=("CI_lower", "median"),
        CI_upper_median=("CI_upper", "median"),
        CI_width_median=("CI_width", "median"),
        mean_median=("mean", "median"),
        ratio=("ratio", "median")
    )
    .reset_index()
)

print(df_results_medianfolds_Acc)

### Plot median CIs
df_results_medianfolds_Acc = df_results_medianfolds_Acc.sort_values("type", ascending=False).reset_index(drop=True)
plt.figure(figsize=(4, 5))

x = [0, 1]  # x positions for naive and hierarchical
plt.vlines(x, df_results_medianfolds_Acc["CI_lower_median"], df_results_medianfolds_Acc["CI_upper_median"], color="black", lw=2)
plt.scatter(x, df_results_medianfolds_Acc["mean_median"], color="black", s=50)

plt.xticks(x, df_results_medianfolds_Acc["type"], rotation=15)
plt.ylabel("top-5 Accuracy")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

