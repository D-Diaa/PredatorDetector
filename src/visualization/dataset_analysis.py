import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    ConversationSequenceDataset,
    AuthorConversationSequenceDataset,
    create_dataloaders, BaseDataset
)


def temporal_correlation_analysis(
        features_array: np.ndarray,
        feature_names: List[str],
        output_dir: str,
        subset: str
) -> Dict[str, np.ndarray]:
    """
    Analyze temporal autocorrelations within features and cross-feature correlations across different time lags.

    This function computes and visualizes:
    1. Temporal autocorrelation for each feature across time lags.
    2. Cross-feature correlations at specified time lags.

    Args:
        features_array (np.ndarray): Array of shape (n_samples, sequence_length, n_features) containing feature data.
        feature_names (List[str]): List of feature names corresponding to the features in `features_array`.
        output_dir (str): Directory path where the generated plots will be saved.
        subset (str): Identifier for the subset of data being analyzed (e.g., 'all', 'positive', 'negative').

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'temporal_autocorr_{subset}': Autocorrelation matrix of shape (n_features, seq_length).
            - 'time_lagged_cross_corr_{subset}': Cross-feature correlation tensor of shape (n_features, n_features, seq_length).
    """
    n_samples, seq_length, n_features = features_array.shape

    # Precompute means and standard deviations for all features
    means = features_array.mean(axis=0)  # Shape: (seq_length, n_features)
    stds = features_array.std(axis=0)  # Shape: (seq_length, n_features)

    # Handle zero standard deviation to avoid division by zero
    stds[stds == 0] = 1e-8

    # Center the data
    centered = features_array - means  # Shape: (n_samples, seq_length, n_features)

    # 1. Compute Temporal Autocorrelation
    autocorr_matrix = np.zeros((n_features, seq_length))
    autocorr_matrix[:, 0] = 1.0  # Autocorrelation at lag 0 is 1

    for t in tqdm(range(1, seq_length), desc=f"Computing autocorrelation for {subset} subset"):
        # Shifted and non-shifted data
        shifted = centered[:, t:, :]  # Shape: (n_samples, seq_length - t, n_features)
        non_shifted = centered[:, :-t, :]  # Shape: (n_samples, seq_length - t, n_features)

        # Compute covariance
        cov = np.mean(non_shifted * shifted, axis=(0, 1))  # Shape: (n_features,)

        # Compute variance (at lag 0)
        var = np.mean(centered[:, :-t, :] * centered[:, :-t, :], axis=(0, 1))  # Shape: (n_features,)

        # Compute autocorrelation
        autocorr = cov / var
        autocorr_matrix[:, t] = autocorr

    # Visualize Temporal Autocorrelation
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        autocorr_matrix,
        xticklabels=range(seq_length),
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0
    )
    plt.title(f'Temporal Autocorrelation by Feature ({subset})')
    plt.xlabel('Time Lag')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'temporal_autocorr_{subset}.png'))
    plt.close()

    # 2. Compute Cross-Feature Correlations at Different Time Lags
    time_lagged_corr = np.zeros((n_features, n_features, seq_length))

    for t in tqdm(range(seq_length), desc=f"Computing cross-feature correlations for lagged {subset} subset"):
        if t == 0:
            # Compute correlation matrix at lag 0
            data = centered[:, :, :]  # Shape: (n_samples, seq_length, n_features)
            # Reshape to (n_samples * seq_length, n_features)
            data_reshaped = data.reshape(-1, n_features)
            corr_matrix = np.corrcoef(data_reshaped, rowvar=False)
            time_lagged_corr[:, :, t] = corr_matrix
        else:
            # Compute correlation with time lag t
            shifted = centered[:, t:, :]  # Shape: (n_samples, seq_length - t, n_features)
            non_shifted = centered[:, :-t, :]  # Shape: (n_samples, seq_length - t, n_features)

            # Reshape to (n_samples * (seq_length - t), n_features)
            shifted_reshaped = shifted.reshape(-1, n_features)
            non_shifted_reshaped = non_shifted.reshape(-1, n_features)

            # Compute correlation matrix
            corr_matrix = np.corrcoef(non_shifted_reshaped, shifted_reshaped, rowvar=False)

            # Extract the relevant cross-correlations
            # corr_matrix is (2 * n_features, 2 * n_features)
            # Top-left: non_shifted vs non_shifted
            # Top-right: non_shifted vs shifted
            # Bottom-left: shifted vs non_shifted
            # Bottom-right: shifted vs shifted
            # We need the Top-Right or Bottom-Left block
            cross_corr = corr_matrix[:n_features, n_features:]
            time_lagged_corr[:, :, t] = cross_corr

    # Visualize Cross-Feature Correlations at Key Time Lags
    lags_to_plot = [0, seq_length // 4, seq_length // 2, min(seq_length - 1, 3)]  # Adjusted to avoid negative index
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for idx, lag in enumerate(lags_to_plot):
        ax = axes[idx // 2, idx % 2]
        sns.heatmap(
            time_lagged_corr[:, :, lag],
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap='coolwarm',
            center=0,
            ax=ax
        )
        ax.set_title(f'Cross-Feature Correlation at Lag {lag}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cross_feature_temporal_corr_{subset}.png'))
    plt.close()

    return {
        f'temporal_autocorr_{subset}': autocorr_matrix,
        f'time_lagged_cross_corr_{subset}': time_lagged_corr
    }


def analyze_sequential_features(
        dataloader: DataLoader,
        feature_names: List[str],
        output_dir: str = 'sequential_analysis'
) -> Dict[str, Any]:
    """
    Perform sequence-aware dataset analysis including temporal correlation and trajectory statistics.

    This function processes sequential data to compute:
    - Temporal autocorrelations.
    - Cross-feature correlations.
    - Trajectory statistics (mean, std, trend) for each feature.
    - Trajectory divergence using KL divergence.

    Args:
        dataloader (DataLoader): DataLoader containing the sequential dataset.
        feature_names (List[str]): List of feature names corresponding to the dataset features.
        output_dir (str, optional): Directory path to save the generated visualization plots. Defaults to 'sequential_analysis'.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - Autocorrelation matrices for 'all', 'positive', and 'negative' subsets.
            - Time-lagged cross-correlation tensors for each subset.
            - Trajectory statistics including mean, std, trend, and KL divergence.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all sequential data
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Collecting sequential data"):
        features: np.ndarray = batch["sequence"].numpy()  # Shape: (batch_size, seq_length, n_features)
        labels: np.ndarray = batch["label"].numpy()

        all_features.append(features)
        all_labels.append(labels)

    # Combine all batches
    features_array: np.ndarray = np.concatenate(all_features, axis=0)
    labels_array: np.ndarray = np.concatenate(all_labels, axis=0)

    # Split by class
    pos_features: np.ndarray = features_array[labels_array == 1]
    neg_features: np.ndarray = features_array[labels_array == 0]

    # Temporal correlation analysis
    results: Dict[str, Any] = temporal_correlation_analysis(features_array, feature_names, output_dir, 'all')
    results.update(temporal_correlation_analysis(pos_features, feature_names, output_dir, 'positive'))
    results.update(temporal_correlation_analysis(neg_features, feature_names, output_dir, 'negative'))

    # Analyze sequential patterns
    seq_length: int = features_array.shape[1]

    # Compute trajectory statistics
    trajectory_stats: Dict[str, Any] = {}
    for i, feature_name in enumerate(tqdm(feature_names, desc="Analyzing feature trajectories")):
        # Analyze feature trajectories
        pos_mean: np.ndarray = np.mean(pos_features[:, :, i], axis=0)
        pos_std: np.ndarray = np.std(pos_features[:, :, i], axis=0)
        pos_trend: float = np.polyfit(range(seq_length), pos_mean, 1)[0]

        neg_mean: np.ndarray = np.mean(neg_features[:, :, i], axis=0)
        neg_std: np.ndarray = np.std(neg_features[:, :, i], axis=0)
        neg_trend: float = np.polyfit(range(seq_length), neg_mean, 1)[0]

        pos_trajectories: Dict[str, Any] = {
            'mean_trajectory': pos_mean,
            'std_trajectory': pos_std,
            'trend': pos_trend
        }

        neg_trajectories: Dict[str, Any] = {
            'mean_trajectory': neg_mean,
            'std_trajectory': neg_std,
            'trend': neg_trend
        }

        # Compute trajectory divergence using KL divergence
        kl_divs: List[float] = []
        for t in range(seq_length):
            pos_hist, _ = np.histogram(pos_features[:, t, i], bins=50, density=True)
            neg_hist, _ = np.histogram(neg_features[:, t, i], bins=50, density=True)

            # Add small epsilon to avoid division by zero
            eps: float = 1e-10
            pos_hist = pos_hist + eps
            neg_hist = neg_hist + eps

            kl_divs.append(stats.entropy(pos_hist, neg_hist))

        trajectory_stats[feature_name] = {
            'positive': pos_trajectories,
            'negative': neg_trajectories,
            'temporal_kl_divergence': np.mean(kl_divs),
            'max_kl_timestep': int(np.argmax(kl_divs))
        }

        # Visualize feature trajectories
        plt.figure(figsize=(12, 6))

        # Plot mean trajectories with confidence intervals
        time_points: range = range(seq_length)
        plt.plot(time_points, pos_trajectories['mean_trajectory'],
                 label='Positive Class', color='blue')
        plt.fill_between(time_points,
                         pos_trajectories['mean_trajectory'] - pos_trajectories['std_trajectory'],
                         pos_trajectories['mean_trajectory'] + pos_trajectories['std_trajectory'],
                         alpha=0.3, color='blue')

        plt.plot(time_points, neg_trajectories['mean_trajectory'],
                 label='Negative Class', color='red')
        plt.fill_between(time_points,
                         neg_trajectories['mean_trajectory'] - neg_trajectories['std_trajectory'],
                         neg_trajectories['mean_trajectory'] + neg_trajectories['std_trajectory'],
                         alpha=0.3, color='red')

        plt.title(f'Temporal Evolution of {feature_name}')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'trajectory_{feature_name}.png'))
        plt.close()

    results['trajectory_stats'] = trajectory_stats

    return results


def correlation_heatmap(
        features_array: np.ndarray,
        feature_names: List[str],
        output_dir: str,
        subset: str
) -> Dict[str, np.ndarray]:
    """
    Compute and visualize the correlation matrix of features.

    Args:
        features_array (np.ndarray): Array of shape (n_samples, n_features) containing feature data.
        feature_names (List[str]): List of feature names corresponding to the features in `features_array`.
        output_dir (str): Directory path where the generated heatmap will be saved.
        subset (str): Identifier for the subset of data being analyzed (e.g., 'all', 'positive', 'negative').

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'correlation_matrix_{subset}': Correlation matrix of shape (n_features, n_features).
    """
    correlation_matrix: np.ndarray = np.corrcoef(features_array.T)
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_correlation_{subset}.png'))
    plt.close()
    return {
        f'correlation_matrix_{subset}': correlation_matrix
    }


def analyze_dataset_features(
        dataloader: DataLoader,
        feature_names: List[str],
        output_dir: str = 'dataset_analysis'
) -> Dict[str, Any]:
    """
    Perform dataset-level feature analysis including correlation matrices and distribution statistics.

    This function processes the entire dataset to compute:
    - Feature correlation matrices for 'all', 'positive', and 'negative' subsets.
    - Distribution statistics (mean, std, median, skew, kurtosis) for each feature.
    - KL divergence between positive and negative class distributions.
    - Visualization of feature distributions for top features based on KL divergence.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        feature_names (List[str]): List of feature names corresponding to the dataset features.
        output_dir (str, optional): Directory path to save the generated visualization plots. Defaults to 'dataset_analysis'.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - Correlation matrices for 'all', 'positive', and 'negative' subsets.
            - Distribution statistics including mean, std, median, skew, kurtosis, and KL divergence.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all data
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Collecting dataset features"):
        features: np.ndarray = batch["sequence"].numpy()
        labels: np.ndarray = batch["label"].numpy()

        # For sequence data, take mean across sequence dimension
        if len(features.shape) > 2:
            features = features.mean(axis=1)

        all_features.append(features)
        all_labels.append(labels)

    # Combine all batches
    features_array: np.ndarray = np.concatenate(all_features, axis=0)
    labels_array: np.ndarray = np.concatenate(all_labels, axis=0)

    # Feature Distributions by Class
    pos_features: np.ndarray = features_array[labels_array == 1]
    neg_features: np.ndarray = features_array[labels_array == 0]

    # Correlation Heatmap
    results: Dict[str, Any] = correlation_heatmap(features_array, feature_names, output_dir, 'all')
    results.update(correlation_heatmap(pos_features, feature_names, output_dir, 'positive'))
    results.update(correlation_heatmap(neg_features, feature_names, output_dir, 'negative'))

    # Compute distribution statistics
    distribution_stats: Dict[str, Any] = {}
    for i, feature_name in enumerate(tqdm(feature_names, desc="Computing distribution statistics")):
        pos_stats: Dict[str, float] = {
            'mean': float(np.mean(pos_features[:, i])),
            'std': float(np.std(pos_features[:, i])),
            'median': float(np.median(pos_features[:, i])),
            'skew': float(stats.skew(pos_features[:, i])),
            'kurtosis': float(stats.kurtosis(pos_features[:, i]))
        }

        neg_stats: Dict[str, float] = {
            'mean': float(np.mean(neg_features[:, i])),
            'std': float(np.std(neg_features[:, i])),
            'median': float(np.median(neg_features[:, i])),
            'skew': float(stats.skew(neg_features[:, i])),
            'kurtosis': float(stats.kurtosis(neg_features[:, i]))
        }

        # Compute KL divergence between positive and negative distributions
        pos_hist: np.ndarray
        neg_hist: np.ndarray
        pos_hist, _ = np.histogram(pos_features[:, i], bins=50, density=True)
        neg_hist, _ = np.histogram(neg_features[:, i], bins=50, density=True)

        # Add small epsilon to avoid division by zero
        eps: float = 1e-10
        pos_hist = pos_hist + eps
        neg_hist = neg_hist + eps

        kl_div: float = float(stats.entropy(pos_hist, neg_hist))

        distribution_stats[feature_name] = {
            'positive': pos_stats,
            'negative': neg_stats,
            'kl_divergence': kl_div
        }

    results['distribution_stats'] = distribution_stats

    # Visualize distributions for top features by KL divergence
    top_features: List[Any] = sorted(
        distribution_stats.items(),
        key=lambda x: x[1]['kl_divergence'],
        reverse=True
    )[:20]

    for feature_name, _ in tqdm(top_features, desc="Visualizing top feature distributions"):
        plt.figure(figsize=(10, 6))
        try:
            idx: int = feature_names.index(feature_name)
        except ValueError:
            print(f"Feature {feature_name} not found in feature_names list.")
            continue

        sns.kdeplot(data=pos_features[:, idx], label='Positive Class', color='blue')
        sns.kdeplot(data=neg_features[:, idx], label='Negative Class', color='red')

        plt.title(f'Distribution of {feature_name} by Class')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'distribution_{feature_name}.png'))
        plt.close()

    return results


def main() -> None:
    """
    Main function to execute dataset analysis and sequential feature analysis.

    This function performs the following steps:
    1. Loads the dataset based on the specified model type.
    2. Creates data loaders for training, validation, and testing.
    3. Performs dataset-level feature analysis.
    4. Identifies top features with the most significant differences between classes.
    5. Performs sequential dataset analysis.
    6. Identifies top features with strong temporal autocorrelation and distinct temporal patterns.

    Returns:
        None
    """
    model_type: str = 'conversation'
    dataset_path: str = 'data/analyzed_conversations'

    # Load dataset
    if model_type == 'conversation':
        dataset: BaseDataset = ConversationSequenceDataset(
            dataset_path=dataset_path,
            max_seq_length=256,
            min_seq_length=8
        )
    else:
        dataset = AuthorConversationSequenceDataset(
            dataset_path,
            max_seq_length=256,
            min_seq_length=8
        )

    # Create dataloaders
    test_loader: DataLoader
    train_loader: DataLoader
    train_loader, _, test_loader = create_dataloaders(
        dataset, batch_size=64
    )
    feature_names: List[str] = dataset.feature_keys

    # Perform dataset-level analysis
    print("Performing dataset analysis...")
    dataset_results: Dict[str, Any] = analyze_dataset_features(
        train_loader,
        dataset.feature_keys,
        output_dir='dataset_analysis'
    )

    # Print some key findings from dataset analysis
    print("\nKey findings from dataset analysis:")

    # 1. Most different features between classes (by KL divergence)
    kl_divs: Dict[str, float] = {
        k: v['kl_divergence'] for k, v in dataset_results['distribution_stats'].items()
    }
    top_different: List[tuple] = sorted(kl_divs.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 features with most different distributions between classes:")
    for feature, kl_div in top_different[:5]:
        print(f"{feature}: KL divergence = {kl_div:.4f}")

    # Perform sequential analysis
    print("\nPerforming sequential dataset analysis...")
    results: Dict[str, Any] = analyze_sequential_features(
        test_loader,
        dataset.feature_keys,
        output_dir='sequential_analysis'
    )

    # Print key findings
    print("\nKey findings from sequential analysis:")

    # 1. Features with strongest temporal autocorrelation
    temporal_autocorr: np.ndarray = results['temporal_autocorr_all']
    mean_autocorr: np.ndarray = np.mean(temporal_autocorr, axis=1)
    top_temporal_indices: np.ndarray = np.argsort(mean_autocorr)[-5:]
    top_temporal: List[tuple] = [
        (feature_names[i], mean_autocorr[i]) for i in top_temporal_indices
    ]
    print("\nTop 5 features with strongest temporal autocorrelation:")
    for feature, autocorr in reversed(top_temporal):
        print(f"{feature}: mean autocorrelation = {autocorr:.4f}")

    # 2. Features with most different trajectories between classes
    trajectory_kl: Dict[str, float] = {
        k: v['temporal_kl_divergence'] for k, v in results['trajectory_stats'].items()
    }
    top_different_traj: List[tuple] = sorted(trajectory_kl.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 features with most different temporal patterns between classes:")
    for feature, kl_div in top_different_traj:
        print(f"{feature}: temporal KL divergence = {kl_div:.4f}")


if __name__ == "__main__":
    main()