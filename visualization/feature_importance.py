from typing import List, Callable, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from numpy import floating
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_metrics_threshold

from dataset import (
    ConversationSequenceDataset,
    create_dataloaders,
    AuthorConversationSequenceDataset
)
from models import SequenceClassifier


def compute_shap_values(
        inputs: torch.Tensor,
        model: nn.Module,
        background_samples: Optional[torch.Tensor] = None,
        num_samples: int = 100
) -> torch.Tensor:
    """
    Compute SHAP values for the given inputs using a specified model.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).
        model (nn.Module): Trained PyTorch model for which SHAP values are computed.
        background_samples (Optional[torch.Tensor], optional): Background samples for SHAP.
            Defaults to None, which uses a zero tensor as background.
        num_samples (int, optional): Number of random permutations for SHAP value estimation.
            Defaults to 100.

    Returns:
        torch.Tensor: SHAP values tensor of the same shape as `inputs`.
    """
    if background_samples is None:
        # Use zero tensor as background if not provided
        background_samples = torch.zeros_like(inputs)

    batch_size, seq_length, num_features = inputs.size()
    shap_values: torch.Tensor = torch.zeros_like(inputs)

    # Generate random permutations for feature ordering
    permutations: torch.Tensor = torch.randperm(num_features).repeat(num_samples, 1)

    for sample_idx in tqdm(range(num_samples), desc="Computing SHAP values"):
        # Get current permutation
        perm: torch.Tensor = permutations[sample_idx]

        # Initialize coalition tensors
        with_feature: torch.Tensor = background_samples.clone()
        without_feature: torch.Tensor = background_samples.clone()

        for feature_idx in perm:
            # Add current feature to coalition
            with_feature[:, :, feature_idx] = inputs[:, :, feature_idx]

            # Forward pass for both coalitions
            with torch.no_grad():
                output_with: torch.Tensor = model(with_feature)
                output_without: torch.Tensor = model(without_feature)

            # Compute marginal contribution
            marginal_contrib: torch.Tensor = output_with - output_without

            # Update SHAP values
            shap_values[:, :, feature_idx] += marginal_contrib.view(-1, 1) / num_samples

            # Update coalition without feature for next iteration
            without_feature[:, :, feature_idx] = inputs[:, :, feature_idx]

    return shap_values


def compute_permutation_importance(
        dataloader: DataLoader,
        model: nn.Module,
        device: str,
        feature_names: List[str],
        metric_fn: Callable[[List[Any], List[Any]], float],
        n_repeats: int = 5,
) -> Dict[str, floating]:
    """
    Compute permutation importance for each feature based on a specified metric.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        model (nn.Module): Trained PyTorch model for evaluation.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        feature_names (List[str]): List of feature names corresponding to the dataset.
        metric_fn (Callable[[List[Any], List[Any]], float]): Metric function to evaluate model performance.
        n_repeats (int, optional): Number of permutation repetitions per feature. Defaults to 5.

    Returns:
        Dict[str, floating]: Dictionary mapping feature names to their permutation importance scores.
    """
    # Cache all data
    all_inputs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching data for permutation importance"):
            inputs: torch.Tensor = batch["sequence"].to(device)
            labels: torch.Tensor = batch["label"]
            all_inputs.append(inputs)
            all_labels.append(labels)

    # Compute baseline metric
    baseline_outputs: List[Any] = []
    baseline_labels: List[Any] = []
    with torch.no_grad():
        for inputs, labels in zip(all_inputs, all_labels):
            outputs: torch.Tensor = model(inputs)
            baseline_outputs.extend(outputs.cpu().numpy().tolist())
            baseline_labels.extend(labels.cpu().numpy().tolist())
    baseline_metric: float = metric_fn(baseline_labels, baseline_outputs)

    # Compute importance for each feature
    importance_scores: Dict[str, float] = {}

    for feature_idx, feature_name in tqdm(enumerate(feature_names), total=len(feature_names), desc="Computing permutation importance for features"):
        feature_scores: List[float] = []

        for _ in range(n_repeats):
            permuted_outputs: List[Any] = []
            permuted_labels: List[Any] = []

            with torch.no_grad():
                for inputs, labels in zip(all_inputs, all_labels):
                    # Create copy of inputs and permute specific feature
                    permuted_inputs: torch.Tensor = inputs.clone()

                    # Permute along sequence dimension for the specific feature
                    perm_idx: torch.Tensor = torch.randperm(inputs.size(1))
                    permuted_inputs[:, :, feature_idx] = inputs[:, perm_idx, feature_idx]

                    outputs: torch.Tensor = model(permuted_inputs)
                    permuted_outputs.extend(outputs.cpu().numpy().tolist())
                    permuted_labels.extend(labels.cpu().numpy().tolist())

            # Compute metric drop
            permuted_metric: float = metric_fn(permuted_labels, permuted_outputs)
            importance: float = baseline_metric - permuted_metric
            feature_scores.append(importance)

        # Store mean importance score for the feature
        importance_scores[feature_name] = np.mean(feature_scores)

    return importance_scores


def compute_gradient_saliency(
        inputs: torch.Tensor,
        model: nn.Module,
        smooth_samples: int = 50,
        noise_scale: float = 0.1
) -> torch.Tensor:
    """
    Compute gradient saliency scores for the given inputs.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).
        model (nn.Module): Trained PyTorch model for which gradients are computed.
        smooth_samples (int, optional): Number of noisy samples for smoothing. Defaults to 50.
        noise_scale (float, optional): Scale of Gaussian noise added to inputs. Defaults to 0.1.

    Returns:
        torch.Tensor: Saliency scores tensor of the same shape as `inputs`.
    """
    saliency: torch.Tensor = torch.zeros_like(inputs)

    for _ in tqdm(range(smooth_samples), desc="Computing gradient saliency"):
        # Add random Gaussian noise
        noisy_input: torch.Tensor = inputs + torch.randn_like(inputs) * noise_scale
        noisy_input.requires_grad_(True)

        # Forward pass
        model.zero_grad()
        outputs: torch.Tensor = model(noisy_input)

        # Compute gradients
        gradients: torch.Tensor = torch.autograd.grad(outputs.sum(), noisy_input)[0]

        # Accumulate absolute gradients
        saliency += gradients.abs()

    # Average across samples
    saliency /= smooth_samples

    return saliency


def compute_attention_attribution(
        inputs: torch.Tensor,
        model: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention-based feature attributions using the model's attention weights.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).
        model (nn.Module): Trained PyTorch model containing MultiheadAttention layers.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Average attention weights tensor.
            - Feature attribution tensor computed from attention weights and inputs.
    """
    attention_weights: List[torch.Tensor] = []

    def pre_hook_fn(_module: nn.Module, args: Tuple, kwargs: Dict) -> Optional[Tuple[Tuple, Dict]]:
        # Add need_weights=True to the forward arguments
        if isinstance(_module, nn.MultiheadAttention):
            kwargs['need_weights'] = True
            return args, kwargs
        return None

    def hook_fn(_module: nn.Module, _, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        # Capture the attention weights from the output
        if isinstance(_module, nn.MultiheadAttention):
            attn_output, attn_weights = output
            attention_weights.append(attn_weights.detach())

    # Register hooks on all MultiheadAttention modules
    pre_hooks: List[torch.utils.hooks.RemovableHandle] = []
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            pre_hooks.append(module.register_forward_pre_hook(pre_hook_fn, with_kwargs=True))
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        _ = model(inputs)

    # Remove hooks
    for hook in pre_hooks:
        hook.remove()
    for hook in hooks:
        hook.remove()

    if not attention_weights:
        raise ValueError("No attention weights were captured. Check model architecture.")

    # Stack and average attention weights across layers and heads
    stacked_weights: torch.Tensor = torch.stack(attention_weights)
    avg_attention: torch.Tensor = stacked_weights.mean(dim=(0, 2))  # Average across layers and heads

    # Compute feature attribution
    feature_attribution: torch.Tensor = torch.matmul(avg_attention, inputs)

    return avg_attention, feature_attribution


def compute_integrated_gradients(
        inputs: torch.Tensor,
        model: nn.Module,
        device: str,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
) -> torch.Tensor:
    """
    Compute Integrated Gradients for the given inputs and model.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_features).
        model (nn.Module): Trained PyTorch model for which attributions are computed.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        baseline (Optional[torch.Tensor], optional): Baseline tensor for Integrated Gradients.
            Defaults to None, which uses a zero tensor.
        steps (int, optional): Number of interpolation steps between baseline and input. Defaults to 50.

    Returns:
        torch.Tensor: Integrated gradients tensor of the same shape as `inputs`.
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(device)

    # Generate interpolated points between baseline and input
    alphas: torch.Tensor = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(device)
    interpolated: torch.Tensor = baseline + alphas * (inputs - baseline)
    interpolated.requires_grad_(True)

    # collapse first 2 dimensions
    interpolated = interpolated.view(-1, interpolated.size(2), interpolated.size(3))

    # Forward pass with interpolated inputs
    model.zero_grad()
    outputs: torch.Tensor = model(interpolated)

    # Compute gradients
    gradients: torch.Tensor = torch.autograd.grad(outputs.sum(), interpolated)[0]

    # Compute integral approximation
    attributions: torch.Tensor = (inputs - baseline) * gradients.mean(dim=0)
    return attributions


def compute_feature_ablation(
        dataloader: DataLoader,
        model: nn.Module,
        device: str,
        feature_names: List[str],
        metric_fn: Callable[[List[Any], List[Any]], float]
) -> Dict[str, float]:
    """
    Compute feature importance via ablation by systematically ablating each feature and measuring performance drop.

    Args:
        dataloader (DataLoader): DataLoader containing the dataset.
        model (nn.Module): Trained PyTorch model for evaluation.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        feature_names (List[str]): List of feature names corresponding to the dataset.
        metric_fn (Callable[[List[Any], List[Any]], float]): Metric function to evaluate model performance.

    Returns:
        Dict[str, float]: Dictionary mapping feature names to their ablation importance scores.
    """
    # Cache all data
    all_inputs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching data for feature ablation"):
            inputs: torch.Tensor = batch["sequence"].to(device).clone()
            label: torch.Tensor = batch["label"]
            all_inputs.append(inputs)
            all_labels.append(label)

    # Compute baseline metric
    baseline_outputs: List[Any] = []
    baseline_labels: List[Any] = []
    with torch.no_grad():
        for inputs, label in zip(all_inputs, all_labels):
            outputs: torch.Tensor = model(inputs)
            baseline_outputs.extend(outputs.cpu().numpy().tolist())
            baseline_labels.extend(label.cpu().numpy().tolist())
    baseline_metric: float = metric_fn(baseline_labels, baseline_outputs)

    # Compute feature importance via ablation
    feature_scores: Dict[str, float] = {}
    for feature_idx, feature_name in tqdm(enumerate(feature_names), total=len(feature_names), desc="Ablating features"):
        ablated_outputs: List[Any] = []
        ablated_labels: List[Any] = []
        with torch.no_grad():
            for inputs, labels in zip(all_inputs, all_labels):
                ablated_inputs: torch.Tensor = inputs.clone()
                ablated_inputs[:, :, feature_idx] = 0  # Ablate by zeroing the feature
                outputs: torch.Tensor = model(ablated_inputs)
                ablated_outputs.extend(outputs.cpu().numpy().tolist())
                ablated_labels.extend(labels.cpu().numpy().tolist())
        ablated_metric: float = metric_fn(ablated_labels, ablated_outputs)
        feature_scores[feature_name] = baseline_metric - ablated_metric

    return feature_scores


def analyze_conv_filters(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Analyze convolutional filters in the model by computing their L2 norm.

    Args:
        model (nn.Module): Trained PyTorch model containing convolutional layers.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping convolutional layer identifiers to their filter importance scores.
    """
    filter_importance: Dict[str, np.ndarray] = {}

    for idx, conv_layer in enumerate(model.pooling_layers):
        # Get the weights of the Conv1d layer
        weights: np.ndarray = conv_layer[0].weight.detach().cpu().numpy()

        # Compute importance as L2 norm of filter weights
        filter_importance[f'kernel_size_{conv_layer[0].kernel_size[0]}'] = np.linalg.norm(weights, axis=(1, 2))

    return filter_importance

def visualize_feature_importance(
        importance_scores: Dict[str, float],
        top_k: int = 10,
        title: str = "Feature Importance Scores"
) -> None:
    """
    Visualize feature importance scores as a horizontal bar plot.

    Args:
        importance_scores (Dict[str, float]): Dictionary mapping feature names to importance scores.
        top_k (int, optional): Number of top features to display. Defaults to 10.
        title (str, optional): Title of the plot. Defaults to "Feature Importance Scores".
    """
    if not importance_scores:
        print("No importance scores to visualize.")
        return

    # Sort features by importance
    sorted_features: List[Tuple[str, float]] = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Unpack feature names and scores
    feature_names, scores = zip(*sorted_features)

    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(scores),
        y=list(feature_names),
        palette="viridis"
    )
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.tight_layout()


def analyze_feature_importance(
        model_type: str,
        model_path: str,
        dataset_path: str,
        device: str = 'cuda',
        batch_size: int = 64
) -> Dict[str, Dict[str, float]]:
    """
    Analyze feature importance using multiple methods.

    Args:
        model_type (str): Type of model ('conversation' or 'author').
        model_path (str): Path to the saved model checkpoint.
        dataset_path (str): Path to the dataset.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing results from different analysis methods.
    """
    # Load dataset
    if model_type == 'conversation':
        dataset: ConversationSequenceDataset = ConversationSequenceDataset(
            dataset_path=dataset_path,
            max_seq_length=256,
            min_seq_length=8
        )
    else:
        dataset = AuthorConversationSequenceDataset(
            dataset_path=dataset_path,
            max_seq_length=256,
            min_seq_length=8
        )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=batch_size
    )

    # Initialize model
    input_size: int = len(dataset.feature_keys)
    model: SequenceClassifier = SequenceClassifier(input_size=input_size).to(device)

    # Load model weights
    checkpoint: Dict[str, Any] = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get feature names
    feature_names: List[str] = dataset.feature_keys

    results: Dict[str, Dict[str, float]] = {}

    # 1. Feature Ablation Analysis
    print("Computing feature ablation importance...")

    def f1_score(y_true: List[Any], y_pred: List[Any]) -> float:
        return calculate_metrics_threshold(y_true, y_pred)[0]['f1']

    ablation_scores: Dict[str, float] = compute_feature_ablation(
        test_loader,
        model,
        device,
        feature_names,
        f1_score
    )
    results['ablation'] = ablation_scores

    # 2. Integrated Gradients Analysis
    print("Computing integrated gradients...")
    # Get a batch of representative samples
    sample_batch = next(iter(test_loader))
    sample_inputs: torch.Tensor = sample_batch['sequence'].to(device)

    ig_attributions: torch.Tensor = compute_integrated_gradients(
        sample_inputs,
        model,
        device,
        steps=50
    )

    # Average attributions across sequence length and batch
    avg_ig_importance: np.ndarray = ig_attributions.abs().mean(dim=(0, 1)).cpu().numpy()
    ig_scores: Dict[str, float] = dict(zip(feature_names, avg_ig_importance))
    results['integrated_gradients'] = ig_scores

    # 3. Attention Attribution Analysis
    print("Computing attention attribution...")
    attn_weights: torch.Tensor
    attn_attribution: torch.Tensor
    attn_weights, attn_attribution = compute_attention_attribution(sample_inputs, model)

    # Average attention attribution across sequence length and batch
    avg_attn_importance: np.ndarray = attn_attribution.abs().mean(dim=(0, 1)).cpu().numpy()
    attention_scores: Dict[str, float] = dict(zip(feature_names, avg_attn_importance))
    results['attention'] = attention_scores

    # 4. Convolutional Filter Analysis
    print("Analyzing convolutional filters...")
    filter_importance: Dict[str, np.ndarray] = analyze_conv_filters(model)
    results['conv_filters'] = filter_importance

    # 5. SHAP Analysis
    print("Computing SHAP values...")
    shap_attributions: torch.Tensor = compute_shap_values(
        sample_inputs,
        model,
        num_samples=100
    )
    avg_shap_importance: np.ndarray = shap_attributions.abs().mean(dim=(0, 1)).cpu().numpy()
    shap_scores: Dict[str, float] = dict(zip(feature_names, avg_shap_importance))
    results['shap'] = shap_scores

    # 6. Permutation Importance
    print("Computing permutation importance...")
    permutation_scores: Dict[str, float] = compute_permutation_importance(
        test_loader,
        model,
        device,
        feature_names,
        f1_score
    )
    results['permutation'] = permutation_scores

    # 7. Gradient Saliency
    print("Computing gradient saliency...")
    saliency_scores: torch.Tensor = compute_gradient_saliency(
        sample_inputs,
        model,
        smooth_samples=50,
        noise_scale=0.1
    )
    avg_saliency_importance: np.ndarray = saliency_scores.abs().mean(dim=(0, 1)).cpu().numpy()
    saliency_scores_dict: Dict[str, float] = dict(zip(feature_names, avg_saliency_importance))
    results['saliency'] = saliency_scores_dict

    return results


def visualize_all_results(
        results: Dict[str, Dict[str, float]],
        output_dir: str = 'feature_importance_plots'
) -> None:
    """
    Create and save visualizations for all feature importance analysis methods.

    Args:
        results (Dict[str, Dict[str, float]]): Dictionary containing results from different analysis methods.
        output_dir (str, optional): Directory path to save plots. Defaults to 'feature_importance_plots'.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Feature Importance Comparison
    methods: List[str] = ['ablation', 'integrated_gradients', 'attention', 'shap', 'saliency', 'permutation']

    for method in methods:
        if method in results:
            plt.figure(figsize=(12, 6))
            visualize_feature_importance(
                results[method],
                title=f"Feature Importance - {method.replace('_', ' ').title()}",
                top_k=10
            )
            plt.savefig(os.path.join(output_dir, f'{method}_importance.png'))
            plt.close()

    # 2. Convolutional Filter Analysis
    if 'conv_filters' in results:
        plt.figure(figsize=(10, 6))
        filter_data: Dict[str, np.ndarray] = results['conv_filters']

        # Prepare data for boxplot
        data_to_plot: List[np.ndarray] = [weights for weights in filter_data.values()]
        labels: List[str] = [f'Kernel {k}' for k in filter_data.keys()]

        sns.boxplot(data=data_to_plot, orient='h')
        plt.yticks(ticks=range(len(labels)), labels=labels)
        plt.title('Convolutional Filter Importance Distribution')
        plt.xlabel('Filter Weight Magnitude (L2 Norm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'conv_filter_importance.png'))
        plt.close()

    # 3. Feature Importance Correlation
    if all(method in results for method in methods):
        plt.figure(figsize=(8, 8))

        # Create importance matrix
        importance_matrix: np.ndarray = np.array([
            list(results[method].values()) for method in methods
        ])
        correlation: np.ndarray = np.corrcoef(importance_matrix)

        # Plot correlation heatmap
        sns.heatmap(correlation,
                    annot=True,
                    xticklabels=methods,
                    yticklabels=methods,
                    cmap='coolwarm')
        plt.title('Feature Importance Method Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_correlation.png'))
        plt.close()


def main() -> None:
    """
    Main function to execute feature importance analysis and visualization.

    This function performs the following steps:
    1. Loads the dataset based on the specified model type.
    2. Creates data loaders for training, validation, and testing.
    3. Loads the trained model.
    4. Performs feature importance analysis using multiple methods.
    5. Creates visualizations for the analysis results.
    6. Prints the top 10 most important features for each analysis method.

    Returns:
        None
    """
    # Define model paths
    model_paths: Dict[str, str] = {
        'conversation': 'models/conversation_classifier_20241129_041816.pt',
        'author': 'models/author_classifier_20241129_042403.pt'
    }
    model_type: str = 'conversation'  # Choose between 'conversation' or 'author'
    dataset_path: str = 'data/analyzed_conversations'

    print("Analyzing feature importance...")
    results: Dict[str, Dict[str, float]] = analyze_feature_importance(
        model_type=model_type,
        model_path=model_paths[model_type],
        dataset_path=dataset_path,
        device='cuda',
        batch_size=64
    )

    print("\nCreating visualizations...")
    visualize_all_results(results)

    # Print top 10 most important features for each method
    for method, scores in results.items():
        if method == 'conv_filters':
            continue  # Skip convolutional filter analysis for top features
        print(f"\nTop 10 features by {method.replace('_', ' ').title()}:")
        sorted_features: List[Tuple[str, float]] = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()