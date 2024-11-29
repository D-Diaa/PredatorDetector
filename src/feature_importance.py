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

from dataset import ConversationSequenceDataset, create_dataloaders, AuthorConversationSequenceDataset
from models import SequenceClassifier


def compute_shap_values(
        inputs: torch.Tensor,
        model: nn.Module,
        background_samples: Optional[torch.Tensor] = None,
        num_samples: int = 100
) -> torch.Tensor:
    if background_samples is None:
        # Use zero tensor as background if not provided
        background_samples = torch.zeros_like(inputs)

    batch_size, seq_length, num_features = inputs.size()
    shap_values = torch.zeros_like(inputs)

    # Generate random permutations for feature ordering
    permutations = torch.randperm(num_features).repeat(num_samples, 1)

    for sample_idx in range(num_samples):
        # Get current permutation
        perm = permutations[sample_idx]

        # Initialize coalition tensors
        with_feature = background_samples.clone()
        without_feature = background_samples.clone()

        for feature_idx in perm:
            # Add current feature to coalition
            with_feature[:, :, feature_idx] = inputs[:, :, feature_idx]

            # Forward pass for both coalitions
            with torch.no_grad():
                output_with = model(with_feature)
                output_without = model(without_feature)

            # Compute marginal contribution
            marginal_contrib = output_with - output_without

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
) -> dict[str, floating[Any]]:
    # Cache all data
    all_inputs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["sequence"].to(device)
            labels = batch["label"]
            all_inputs.append(inputs)
            all_labels.append(labels)

    # Compute baseline metric
    baseline_outputs = []
    baseline_labels = []
    with torch.no_grad():
        for inputs, labels in zip(all_inputs, all_labels):
            outputs = model(inputs)
            baseline_outputs.extend(outputs.cpu().numpy().tolist())
            baseline_labels.extend(labels.cpu().numpy().tolist())
    baseline_metric = metric_fn(baseline_labels, baseline_outputs)

    # Compute importance for each feature
    importance_scores = {}

    for feature_idx, feature_name in tqdm(enumerate(feature_names), total=len(feature_names)):
        feature_scores = []

        for _ in range(n_repeats):
            permuted_outputs = []
            permuted_labels = []

            with torch.no_grad():
                for inputs, labels in zip(all_inputs, all_labels):
                    # Create copy of inputs and permute specific feature
                    permuted_inputs = inputs.clone()

                    # Permute along sequence dimension for the specific feature
                    perm_idx = torch.randperm(inputs.size(1))
                    permuted_inputs[:, :, feature_idx] = inputs[:, perm_idx, feature_idx]

                    outputs = model(permuted_inputs)
                    permuted_outputs.extend(outputs.cpu().numpy().tolist())
                    permuted_labels.extend(labels.cpu().numpy().tolist())

            # Compute metric drop
            permuted_metric = metric_fn(permuted_labels, permuted_outputs)
            importance = baseline_metric - permuted_metric
            feature_scores.append(importance)

        # Store mean and std of importance scores
        importance_scores[feature_name] = np.mean(feature_scores)

    return importance_scores

def compute_gradient_saliency(
        inputs: torch.Tensor,
        model: nn.Module,
        smooth_samples: int = 50,
        noise_scale: float = 0.1
) -> torch.Tensor:
    saliency = torch.zeros_like(inputs)

    for _ in range(smooth_samples):
        # Add random Gaussian noise
        noisy_input = inputs + torch.randn_like(inputs) * noise_scale
        noisy_input.requires_grad_(True)

        # Forward pass
        model.zero_grad()
        outputs = model(noisy_input)

        # Compute gradients
        gradients = torch.autograd.grad(outputs.sum(), noisy_input)[0]

        # Accumulate absolute gradients
        saliency += gradients.abs()

    # Average across samples
    saliency /= smooth_samples

    return saliency


def compute_attention_attribution(
        inputs: torch.Tensor,
        model: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    attention_weights = []

    def pre_hook_fn(_module, args, kwargs):
        # Add need_weights=True to the forward arguments
        if isinstance(_module, nn.MultiheadAttention):
            kwargs['need_weights'] = True
            return args, kwargs
        return None

    def hook_fn(_module, _, output):
        # Capture the attention weights from the output
        if isinstance(_module, nn.MultiheadAttention):
            attn_output, attn_weights = output
            attention_weights.append(attn_weights.detach())

    # Register hooks on all MultiheadAttention modules
    pre_hooks = []
    hooks = []
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
    stacked_weights = torch.stack(attention_weights)
    avg_attention = stacked_weights.mean(dim=(0, 2))  # Average across layers and heads

    # Compute feature attribution
    feature_attribution = torch.matmul(avg_attention, inputs)

    return avg_attention, feature_attribution


def compute_integrated_gradients(
        inputs: torch.Tensor,
        model: nn.Module,
        device: str,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
) -> torch.Tensor:
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    # Generate interpolated points between baseline and input
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1).to(device)
    interpolated = baseline + alphas * (inputs - baseline)
    interpolated.requires_grad_(True)

    # collapse first 2 dimensions
    interpolated = interpolated.view(-1, interpolated.size(2), interpolated.size(3))

    # Forward pass with interpolated inputs
    model.zero_grad()
    outputs = model(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(outputs.sum(), interpolated)[0]

    # Compute integral approximation
    attributions = (inputs - baseline) * gradients.mean(dim=0)
    return attributions


def compute_feature_ablation(
        dataloader: DataLoader,
        model: nn.Module,
        device: str,
        feature_names: List[str],
        metric_fn: Callable[[List[Any], List[Any]], float]
) -> Dict[str, float]:

    # Cache all data
    all_inputs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["sequence"].to(device).clone()
            label = batch["label"]
            all_inputs.append(inputs)
            all_labels.append(label)

    # Compute baseline metric
    baseline_outputs = []
    baseline_labels = []
    with torch.no_grad():
        for inputs, label in zip(all_inputs, all_labels):
            outputs = model(inputs)
            baseline_outputs.extend(outputs.cpu().numpy().tolist())
            baseline_labels.extend(label.cpu().numpy().tolist())
    baseline_metric = metric_fn(baseline_labels, baseline_outputs)

    # Compute feature importance via ablation
    feature_scores = {}
    for feature_idx, feature_name in tqdm(enumerate(feature_names), total=len(feature_names)):
        ablated_outputs = []
        ablated_labels = []
        with torch.no_grad():
            for inputs, labels in zip(all_inputs, all_labels):
                ablated_inputs = inputs.clone()
                ablated_inputs[:, :, feature_idx] = 0  # Or use a different ablation strategy
                outputs = model(ablated_inputs)
                ablated_outputs.extend(outputs.cpu().numpy().tolist())
                ablated_labels.extend(labels.cpu().numpy().tolist())
        ablated_metric = metric_fn(ablated_labels, ablated_outputs)
        feature_scores[feature_name] = baseline_metric - ablated_metric

    return feature_scores


def analyze_conv_filters(model: nn.Module) -> Dict[str, np.ndarray]:
    filter_importance = {}

    for idx, conv_layer in enumerate(model.pooling_layers):
        # Get the weights of the Conv1d layer
        weights = conv_layer[0].weight.detach().cpu().numpy()

        # Compute importance as L2 norm of filter weights
        filter_importance[f'kernel_size_{conv_layer[0].kernel_size[0]}'] = np.linalg.norm(weights, axis=(1, 2))

    return filter_importance


class FeatureImportanceAnalyzer:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode


def visualize_feature_importance(
        importance_scores: Dict[str, float],
        top_k: int = 10,
        title: str = "Feature Importance Scores"
) -> None:

    # Sort features by importance
    sorted_features = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=[x[1] for x in sorted_features],
        y=[x[0] for x in sorted_features],
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
        model_type: Type of model ('conversation' or 'author')
        model_path: Path to saved model checkpoint
        dataset_path: Path to dataset
        device: Device to run analysis on
        batch_size: Batch size for DataLoader

    Returns:
        Dictionary containing results from different analysis methods
    """
    # Load dataset
    if model_type == 'conversation':
        dataset = ConversationSequenceDataset(
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
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=batch_size
    )

    # Initialize model
    input_size = len(dataset.feature_keys)
    model = SequenceClassifier(input_size=input_size).to(device)

    # Load model weights
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get feature names
    feature_names = dataset.feature_keys

    results = {}

    # 1. Feature Ablation Analysis
    print("Computing feature ablation importance...")

    def f1_score(y_true, y_pred):
        return calculate_metrics_threshold(y_true, y_pred)[0]['f1']

    ablation_scores = compute_feature_ablation(
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
    sample_inputs = sample_batch['sequence'].to(device)

    ig_attributions = compute_integrated_gradients(
        sample_inputs,
        model,
        device,
        steps=50
    )

    # Average attributions across sequence length and batch
    avg_ig_importance = ig_attributions.abs().mean(dim=(0, 1)).cpu().numpy()
    ig_scores = dict(zip(feature_names, avg_ig_importance))
    results['integrated_gradients'] = ig_scores

    # 3. Attention Attribution Analysis
    print("Computing attention attribution...")
    attn_weights, attn_attribution = compute_attention_attribution(sample_inputs, model)

    # Average attention attribution across sequence length and batch
    avg_attn_importance = attn_attribution.abs().mean(dim=(0, 1)).cpu().numpy()
    attention_scores = dict(zip(feature_names, avg_attn_importance))
    results['attention'] = attention_scores

    # 4. Convolutional Filter Analysis
    print("Analyzing convolutional filters...")
    filter_importance = analyze_conv_filters(model)
    results['conv_filters'] = filter_importance

    # 5. SHAP Analysis
    print("Computing SHAP values...")
    shap_attributions = compute_shap_values(
        sample_inputs,
        model,
        num_samples=100
    )
    avg_shap_importance = shap_attributions.abs().mean(dim=(0, 1)).cpu().numpy()
    shap_scores = dict(zip(feature_names, avg_shap_importance))
    results['shap'] = shap_scores

    # 6. Permutation Importance
    print("Computing permutation importance...")
    permutation_scores = compute_permutation_importance(
        test_loader,
        model,
        device,
        feature_names,
        f1_score
    )
    results['permutation'] = permutation_scores

    # 7. Gradient Saliency
    print("Computing gradient saliency...")
    saliency_scores = compute_gradient_saliency(
        sample_inputs,
        model,
        smooth_samples=50,
        noise_scale=0.1
    )
    avg_saliency_importance = saliency_scores.abs().mean(dim=(0, 1)).cpu().numpy()
    saliency_scores_dict = dict(zip(feature_names, avg_saliency_importance))
    results['saliency'] = saliency_scores_dict

    return results


def visualize_all_results(
        results: Dict[str, Dict[str, float]],
        output_dir: str = 'feature_importance_plots'
):
    """
    Create visualizations for all analysis methods.

    Args:
        results: Dictionary containing results from different analysis methods
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Feature Importance Comparison
    methods = ['ablation', 'integrated_gradients', 'attention', 'shap', 'saliency', 'permutation']

    for method in methods:
        if method in results:
            plt.figure(figsize=(12, 6))
            visualize_feature_importance(
                results[method],
                title=f"Feature Importance - {method.replace('_', ' ').title()}"
            )
            plt.savefig(os.path.join(output_dir, f'{method}_importance.png'))
            plt.close()

    # 2. Convolutional Filter Analysis
    if 'conv_filters' in results:
        plt.figure(figsize=(10, 6))
        filter_data = results['conv_filters']

        # Create boxplot for each kernel size
        plt.boxplot([weights for weights in filter_data.values()],
                    labels=[f'Kernel {k}' for k in filter_data.keys()])
        plt.title('Convolutional Filter Importance Distribution')
        plt.ylabel('Filter Weight Magnitude')
        plt.xlabel('Kernel Size')
        plt.savefig(os.path.join(output_dir, 'conv_filter_importance.png'))
        plt.close()

    # 3. Feature Importance Correlation
    if all(method in results for method in methods):
        plt.figure(figsize=(8, 8))

        # Create correlation matrix
        importance_matrix = np.array([
            list(results[method].values()) for method in methods
        ])
        correlation = np.corrcoef(importance_matrix)

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


def main():
    # Example usage
    model_paths = {
        'conversation': 'models/conversation_classifier_20241129_041816.pt',
        'author': 'models/author_classifier_20241129_042403.pt'
    }
    model_type = 'conversation'
    dataset_path = 'data/analyzed_conversations'

    print("Analyzing feature importance...")
    results = analyze_feature_importance(
        model_path=model_paths[model_type],
        dataset_path=dataset_path,
        device='cuda',
        model_type=model_type
    )

    print("\nCreating visualizations...")
    visualize_all_results(results)

    # Print top 10 most important features for each method
    for method in results:
        if method == 'conv_filters':
            continue
        print(f"\nTop 10 features by {method}:")
        sorted_features = sorted(
            results[method].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
