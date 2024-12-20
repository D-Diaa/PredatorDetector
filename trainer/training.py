import json
import logging
import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datahandler import (
    ConversationSequenceDataset,
    AuthorConversationSequenceDataset,
    create_dataloaders
)
from extractors import feature_sets
from trainer.models import SequenceClassifier, ProfileClassifier
from trainer.utils import estimate_pos_weight, calculate_metrics_threshold, evaluate, train_epoch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(model_type, model_path, dataset_path, device='cuda', seed=42):
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create dataset
    if model_type == 'conversation':
        dataset = ConversationSequenceDataset(dataset_path, max_seq_length=256, min_seq_length=8)
    elif model_type == 'author':
        dataset = AuthorConversationSequenceDataset(dataset_path, max_seq_length=256, min_seq_length=8)
    else:
        raise ValueError(f'Invalid model type: {model_type}')
    # Create dataloaders
    _, _, test_loader = create_dataloaders(
        dataset, batch_size=256
    )

    # Initialize model
    input_size = len(dataset.feature_keys)
    model = SequenceClassifier(input_size=input_size).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']

    # Calculate pos_weight for weighted BCE loss
    pos_weight = torch.tensor(
        estimate_pos_weight(dataset.labels.numpy(), method='focal')
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Evaluate on test set
    test_loss, test_preds, test_labels, test_metrics, _ = evaluate(
        model, test_loader, criterion, device, threshold
    )

    # Print final results
    logger.info('\nTest Set Results:')
    logger.info(f'Loss: {test_loss:.4f}')
    logger.info(f'Metrics with optimal threshold {threshold:.4f}:')
    for metric, value in test_metrics.items():
        logger.info(f'{metric}: {value:.4f}')
    return test_metrics


def train_model(model_type='conversation',
                dataset_path='data/analyzed_conversations',
                feature_set='models',
                feature_keys=None,
                num_epochs=50,
                batch_size=256,
                learning_rate=1e-4,
                device='cuda',
                seed=42):
    """
    Train either a conversation-level or author-level classifier
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    directory = os.path.join('models', feature_set, model_type)
    os.makedirs(directory, exist_ok=True)
    # Create dataset
    if model_type == 'conversation':
        dataset = ConversationSequenceDataset(dataset_path, max_seq_length=256, min_seq_length=8,
                                              feature_keys=feature_keys)
    elif model_type == 'author':
        dataset = AuthorConversationSequenceDataset(dataset_path, max_seq_length=256, min_seq_length=8,
                                                    feature_keys=feature_keys)
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=batch_size
    )

    # Initialize model
    input_size = len(dataset.feature_keys)
    model = SequenceClassifier(input_size=input_size).to(device)

    # Calculate pos_weight for weighted BCE loss
    pos_weight = torch.tensor(
        estimate_pos_weight(dataset.labels.numpy(), method='focal')
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    best_val_f1 = 0
    best_threshold = 0.5
    best_model_path = f'{directory}/classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'

    for epoch in range(num_epochs):
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_metrics, _ = calculate_metrics_threshold(train_labels, train_preds)

        # Validate
        val_loss, val_preds, val_labels, val_metrics, new_threshold = evaluate(
            model, val_loader, criterion, device
        )

        # Update learning rate based on F1 score
        scheduler.step(val_metrics['f1'])

        # Save best model based on F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_threshold = new_threshold
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold,
                'metrics': val_metrics
            }, best_model_path)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info(f'Train - Loss: {train_loss:.4f}, Metrics: {train_metrics}')
        logger.info(f'Val - Loss: {val_loss:.4f}, Metrics: {val_metrics}')
        logger.info(f'Best threshold: {best_threshold:.4f}')

    # Load best model and evaluate on test set
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint['threshold']

    test_loss, test_preds, test_labels, test_metrics, _ = evaluate(
        model, test_loader, criterion, device, best_threshold
    )

    # Print final results
    logger.info('\nTest Set Results:')
    logger.info(f'Loss: {test_loss:.4f}')
    logger.info(f'Metrics with optimal threshold {best_threshold:.4f}:')
    for metric, value in test_metrics.items():
        logger.info(f'{metric}: {value:.4f}')
    with open(f"{directory}/test_results.json", "w") as f:
        json.dump(test_metrics, f)
    return best_model_path


def evaluate_profile_classifier(
        author_model_path: str,
        dataset_path: str,
        device: str = 'cuda',
        aggregation: str = 'mean'
) -> Dict[str, float]:
    """
    Evaluate a profile classifier using a pretrained author classifier.

    Args:
        author_model_path: Path to pretrained author classifier checkpoint
        dataset_path: Path to the analyzed conversations dataset
        device: Device to run evaluation on
        aggregation: Method to aggregate sequence predictions ('mean', 'median', or 'majority')

    Returns:
        Dictionary containing evaluation metrics
    """
    # Create profile dataset
    profile_dataset = AuthorConversationSequenceDataset(
        dataset_path=dataset_path,
        max_seq_length=256,
        min_seq_length=8
    )
    authors = profile_dataset.get_authors()

    # Load pretrained author classifier
    author_classifier = SequenceClassifier(input_size=len(profile_dataset.feature_keys)).to(device)
    checkpoint = torch.load(author_model_path, weights_only=False)
    author_classifier.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']

    # Create profile classifier
    profile_classifier = ProfileClassifier(
        sequence_classifier=author_classifier,
        threshold=threshold,
        aggregation=aggregation
    ).to(device)

    # Evaluate profile by profile
    profile_classifier.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for author in tqdm(authors, desc='Evaluating profiles'):
            # Get single profile data
            sequences, label = profile_dataset.get_sequences_by_author(author)
            sequences = sequences.to(device)

            # Get profile-level prediction
            prob = profile_classifier(sequences)

            predictions.append(prob.cpu().item())
            true_labels.append(label)

    # Calculate metrics
    metrics, _ = calculate_metrics_threshold(
        np.array(true_labels),
        np.array(predictions),
    )

    # Log results
    logger.info('\nProfile Classifier Results:')
    logger.info(f'Aggregation method: {aggregation}')
    logger.info(f'Number of profiles evaluated: {len(predictions)}')
    for metric, value in metrics.items():
        logger.info(f'{metric}: {value:.4f}')

    return metrics


def compare_aggregation_methods(
        author_model_path: str,
        dataset_path: str
) -> Dict[str, Dict[str, float]]:
    """Compare different aggregation methods for profile classification."""
    results = {}
    for method in ['mean', 'median', 'mean_vote', 'total_vote']:
        logger.info(f'\nEvaluating {method} aggregation...')
        metrics = evaluate_profile_classifier(
            author_model_path=author_model_path,
            dataset_path=dataset_path,
            aggregation=method
        )
        results[method] = metrics

    # Find best method
    best_method = max(results.items(), key=lambda x: x[1]['f1'])[0]
    logger.info(f'\nBest aggregation method: {best_method}')
    return results


def eval_main(
        conversation_model_path='models/conversation_classifier_20241129_041816.pt',
        author_model_path='models/author_classifier_20241129_042403.pt'
):
    # Evaluate conversation classifier
    logger.info('Evaluating Conversation Classifier...')
    test_model(model_type='conversation', model_path=conversation_model_path,
               dataset_path='data/analyzed_conversations')
    # Evaluate author classifier
    logger.info('\nEvaluating Author Classifier...')
    test_model(model_type='author', model_path=author_model_path, dataset_path='data/analyzed_conversations')
    # Evaluate profile classifier with different aggregation methods
    logger.info('\nEvaluating Profile Classifier...')
    compare_aggregation_methods(
        author_model_path=author_model_path,
        dataset_path='data/analyzed_conversations'
    )


def train_main(feature_keys, feature_set_key):
    # Train conversation classifier
    logger.info('Training Conversation Classifier...')
    conv_model_path = train_model(
        model_type='conversation',
        feature_set=feature_set_key,
        feature_keys=feature_keys,
        num_epochs=100,
        batch_size=256,
        learning_rate=1e-4
    )
    print(f"Conversation model path: {conv_model_path}")
    # # Train author classifier
    logger.info('\nTraining Author Classifier...')
    author_model_path = train_model(
        model_type='author',
        feature_set=feature_set_key,
        feature_keys=feature_keys,
        num_epochs=100,
        batch_size=256,
        learning_rate=1e-4
    )
    print(f"Author model path: {author_model_path}")


if __name__ == '__main__':
    # for set_key in feature_sets.keys():
    set_key = "best"
    feature_keys = [f"feat_{feat}" for feat in feature_sets[set_key]]
    train_main(feature_keys, set_key)
