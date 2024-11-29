import hashlib
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple

import datasets
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm


def check_data(loader):
    """Check for NaN or Inf values in the data loader batches."""
    for batch_idx, batch in enumerate(loader):
        sequences = batch['sequence']
        labels = batch['label']
        if torch.isnan(sequences).any() or torch.isinf(sequences).any():
            print(f'NaN or Inf detected in sequences at batch {batch_idx}')
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print(f'NaN or Inf detected in labels at batch {batch_idx}')


class BaseDataset(Dataset):
    """Base dataset class with caching, memory-mapped loading, batch processing, vectorized operations,
    and normalization."""

    def __init__(self,
                 dataset_path: str,
                 feature_keys: List[str] = None,
                 max_seq_length: Optional[int] = None,
                 min_seq_length: int = 5,
                 pad_value: float = 0.0,
                 cache_prefix: str = "base",
                 normalize: bool = True):
        """
        Args:
            dataset_path: Path to the analyzed conversations dataset
            feature_keys: List of feature names to include
            max_seq_length: Maximum sequence length (will pad/truncate to this length)
            min_seq_length: Minimum sequence length to include
            pad_value: Value to use for padding sequences
            cache_prefix: Prefix for cache file naming
            normalize: Whether to apply standard normalization (zero mean, unit variance)
        """
        self.dataset_path = dataset_path
        self.feature_keys = feature_keys
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.pad_value = pad_value
        self.normalize = normalize

        # Define cache file path
        dataset_mod_time = os.path.getmtime(dataset_path)
        cache_filename = f"{cache_prefix}_dataset_cache_{self._hash_features()}.pkl"
        self.cache_path = os.path.join(os.path.dirname(dataset_path), cache_filename)

        # Initialize normalization parameters
        self.mean = None
        self.std = None

        if self._is_cache_valid(dataset_mod_time):
            print(f"Loading {cache_prefix} dataset from cache...")
            self._load_cache()
        else:
            print(f"Processing {cache_prefix} dataset...")
            self._process_dataset()
            self._save_cache()

    def _hash_features(self):
        """Create a hash based on feature keys and parameters to uniquely identify the cache."""
        feature_str = "_".join(self.feature_keys) if self.feature_keys else "all_features"
        params = f"max_seq_length={self.max_seq_length}_min_seq_length={self.min_seq_length}_pad_value={self.pad_value}_normalize={self.normalize}"
        hash_input = feature_str + "_" + params + "_" + self.dataset_path
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _is_cache_valid(self, dataset_mod_time):
        """Check if cache exists and is up-to-date."""
        if not os.path.exists(self.cache_path):
            return False
        cache_mod_time = os.path.getmtime(self.cache_path)
        if cache_mod_time < dataset_mod_time:
            print("Cache is outdated.")
            return False
        return True

    def _load_cache(self):
        """Load processed data and normalization parameters from cache."""
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            self.processed_sequences = cache_data['processed_sequences']
            self.labels = cache_data['labels']
            self.feature_keys = cache_data['feature_keys']
            if 'metadata' in cache_data:
                self.metadata = cache_data['metadata']
            if 'mean' in cache_data and 'std' in cache_data:
                self.mean = cache_data['mean']
                self.std = cache_data['std']
            if 'author_to_indices' in cache_data:
                self.author_to_indices = cache_data['author_to_indices']
            print(f"Loaded {len(self.labels)} samples from cache.")

    def _save_cache(self):
        """Save processed data and normalization parameters to cache."""
        cache_data = {
            'processed_sequences': self.processed_sequences,
            'labels': self.labels,
            'feature_keys': self.feature_keys,
            'mean': self.mean,
            'std': self.std
        }
        if hasattr(self, 'metadata'):
            cache_data['metadata'] = self.metadata
        if hasattr(self, 'author_to_indices'):
            cache_data['author_to_indices'] = self.author_to_indices
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved processed data to cache: {self.cache_path}")

    def _process_dataset(self):
        """Process the dataset, compute normalization parameters, normalize sequences, and prepare labels."""
        # Load dataset with memory mapping
        with datasets.load_from_disk(self.dataset_path, keep_in_memory=False) as analyzed_dataset:
            print(f"Loaded {len(analyzed_dataset)} conversations")
            self.conversations = conversations = [conv for conv in analyzed_dataset]

            if not conversations:
                raise ValueError("The dataset is empty.")

            # Determine feature keys
            if self.feature_keys is None:
                keys = [key for key in conversations[0].keys() if key.startswith('feat')]
                self.feature_keys = sorted(keys)

            num_features = len(self.feature_keys)

            # Initialize lists
            self.processed_sequences = []
            self.labels = []
            if hasattr(self, 'metadata_needed') and self.metadata_needed:
                self.metadata = []

            # Initialize variables for normalization
            if self.normalize:
                sum_features = np.zeros(num_features, dtype=np.float64)
                sum_sq_features = np.zeros(num_features, dtype=np.float64)
                count = 0

            # Define batch parameters
            batch_size = 1024
            num_batches = len(conversations) // batch_size + (1 if len(conversations) % batch_size != 0 else 0)

            # First pass: Compute sum and sum of squares for normalization
            print("Computing normalization parameters...")
            for batch_num in tqdm(range(num_batches), desc="Computing normalization"):
                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size
                batch_conversations = conversations[start_idx:end_idx]

                for conv in batch_conversations:
                    if len(conv['authors']) < self.min_seq_length:
                        continue

                    try:
                        # Extract features up to max_seq_length
                        seq_length = min(len(conv['authors']), self.max_seq_length) if self.max_seq_length else len(
                            conv['authors'])
                        features = np.array([conv[key][:seq_length] for key in self.feature_keys],
                                            dtype=np.float64).T  # Shape: (seq_length, num_features)

                        # Update normalization statistics
                        sum_features += features.sum(axis=0)
                        sum_sq_features += (features ** 2).sum(axis=0)
                        count += features.shape[0]
                    except Exception as e:
                        print(
                            f"Error processing conversation ID {conv.get('conversation_id', 'N/A')} for normalization: {e}")
                        continue

            if self.normalize:
                self.mean = sum_features / count
                variance = (sum_sq_features / count) - (self.mean ** 2)
                self.std = np.sqrt(variance)
                # To avoid division by zero
                self.std[self.std == 0] = 1.0
                print(f"Computed mean: {self.mean}")
                print(f"Computed std: {self.std}")

            # Second pass: Process and normalize sequences
            print("Processing and normalizing sequences...")
            for batch_num in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size
                batch_conversations = conversations[start_idx:end_idx]

                for conv in batch_conversations:
                    # Implement dataset-specific processing in subclasses
                    self._process_conversation(conv)

                # Clear intermediate data
                del batch_conversations
                torch.cuda.empty_cache()  # If using CUDA

            # Convert labels to tensor
            self.labels = torch.LongTensor(self.labels)
            print(f"Created dataset with {len(self.labels)} samples")

    def _process_conversation(self, conv):
        """Process a single conversation. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self):
        return len(self.processed_sequences)

    def __getitem__(self, idx):
        item = {
            'sequence': self.processed_sequences[idx],
            'label': self.labels[idx]
        }
        if hasattr(self, 'metadata'):
            item['metadata'] = self.metadata[idx]
        return item


class ConversationSequenceDataset(BaseDataset):
    """Dataset for conversation-level sequence classification"""

    def __init__(self, dataset_path: str,
                 feature_keys: List[str] = None,
                 max_seq_length: Optional[int] = None,
                 min_seq_length: int = 5,
                 pad_value: float = 0.0,
                 normalize: bool = True):
        super().__init__(dataset_path, feature_keys, max_seq_length, min_seq_length, pad_value, cache_prefix="conv",
                         normalize=normalize)

    def _process_conversation(self, conv):
        """Process a single conversation and append to processed_sequences and labels."""
        if len(conv['authors']) < self.min_seq_length:
            return

        sequence = self._create_sequence(conv)
        if sequence is not None:
            self.processed_sequences.append(sequence)
            label = 1 if len(conv['attackers']) > 0 and conv['attackers'] != ['none'] else 0
            self.labels.append(label)

    def _create_sequence(self, conv):
        """Create a feature sequence matrix for a single conversation using vectorized operations and apply normalization."""
        authors_length = len(conv['authors'])
        seq_length = min(authors_length, self.max_seq_length) if self.max_seq_length else authors_length

        # Initialize an array with pad_value
        if self.max_seq_length:
            sequence = np.full((self.max_seq_length, len(self.feature_keys)), self.pad_value, dtype=np.float32)
        else:
            sequence = np.full((seq_length, len(self.feature_keys)), self.pad_value, dtype=np.float32)

        # Extract all feature data at once
        try:
            features = np.array([conv[key][:seq_length] for key in self.feature_keys],
                                dtype=np.float32).T  # Shape: (seq_length, num_features)
        except IndexError as e:
            print(f"IndexError while processing conversation ID {conv.get('conversation_id', 'N/A')}: {e}")
            return None

        # Handle padding if necessary
        if self.max_seq_length and features.shape[0] < self.max_seq_length:
            pad_width = self.max_seq_length - features.shape[0]
            padding = np.full((pad_width, len(self.feature_keys)), self.pad_value, dtype=np.float32)
            features = np.vstack((features, padding))

        # Apply normalization if enabled
        if self.normalize and self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        sequence[:features.shape[0], :] = features

        # Check for NaN or Inf
        if np.isnan(sequence).any() or np.isinf(sequence).any():
            return None  # Skip sequences with invalid values

        # Convert to tensor
        return torch.FloatTensor(sequence)


class AuthorConversationSequenceDataset(BaseDataset):
    """Dataset for author-level sequence classification within conversations"""

    def __init__(self, dataset_path: str,
                 feature_keys: List[str] = None,
                 max_seq_length: Optional[int] = None,
                 min_seq_length: int = 5,
                 pad_value: float = 0.0,
                 normalize: bool = True):
        self.metadata_needed = True
        self.author_to_indices = defaultdict(list)
        super().__init__(dataset_path, feature_keys, max_seq_length, min_seq_length, pad_value, cache_prefix="author",
                         normalize=normalize)

    def _process_conversation(self, conv):
        """Process a single conversation and append to processed_sequences, labels, and metadata."""
        if len(conv['authors']) < self.min_seq_length:
            return

        author_sequences = self._create_author_sequences(conv)
        for author, sequence in author_sequences.items():
            if len(sequence) < self.min_seq_length:
                continue
            sequence_padded = self._pad_sequence(sequence)
            if sequence_padded is not None:
                self.processed_sequences.append(sequence_padded)
                label = 1 if author in conv['attackers'] else 0
                self.labels.append(label)
                self.metadata.append({
                    'conversation_id': conv.get('conversation_id', 'N/A'),
                    'author': author
                })
                self.author_to_indices[author].append(len(self.labels) - 1)

    def _create_author_sequences(self, conv):
        """Create feature sequences for each author in the conversation."""
        author_sequences = defaultdict(list)
        authors = conv['authors']

        for i, author in enumerate(authors):
            message_features = [conv[key][i] for key in self.feature_keys]
            author_sequences[author].append(message_features)

        return author_sequences

    def get_authors(self):
        return list(self.author_to_indices.keys())

    def get_sequences_by_author(self, author):
        indices = self.author_to_indices[author]
        sequences = [self.processed_sequences[idx] for idx in indices]
        label = self.labels[indices[0]]
        sequences = torch.stack(sequences)
        return sequences, label

    def _pad_sequence(self, sequence: List[List[float]]) -> Optional[torch.FloatTensor]:
        """Pad or truncate sequence to max_seq_length using vectorized operations and apply normalization."""
        seq_length = len(sequence)
        if self.max_seq_length:
            seq_length = min(seq_length, self.max_seq_length)
        else:
            seq_length = seq_length

        # Initialize an array with pad_value
        if self.max_seq_length:
            sequence_array = np.full((self.max_seq_length, len(self.feature_keys)), self.pad_value, dtype=np.float32)
        else:
            sequence_array = np.full((seq_length, len(self.feature_keys)), self.pad_value, dtype=np.float32)

        # Extract features
        try:
            features = np.array(sequence[:seq_length], dtype=np.float32)
        except ValueError as e:
            print(f"ValueError while padding sequence: {e}")
            return None

        # Apply normalization if enabled
        if self.normalize and self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        # Handle padding if necessary
        if self.max_seq_length and features.shape[0] < self.max_seq_length:
            pad_width = self.max_seq_length - features.shape[0]
            padding = np.full((pad_width, len(self.feature_keys)), self.pad_value, dtype=np.float32)
            features = np.vstack((features, padding))

        sequence_array[:features.shape[0], :] = features

        # Check for NaN or Inf
        if np.isnan(sequence_array).any() or np.isinf(sequence_array).any():
            return None  # Skip sequences with invalid values

        # Convert to tensor
        return torch.FloatTensor(sequence_array)


def create_dataloaders(
        dataset: BaseDataset,
        batch_size: int = 32,
        train_split: float = 0.8,
        val_split: float = 0.1,
        random_seed: int = 42,
        use_weighted_sampler: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create stratified train, validation, and test DataLoaders with optional weighted sampling.

    Args:
        dataset (Dataset): PyTorch Dataset with a 'labels' attribute.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
        train_split (float, optional): Proportion of data for training. Defaults to 0.8.
        val_split (float, optional): Proportion of data for validation. Defaults to 0.1.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        use_weighted_sampler (bool, optional): Use weighted sampling for training DataLoader. Defaults to True.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to copy Tensors into CUDA pinned memory. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Extract labels
    labels = np.array(dataset.labels)
    indices = np.arange(len(labels))

    # First split: Train + Val vs Test
    strat_split1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - (train_split + val_split), random_state=random_seed)
    train_val_idx, test_idx = next(strat_split1.split(indices, labels))

    # Second split: Train vs Val
    strat_split2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split / (train_split + val_split),
                                          random_state=random_seed)
    train_idx, val_idx = next(strat_split2.split(train_val_idx, labels[train_val_idx]))

    train_indices, val_indices = train_val_idx[train_idx], train_val_idx[val_idx]

    # Create sampler for training set
    if use_weighted_sampler:
        train_labels = labels[train_indices]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        # Handle potential division by zero
        class_weights = np.where(class_counts > 0, class_weights, 0.0)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float),
                                        num_samples=len(train_indices),
                                        replacement=True)
        drop_last = True
    else:
        sampler = SubsetRandomSampler(train_indices)
        drop_last = False

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices),
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx),
                             num_workers=num_workers, pin_memory=pin_memory)
    check_data(train_loader)
    check_data(val_loader)
    check_data(test_loader)
    return train_loader, val_loader, test_loader


def print_dataset_stats(dataset: BaseDataset, name: str) -> None:
    """Print statistics about a dataset"""
    total_samples = len(dataset)
    positive_labels = sum(dataset.labels == 1)
    negative_labels = sum(dataset.labels == 0)

    print(f"\n{name} Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples (attackers): {positive_labels} ({positive_labels / total_samples * 100:.2f}%)")
    print(f"Negative samples: {negative_labels} ({negative_labels / total_samples * 100:.2f}%)")
    if total_samples > 0:
        print(f"Sequence shape: {dataset[0]['sequence'].shape}")
        print(f"Number of features: {len(dataset.feature_keys)}")
        print(f"Feature keys: {dataset.feature_keys}")


def main(filepath: str):
    """
    Test the dataset implementations using data loaded from disk

    Args:
        filepath: Path to the analyzed conversations dataset
    """
    try:
        # Create conversation-level dataset
        print("\nCreating conversation-level dataset...")
        conv_dataset = ConversationSequenceDataset(
            dataset_path=filepath,
            max_seq_length=50
        )
        print_dataset_stats(conv_dataset, "Conversation-level Dataset")

        # Create author-level dataset
        print("\nCreating author-level dataset...")
        author_dataset = AuthorConversationSequenceDataset(
            dataset_path=filepath,
            max_seq_length=30
        )
        print_dataset_stats(author_dataset, "Author-level Dataset")

        # Test DataLoader creation for all datasets
        print("\nCreating DataLoaders...")
        batch_size = 16

        # Conversation-level dataloaders
        conv_train, conv_val, conv_test = create_dataloaders(
            conv_dataset, batch_size=batch_size
        )
        print(f"Conversation DataLoaders created successfully:")
        print(f"Train batches: {len(conv_train)}")
        print(f"Val batches: {len(conv_val)}")
        print(f"Test batches: {len(conv_test)}")

        # Author-level dataloaders
        author_train, author_val, author_test = create_dataloaders(
            author_dataset, batch_size=batch_size
        )
        print(f"\nAuthor DataLoaders created successfully:")
        print(f"Train batches: {len(author_train)}")
        print(f"Val batches: {len(author_val)}")
        print(f"Test batches: {len(author_test)}")

        # Test batch retrieval for each dataset type
        print("\nTesting batch retrieval...")

        # Conversation batch
        conv_batch = next(iter(conv_train))
        print("\nConversation batch shapes:")
        print(f"Sequences: {conv_batch['sequence'].shape}")
        print(f"Labels: {conv_batch['label'].shape}")

        # Author batch
        author_batch = next(iter(author_train))
        print("\nAuthor batch shapes:")
        print(f"Sequences: {author_batch['sequence'].shape}")
        print(f"Labels: {author_batch['label'].shape}")

        return conv_dataset, author_dataset

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    # The filepath should be provided as an argument when running the script
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <filepath>")
        sys.exit(1)

    filepath = sys.argv[1]
    conv_dataset, author_dataset = main(filepath)
