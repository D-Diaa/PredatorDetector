import gc
from collections import defaultdict
from typing import Any, Dict, List

import datasets
import pandas as pd
from tqdm import tqdm

from extractors import (
    SentimentExtractor,
    EmotionExtractor,
    ToxicityExtractor,
    IntentExtractor,
    Word2AffectExtractor
)


def get_conversation_statistics(analyzed_dataset: datasets.Dataset) -> pd.DataFrame:
    """Get conversation statistics from the analyzed dataset."""
    conv_stats = []
    for conversation in analyzed_dataset:
        stats = {
            'conversation_id': conversation['conversation_id'],
            'num_participants': conversation['num_participants'],
            'num_messages': conversation['num_messages'],
        }
        conv_stats.append(stats)
    return pd.DataFrame(conv_stats)


class ConversationAnalyzer:
    """
    Memory-efficient conversation analyzer that processes data using the map function.
    """

    def __init__(self, batch_size: int = 100) -> None:
        """
        Initialize the streaming analyzer.

        Args:
            batch_size (int): Number of conversations to process in each batch
        """
        self.batch_size = batch_size
        self.extractors = None
        self.attackers = set()
        self._initialize_extractors()

    def _initialize_extractors(self) -> None:
        """Initialize extractors only when needed to save memory."""
        if self.extractors is None:
            self.extractors = [
                SentimentExtractor(config={'device': 'cuda:0', 'batch_size': 32}),
                EmotionExtractor(config={'device': 'cuda:0', 'batch_size': 32}),
                ToxicityExtractor(config={'device': 'cuda:0', 'batch_size': 32}),
                IntentExtractor(config={'device': 'cuda:0',  'batch_size': 32}),
                Word2AffectExtractor(config={'device': 'cuda:0', 'batch_size': 32}, directory="word2affect_english")
            ]

    def load_attackers(self, attackers_path: str) -> None:
        """Load known attackers from file."""
        try:
            with open(attackers_path, 'r', encoding='utf-8') as f:
                self.attackers = {line.strip() for line in f if line.strip()}
            print(f"Loaded {len(self.attackers)} attackers from {attackers_path}.")
        except Exception as e:
            print(f"Error loading attackers: {e}")
            self.attackers = set()

    def process_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Process the dataset using the map function to avoid loading the entire dataset into memory.

        Args:
            dataset: Input dataset containing conversations

        Returns:
            datasets.Dataset: The complete processed dataset
        """
        processed_dataset = dataset.map(
            self.process_conversations,
            batched=True,
            batch_size=self.batch_size,
            desc="Processing conversations"
        )
        return processed_dataset

    def process_conversations(self, conversations: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Process multiple conversations in batch, extracting features from all messages at once.

        Args:
            conversations: Dictionary containing lists of conversation data including
                         messages, authors, timestamps, etc.

        Returns:
            Dict[str, List[Any]]: Processed conversations with extracted features
        """
        # Flatten all messages for batch feature extraction
        all_messages = [msg for conv_messages in conversations['messages'] for msg in conv_messages]

        # Extract features for all messages at once
        all_features = self._extract_batch_features(all_messages)

        # Initialize the processed conversations dictionary
        processed = defaultdict(list)
        processed['conversation_id'] = conversations['conversation_id']
        # Track position in the flattened features list
        feature_idx = 0

        # Process each conversation
        for conv_idx in range(len(conversations['messages'])):
            messages = conversations['messages'][conv_idx]
            authors = conversations['authors'][conv_idx]
            timestamps = conversations['timestamps'][conv_idx]
            # Initialize feature aggregation for this conversation
            conv_features = defaultdict(list)
            first_time = timestamps[0]
            participant_set = set()

            # Process each message in the conversation
            for msg_idx, (author, timestamp, message) in enumerate(zip(authors, timestamps, messages)):
                participant_set.add(author)

                # Calculate time delta
                time_delta = (timestamp.hour * 60 + timestamp.minute) - \
                             (first_time.hour * 60 + first_time.minute)
                if time_delta < -720:
                    time_delta += 1440

                # Get features for this message from the batch-extracted features
                features = all_features[feature_idx]
                feature_idx += 1

                # Aggregate features
                for key, value in features.items():
                    conv_features[key].append(value)
                conv_features['message_lengths'].append(len(message))
                conv_features['time_deltas'].append(time_delta)

            # Add conversation-level data
            attackers = [author for author in participant_set if author in self.attackers] or ["none"]
            processed['attackers'].append(attackers)
            # Add to processed results
            for key, values in conv_features.items():
                processed[f"feat_{key}"].append(values)
            processed['num_messages'].append(len(messages))
            processed['num_participants'].append(len(participant_set))

        return processed

    def process_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a single conversation using all extractors."""
        messages = conversation['messages']
        features_list = self._extract_batch_features(messages)

        conversation['features'] = defaultdict(list)
        first_time = conversation['timestamps'][0]
        participant_set = set()

        for idx, (author, features, timestamp, message) in enumerate(zip(
                conversation['authors'],
                features_list,
                conversation['timestamps'],
                messages
        )):
            participant_set.add(author)
            time_delta = (timestamp.hour * 60 + timestamp.minute) - \
                         (first_time.hour * 60 + first_time.minute)
            if time_delta < -720:
                time_delta += 1440

            for key, value in features.items():
                conversation['features'][key].append(value)
            conversation['features']['message_lengths'].append(len(message))
            conversation['features']['time_deltas'].append(time_delta)

        conversation['attackers'] = [author for author in participant_set if author in self.attackers]
        conversation['num_messages'] = len(messages)

        return conversation

    def _extract_batch_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract features from a batch of texts using all extractors concurrently."""
        features_list = [{} for _ in texts]

        for extractor in self.extractors:
            try:
                extractor_features = extractor.batch_extract(texts)
                for i, feature in enumerate(extractor_features):
                    features_list[i].update(feature)
            except Exception as e:
                print(f"Feature extraction error in {extractor.__class__.__name__}: {e}")

        return features_list

    def collect_user_statistics(self, dataset: datasets.Dataset) -> pd.DataFrame:
        """Collect user statistics from the processed dataset."""
        user_stats = defaultdict(lambda: {
            'message_count': 0,
            'conversations': set(),
            'is_attacker': False
        })

        for conversation in tqdm(dataset, desc="Collecting user statistics"):
            conv_id = conversation['conversation_id']
            for author in conversation['authors']:
                user_stats[author]['message_count'] += 1
                user_stats[author]['conversations'].add(conv_id)
                if author in self.attackers:
                    user_stats[author]['is_attacker'] = True

        # Convert to DataFrame
        user_stats_list = []
        for user, stats in user_stats.items():
            user_stats_list.append({
                'user': user,
                'message_count': stats['message_count'],
                'num_conversations': len(stats['conversations']),
                'is_attacker': stats['is_attacker']
            })

        return pd.DataFrame(user_stats_list)


def main():
    """Main function to execute the streaming conversation analysis workflow."""
    # Configuration
    conversation_directory = 'data/conversations'
    attackers_path = 'data/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
    output_directory = 'data/analyzed_conversations'
    user_stats_output = 'data/user_statistics.csv'
    conv_stats_output = 'data/conversation_statistics.csv'

    try:
        # Initialize analyzer
        analyzer = ConversationAnalyzer(batch_size=16)

        # Load attackers
        analyzer.load_attackers(attackers_path)

        # Load dataset
        print("Loading dataset...")
        dataset = datasets.load_from_disk(conversation_directory)
        print(f"Loaded {len(dataset)} conversations.")

        # Process conversations using map
        print("Processing conversations...")
        analyzed_dataset = analyzer.process_dataset(dataset)

        # Save the final dataset
        print(f"Saving analyzed dataset to {output_directory}...")
        analyzed_dataset.save_to_disk(output_directory)
        print(f"Saved analyzed dataset to {output_directory}")

        # Collect and save user statistics
        print("Collecting user statistics...")
        user_stats_df = analyzer.collect_user_statistics(analyzed_dataset)
        user_stats_df.to_csv(user_stats_output, index=False)
        print(f"Saved user statistics to {user_stats_output}")

        # Get and save conversation statistics
        print("Generating conversation statistics...")
        conv_stats_df = get_conversation_statistics(analyzed_dataset)
        conv_stats_df.to_csv(conv_stats_output, index=False)
        print(f"Saved conversation statistics to {conv_stats_output}")

        # Print summary
        print("\nDataset Statistics:")
        print(f"Total conversations: {len(analyzed_dataset)}")
        print(f"Total users: {len(user_stats_df)}")
        print(f"Total attackers: {user_stats_df['is_attacker'].sum()}")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        # Clean up
        gc.collect()


if __name__ == "__main__":
    main()
