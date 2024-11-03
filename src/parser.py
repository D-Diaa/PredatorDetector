import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from extractors import SentimentExtractor, EmotionExtractor, ToxicityExtractor, IntentExtractor


class ConversationAnalyzer:
    def __init__(self, xml_path: str, attackers_path: str):
        """
        Initialize the analyzer with paths to the XML conversation data and attackers list.
        """
        self.xml_path = xml_path
        self.attackers_path = attackers_path
        self.attackers = set()
        self.conversations = {}
        self.extractors = [
            # LinguisticExtractor(),
            SentimentExtractor(),
            EmotionExtractor(),
            ToxicityExtractor(),
            IntentExtractor(),
            # KeywordExtractor(),
            # MultiLexiconEmotionExtractor()
        ]

        self.extractor_keys = []
        for extractor in self.extractors:
            self.extractor_keys.extend(extractor.keys)

        self.user_stats = defaultdict(lambda: {
            'message_count': 0,
            'conversations': set(),
            'is_attacker': False
        })

    def load_attackers(self) -> None:
        """Load the list of known attackers from the text file."""
        with open(self.attackers_path, 'r') as f:
            self.attackers = set(line.strip() for line in f)

    def parse_time(self, time_str: str) -> datetime:
        """Convert time string to datetime object."""
        return datetime.strptime(time_str, '%H:%M')

    def calculate_message_features(self, text: str) -> Dict:
        """Calculate various features from a message text."""
        total_chars = len(text)
        if total_chars == 0:
            ret = {dim: np.nan for dim in self.extractor_keys}
            return ret

        ret = {}
        for extractor in self.extractors:
            features = extractor.extract(text)
            ret.update(features)
        return ret

    def parse_conversations(self) -> None:
        """Parse the XML file and extract conversation data with features."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        conversations = list(root.findall('conversation'))[:10]
        for conversation in tqdm(conversations, desc="Parsing conversations"):
            conv_id = conversation.get('id')

            # Initialize conversation feature lists
            conv_features = {
                # Message content and metadata
                'messages': [],  # List of actual message texts
                'authors': [],  # List of message authors
                'timestamps': [],  # List of message timestamps
                'response_times': [],  # Time between messages in minutes

                # Conversation-level features
                'duration_minutes': 0,  # Total conversation duration
                'num_participants': 0,  # Number of unique participants
                'num_messages': 0,  # Total number of messages
                'avg_response_time': 0,  # Average time between messages
                'time_of_day': 0,  # Hour when conversation started (0-23)
                'max_msg_per_user': 0,  # Maximum messages from a single user
                'unique_words': set(),  # Set of unique words used
                'vocabulary_size': 0,  # Number of unique words
                'participant_set': set()  # Set of participant IDs
            }

            conv_features.update(
                {dim: [] for dim in self.extractor_keys}
            )

            # Sort messages by line number
            messages = sorted(
                conversation.findall('message'),
                key=lambda m: int(m.get('line'))
            )

            # Process each message
            prev_time = None
            user_message_counts = defaultdict(int)

            for message in messages:
                author = message.find('author').text.strip()
                time = self.parse_time(message.find('time').text.strip())
                text = message.find('text').text or ""

                # Update user message counts
                user_message_counts[author] += 1

                # Calculate message features
                features = self.calculate_message_features(text)

                # Calculate response time
                if prev_time:
                    response_time = (time.hour * 60 + time.minute) - \
                                    (prev_time.hour * 60 + prev_time.minute)
                    if response_time < -720:  # More than 12 hours negative
                        response_time += 1440  # Add 24 hours
                    conv_features['response_times'].append(response_time)
                else:
                    conv_features['response_times'].append(0)

                # Append all features to lists
                conv_features['messages'].append(text)
                conv_features['authors'].append(author)
                conv_features['timestamps'].append(time)

                for dim in self.extractor_keys:
                    conv_features[dim].append(features[dim])

                # Update unique words
                conv_features['unique_words'].update(text.lower().split())

                # Update participant set
                conv_features['participant_set'].add(author)

                prev_time = time

                # Update user statistics
                self.user_stats[author]['message_count'] += 1
                self.user_stats[author]['conversations'].add(conv_id)
                self.user_stats[author]['is_attacker'] = author in self.attackers

            # Calculate conversation-level features
            first_msg_time = conv_features['timestamps'][0]
            last_msg_time = conv_features['timestamps'][-1]

            conv_features['duration_minutes'] = (
                    (last_msg_time.hour * 60 + last_msg_time.minute) -
                    (first_msg_time.hour * 60 + first_msg_time.minute)
            )
            if conv_features['duration_minutes'] < 0:
                conv_features['duration_minutes'] += 1440  # Add 24 hours

            conv_features['num_participants'] = len(conv_features['participant_set'])
            conv_features['num_messages'] = len(conv_features['messages'])
            conv_features['avg_response_time'] = np.mean(conv_features['response_times']) if conv_features[
                'response_times'] else 0
            conv_features['time_of_day'] = first_msg_time.hour
            conv_features['max_msg_per_user'] = max(user_message_counts.values())
            conv_features['vocabulary_size'] = len(conv_features['unique_words'])

            # Store conversation features
            self.conversations[conv_id] = conv_features

    def get_conversation_statistics(self):
        """Get conversation statistics as a DataFrame."""
        conv_stats = []
        for conv_id, features in self.conversations.items():
            stats = {
                'conversation_id': conv_id,
                'num_participants': features['num_participants'],
                'num_messages': features['num_messages'],
                'duration_minutes': features['duration_minutes'],
                'avg_response_time': features['avg_response_time'],
                'time_of_day': features['time_of_day'],
                'max_msg_per_user': features['max_msg_per_user'],
                'vocabulary_size': features['vocabulary_size']
            }
            conv_stats.append(stats)
        return pd.DataFrame(conv_stats)


def main():
    # Initialize analyzer with file paths
    directory = 'data'
    paths = {
        "xml_path": f"{directory}/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml",
        "attackers_path": f"{directory}/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
    }
    analyzer = ConversationAnalyzer(**paths)
    # Load and parse data
    analyzer.load_attackers()
    analyzer.parse_conversations()
    conversations = analyzer.conversations
    user_stats = analyzer.user_stats
    # Save conversation data to JSON
    with open('conversations.json', 'w') as f:
        json.dump(conversations, f, default=str)
    # Save user statistics to JSON
    with open('user_stats.json', 'w') as f:
        json.dump(user_stats, f, default=str)

    # Get conversation statistics
    conv_stats = analyzer.get_conversation_statistics()
    # Print summary statistics
    print("\nDataset Statistics:")
    print(f"Total conversations: {len(analyzer.conversations)}")
    print(f"Total users: {len(analyzer.user_stats)}")
    print(f"Total attackers: {sum(1 for _, stats in analyzer.user_stats.items() if stats['is_attacker'])}")

    # Print conversation statistics summary
    print("\nConversation Statistics Summary:")
    print(conv_stats.describe())
    conv_stats.to_csv('conversation_stats.csv', index=False)


if __name__ == "__main__":
    main()
