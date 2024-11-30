import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List

import datasets
from tqdm import tqdm


class ConversationParser:
    """
    Loads and parses XML conversation files into a HuggingFace datasets.Dataset.
    """

    def __init__(self, xml_path: str, max_conversations: int = -1) -> None:
        """
        Initializes the ConversationLoader with the path to the XML file.

        Args:
            xml_path (str): Path to the XML file containing conversation data.
            max_conversations (int, optional): Maximum number of conversations to load. Defaults to 100.
        """
        self.xml_path = xml_path
        self.max_conversations = max_conversations
        self.dataset = None

    @staticmethod
    def parse_time(time_str: str) -> datetime:
        """
        Converts a time string to a datetime object.

        Args:
            time_str (str): Time string in the format '%H:%M'.

        Returns:
            datetime: Parsed datetime object.
        """
        return datetime.strptime(time_str, '%H:%M')

    def load_conversations(self) -> List[Dict[str, Any]]:
        """
        Parses the XML file and extracts conversation data.

        Returns:
            List[Dict[str, Any]]: List of conversations with their messages and metadata.
        """
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            if self.max_conversations > 0:
                limit = min(self.max_conversations, len(root.findall('conversation')))
            else:
                limit = len(root.findall('conversation'))
            conversations = root.findall('conversation')[:limit]
            print(f"Loaded {len(conversations)} conversations from {self.xml_path}.")
        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            return []
        except FileNotFoundError:
            print(f"XML file not found at {self.xml_path}.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while loading conversations: {e}")
            return []

        conversation_data = []
        for conversation in tqdm(conversations, desc="Loading conversations"):
            conv_id = conversation.get('id')
            if not conv_id:
                print("Conversation without an ID found. Skipping.")
                continue

            messages = sorted(
                conversation.findall('message'),
                key=lambda m: int(m.get('line', '0'))
            )

            conv_features = {
                'conversation_id': conv_id,
                'messages': [],
                'authors': [],
                'timestamps': [],
            }

            for message in messages:
                author = message.find('author').text.strip()
                time_str = message.find('time').text.strip()
                time = self.parse_time(time_str)
                text = message.find('text').text or ""

                conv_features['messages'].append(text)
                conv_features['authors'].append(author)
                conv_features['timestamps'].append(time)

            conversation_data.append(conv_features)

        return conversation_data

    def create_dataset(self) -> datasets.Dataset:
        """
        Creates a HuggingFace Dataset from the loaded conversation data.

        Returns:
            datasets.Dataset: Dataset containing the conversations.
        """
        conversation_data = self.load_conversations()
        if not conversation_data:
            raise ValueError("No conversation data loaded.")

        self.dataset = datasets.Dataset.from_list(conversation_data)
        print(f"Created dataset with {len(self.dataset)} conversations.")
        return self.dataset

    def save_to_disk(self, filepath: str) -> None:
        """
        Saves the dataset to a directory.

        Args:
            filepath (str): Path to the output directory.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not created yet. Call create_dataset() first.")

        try:
            self.dataset.save_to_disk(filepath)
            print(f"Dataset successfully saved to {filepath}.")
        except Exception as e:
            print(f"Failed to save dataset to {filepath}: {e}")

    def load_from_disk(self, filepath: str) -> datasets.Dataset:
        """
        Loads a dataset from a directory.

        Args:
            filepath (str): Path to the directory containing the dataset.

        Returns:
            datasets.Dataset: Loaded dataset.
        """
        try:
            self.dataset = datasets.load_from_disk(filepath)
            print(f"Loaded dataset from {filepath} with {len(self.dataset)} conversations.")
            return self.dataset
        except Exception as e:
            print(f"Failed to load dataset from {filepath}: {e}")
            raise e

def main():
    """
    Main function to execute the conversation loading workflow.
    """
    # Configuration
    xml_path = 'data/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    output_dir = 'data/conversations'

    # Initialize loader
    loader = ConversationParser(xml_path=xml_path, max_conversations=-1)

    # Create dataset
    loader.create_dataset()

    # Save dataset to JSON
    loader.save_to_disk(output_dir)

    # Optionally, load the dataset back
    loaded_dataset = loader.load_from_disk(output_dir)


if __name__ == "__main__":
    main()