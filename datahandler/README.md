# DataHandler

## Overview

**DataHandler** is a Python package designed to streamline the process of loading, parsing, analyzing, and managing conversation datasets. It provides a suite of tools for handling conversation data, particularly in XML format, and offers functionalities for extracting various features, generating statistics, and preparing datasets for machine learning tasks.

## Key Components

### 1. Conversation Parser (`ConversationParser`)

-   **Purpose**: Loads and parses XML conversation files into a HuggingFace `datasets.Dataset`.
-   **Functionality**:
    -   Parses XML files containing conversation data [parser.py].
    -   Extracts conversation ID, messages, authors, and timestamps [parser.py].
    -   Converts time strings to `datetime` objects [parser.py].
    -   Creates a HuggingFace `datasets.Dataset` from the parsed data [parser.py].
    -   Saves the dataset to disk in a specified format [parser.py].
    -   Loads a dataset from disk [parser.py].
-   **Usage Example**:

```python
from datahandler import ConversationParser

# Initialize the parser
parser = ConversationParser(xml_path='path/to/conversations.xml', max_conversations=100)

# Create a dataset
dataset = parser.create_dataset()

# Save the dataset
parser.save_to_disk('path/to/output/directory')

# Load the dataset
loaded_dataset = parser.load_from_disk('path/to/output/directory')
```

### 2. Conversation Analyzer (`ConversationAnalyzer`)

-   **Purpose**: Analyzes conversation datasets to extract features and generate statistics.
-   **Functionality**:
    -   Processes conversation datasets using the `map` function for memory efficiency [analyzer.py].
    -   Extracts features such as sentiment, emotion, toxicity, intent, and word affect [analyzer.py].
    -   Calculates conversation-level statistics (e.g., number of participants, number of messages) [analyzer.py].
    -   Collects user statistics (e.g., message count, number of conversations, identification of attackers) [analyzer.py].
    -   Saves analyzed datasets and statistics to disk [analyzer.py].
-   **Usage Example**:

```python
from datahandler import ConversationAnalyzer

# Initialize the analyzer
analyzer = ConversationAnalyzer(batch_size=16)

# Load attackers (if applicable)
analyzer.load_attackers('path/to/attackers.txt')

# Load a dataset
dataset = datasets.load_from_disk('path/to/conversations')

# Process the dataset
analyzed_dataset = analyzer.process_dataset(dataset)

# Save the analyzed dataset
analyzed_dataset.save_to_disk('path/to/analyzed/conversations')

# Collect and save user statistics
user_stats_df = analyzer.collect_user_statistics(analyzed_dataset)
user_stats_df.to_csv('path/to/user_statistics.csv', index=False)
```

### 3. Dataset Classes and Data Loaders

-   **Purpose**: Provides classes for handling datasets with features like caching, memory-mapped loading, batch processing, and normalization.
-   **Classes**:
    -   `BaseDataset`: Base class for datasets [dataset.py].
    -   `ConversationSequenceDataset`: Dataset for conversation-level sequences [dataset.py].
    -   `AuthorConversationSequenceDataset`: Dataset for author-level sequences [dataset.py].
-   **Functionality**:
    -   Implements caching to speed up data loading [dataset.py].
    -   Supports memory-mapped loading for large datasets [dataset.py].
    -   Performs batch processing and vectorized operations [dataset.py].
    -   Calculates normalization parameters (mean and standard deviation) [dataset.py].
    -   Provides data loaders for training, validation, and testing [dataset.py].
-   **Usage Example**:

```python
from datahandler import ConversationSequenceDataset, AuthorConversationSequenceDataset, create_dataloaders

# Create a conversation dataset
conv_dataset = ConversationSequenceDataset('path/to/analyzed/conversations', normalize=True)

# Create an author dataset
author_dataset = AuthorConversationSequenceDataset('path/to/analyzed/conversations', normalize=True)

# Create data loaders
conv_train, conv_val, conv_test = create_dataloaders(conv_dataset, batch_size=32)
author_train, author_val, author_test = create_dataloaders(author_dataset, batch_size=32)
```

## Installation

To install **DataHandler**, clone the repository and install the required dependencies:

```bash
git clone <repository_url>
cd datahandler
pip install -r requirements.txt
```

## Usage

The `__init__.py` file provides a convenient way to import the key components of the package:

```python
from datahandler import ConversationSequenceDataset, AuthorConversationSequenceDataset, create_dataloaders, ConversationAnalyzer, ConversationParser, BaseDataset
```

Refer to the individual component documentation above for detailed usage examples.

