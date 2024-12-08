# PredatorDetector

## Overview

**PredatorDetector** is a comprehensive toolkit designed to detect sexual predators in online conversations. This repository provides a suite of tools for analyzing, modeling, and understanding the characteristics of predatory behavior in textual data. The toolkit leverages advanced natural language processing, machine learning, and statistical techniques to identify potential threats and provide insights into the dynamics of online grooming.

## Motivation

The motivation behind PredatorDetector is to address the growing concern of online sexual predation, particularly targeting minors. Traditional methods of detection often rely on manual analysis or keyword-based filters, which can be time-consuming, inefficient, and easily circumvented by sophisticated predators. This project aims to develop a more robust, automated, and data-driven approach to identify and prevent online predatory behavior, ultimately contributing to a safer online environment.

## Purpose

The primary purpose of PredatorDetector is to provide researchers, developers, and online safety professionals with a powerful set of tools for:

1. **Detecting sexual predators in online conversations:** The core functionality is to accurately identify conversations and users that exhibit predatory behavior.
2. **Analyzing and understanding predatory behavior:** The toolkit provides methods for extracting relevant features, visualizing patterns, and interpreting model decisions, offering insights into the linguistic and behavioral characteristics of predators.
3. **Developing and evaluating detection models:** PredatorDetector offers a flexible framework for training, testing, and comparing different machine-learning models for predator detection.
4. **Contributing to online safety research:** The project aims to advance the field of online safety by providing a comprehensive and open-source resource for studying and combating online sexual predation.

## Methodology

PredatorDetector employs a multi-faceted methodology that combines several key approaches:

1. **Data Handling and Preprocessing:** The `DataHandler` package is used to load, parse, and manage conversation datasets, particularly in XML format. It provides functionalities for extracting features, generating statistics, and preparing datasets for machine learning tasks.
2. **Feature Extraction:** The `Extractors` package is employed to extract a wide range of linguistic and behavioral features from text data. It leverages transformer-based models to analyze sentiment, emotions, toxicity, intent, and psycholinguistic dimensions.
3. **Model Training and Evaluation:** The `Trainer` package provides a framework for training and evaluating machine learning models for sequence and profile classification. It supports custom model architectures, efficient data handling, and various evaluation metrics.
4. **Visualization and Interpretability:** The `Visualization` package offers tools for analyzing dataset features, sequential patterns, and model interpretability. It implements multiple feature importance methods, including feature ablation, permutation importance, gradient saliency, SHAP, integrated gradients, and attention attribution.

## Key Components

The PredatorDetector repository comprises four main packages:

### 1. DataHandler

This package streamlines the process of loading, parsing, analyzing, and managing conversation datasets . It includes:

* **Conversation Parser:** Loads and parses XML conversation files into a HuggingFace `datasets.Dataset` .
* **Conversation Analyzer:** Analyzes conversation datasets to extract features and generate statistics .
* **Dataset Classes and Data Loaders:** Provides classes for handling datasets with features like caching, memory-mapped loading, batch processing, and normalization .

### 2. Extractors

This package provides a sophisticated text analysis toolkit for extracting linguistic and behavioral features . It includes:

* **Specialized Extractors:**
 * **Word2AffectExtractor:** Analyzes psycholinguistic dimensions like Valence, Arousal, Dominance, Age of Acquisition, and Concreteness .
 * **SentimentExtractor:** Implements sentiment analysis using a RoBERTa model trained on social media content .
 * **EmotionExtractor:** Uses a RoBERTa model for multi-label emotion classification, trained on the GoEmotions dataset .
 * **ToxicityExtractor:** Implements toxic comment classification using the Detoxify library, supporting multiple model variants .
 * **IntentExtractor:** Implements zero-shot intent classification using BART-large-MNLI, trained on the MultiNLI dataset .

### 3. Trainer

This package provides a framework for training and evaluating machine learning models for sequence and profile classification . It includes:

* **Sequence Classification:** Train models to classify sequences of data, such as conversations or time series .
* **Profile Classification:** Aggregate sequence-level predictions to classify profiles, such as authors or users .
* **Customizable Models:** Define and train custom models using a flexible architecture .
* **Model Architecture:**
 * **SequenceClassifier:** A model for classifying sequences of data, using a Transformer encoder and Conv1d layers .
 * **ProfileClassifier:** A model for classifying profiles by aggregating sequence-level predictions .

### 4. Visualization

This package provides tools for analyzing dataset features, sequential patterns, and model interpretability . It includes:

* **Dataset Analysis:** Tools for analyzing dataset features and their distributions .
* **Feature Importance Analysis:** Implements multiple interpretability methods to determine the most relevant features for detecting sexual predators . Methods include Feature Ablation, Permutation Importance, Gradient Saliency, SHAP, Integrated Gradients, and more.

## Conclusion

PredatorDetector offers a comprehensive and powerful toolkit for detecting and analyzing online sexual predation. By combining advanced natural language processing, machine learning, and statistical techniques, it provides a robust and data-driven approach to enhancing online safety. The project's modular design, extensive feature set, and focus on interpretability make it a valuable resource for researchers, developers, and online safety professionals working to combat this critical issue.
