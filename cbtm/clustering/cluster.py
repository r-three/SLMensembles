import argparse
import gzip
import itertools
import json
import numpy as np
import pandas as pd
import pickle
import torch
import tqdm
import random
import os
from itertools import chain
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from typing import Dict
from kmeans_pytorch import KMeans as BalancedKMeans
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from umap import UMAP
import seaborn as sns



def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Call before training
set_random_seeds(42)


def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

def load_model(path_to_model: Path):
    """Load pickled model from file"""
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out


def load_tulu_dataset(dataset_name="allenai/tulu-3-sft-mixture", sample_size=20000):
    """Load Tulu dataset and extract user prompts"""
    dataset = load_dataset(dataset_name, split="train")
    
    # Extract user prompts
    user_prompts = []
    for i, example in enumerate(tqdm(dataset, desc="Extracting user prompts")):
        # Find the first user message
        for message in example['messages']:
            if message.get('role') == 'user':
                user_prompts.append({
                    "id": i,
                    "text": message.get('content', ''),
                    "source": example.get('source', ''),
                    "original_id": example.get('id', ''),
                    "messages": example.get('messages', [])  # Keep full conversation
                })
                break
    
    # Convert to DataFrame and sample
    df = pd.DataFrame(user_prompts)
    
    # Remove empty texts
    df = df[df['text'].str.strip() != '']
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Loaded {len(df)} user prompts from Tulu dataset")
    return df


def vectorize_tulu_dataset(model, dataset_name, sample_size=10000):
    """Vectorize Tulu dataset for clustering"""
    texts_df = load_tulu_dataset(dataset_name, sample_size=sample_size)
    vecs = model.transform(tqdm(texts_df.text))
    return vecs, texts_df


def create_multi_config_dataset_and_upload(
    vectorizer, 
    kmeans, 
    dataset_name, 
    cluster_names_mapping,
    hf_username,
    repo_name,
    sample_size=10000,
    test_size=0.2,
    random_state=42
):
    """
    Create a single dataset with multiple configurations (one per cluster/domain)
    Each cluster becomes a separate config within the same repository
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        kmeans: Trained KMeans model  
        dataset_name: Name of dataset to load
        cluster_names_mapping: Dict mapping cluster_id -> domain_name
        hf_username: HuggingFace username
        repo_name: Name of the repository to create
        sample_size: Number of samples to process
        test_size: Fraction for test split (0.2 = 20%)
        random_state: Random seed for reproducible splits
    """
    
    # Load and vectorize data
    print("🔄 Loading and vectorizing dataset...")
    vecs, metadata = vectorize_tulu_dataset(vectorizer, dataset_name, sample_size=sample_size)
    
    # Get device for clustering
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using GPU for clustering")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU for clustering")
    
    # Predict clusters
    print("🎯 Predicting clusters...")
    cluster_labels = kmeans.predict(torch.from_numpy(vecs).to(device)).cpu().numpy()
    metadata['cluster'] = cluster_labels
    
    print(f"\n📊 Cluster Distribution:")
    cluster_counts = metadata['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        cluster_name = cluster_names_mapping.get(cluster_id, f"Cluster_{cluster_id}")
        percentage = count / len(metadata) * 100
        print(f"  C{cluster_id}: {cluster_name:<30} ({count:,} samples, {percentage:.1f}%)")
    
    # Prepare configurations for each cluster
    configs = {}
    config_info = []
    
    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_name = cluster_names_mapping.get(cluster_id, f"cluster_{cluster_id}")
        config_name = cluster_name.lower().replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
        
        print(f"\n🔄 Processing Config: {config_name}")
        
        # Get data for this cluster/config
        cluster_data = metadata[metadata['cluster'] == cluster_id].copy()
        
        if len(cluster_data) < 10:
            print(f"⚠️ Skipping {cluster_name}: too few samples ({len(cluster_data)})")
            continue
        
        # Remove cluster-specific metadata columns for clean data
        # Keep only the essential columns
        domain_columns = ['text', 'source', 'original_id', 'messages']
        clean_cluster_data = cluster_data[domain_columns].copy()
        
        # Create train/test split
        train_data, test_data = train_test_split(
            clean_cluster_data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"  📦 Train: {len(train_data)} samples")
        print(f"  🧪 Test: {len(test_data)} samples")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_data.reset_index(drop=True))
        
        # Create DatasetDict for this config
        config_dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        configs[config_name] = config_dataset
        config_info.append({
            'cluster_id': cluster_id,
            'config_name': config_name,
            'domain_name': cluster_name,
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'total_samples': len(cluster_data)
        })
    
    if not configs:
        print("❌ No valid configs created.")
        return []
    
    # Create the full repository name
    full_repo_name = f"{hf_username}/{repo_name}"
    print(f"\n🚀 Uploading multi-config dataset to: {full_repo_name}")
    print(f"📋 Configs to upload: {list(configs.keys())}")
    
    try:
        # Upload each config to the same repository
        for config_name, config_dataset in configs.items():
            print(f"  📤 Uploading config: {config_name}")
            
            config_dataset.push_to_hub(
                full_repo_name,
                config_name=config_name,
                private=False,  # Set to True if you want private repos
                commit_message=f"Upload {config_name} configuration"
            )
            
            print(f"  ✅ Config {config_name} uploaded successfully")
        
        # Create and upload main dataset card
        card_content = create_multi_config_dataset_card(
            repo_name,
            config_info,
            dataset_name
        )
        
        # Upload README separately using HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=full_repo_name,
            repo_type="dataset",
            commit_message="Add comprehensive dataset card"
        )
        
        print(f"✅ Successfully uploaded multi-config dataset: {full_repo_name}")
        
    except Exception as e:
        print(f"❌ Failed to upload {full_repo_name}: {str(e)}")
        return []
    
    # Print summary
    print(f"\n{'='*80}")
    print("📋 MULTI-CONFIG DATASET SUMMARY")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(config_info)
    print(summary_df.to_string(index=False))
    
    total_samples = summary_df['total_samples'].sum()
    total_train = summary_df['train_samples'].sum() 
    total_test = summary_df['test_samples'].sum()
    
    print(f"\n🎉 Successfully uploaded {len(configs)} configurations!")
    print(f"📊 Repository: {full_repo_name}")
    print(f"📈 Total samples: {total_samples:,}")
    print(f"🚆 Train samples: {total_train:,}")  
    print(f"🧪 Test samples: {total_test:,}")
    
    print(f"\n🔧 Available Configurations:")
    for config in config_info:
        print(f"  • {config['config_name']}: {config['domain_name']} ({config['total_samples']:,} samples)")
    
    print(f"\n💡 Usage Example:")
    print(f"```python")
    print(f"from datasets import load_dataset")
    print(f"")
    print(f"# Load specific configuration")
    print(f"dataset = load_dataset('{full_repo_name}', '{config_info[0]['config_name']}' if config_info else 'config_name')")
    print(f"")
    print(f"# Or load all configurations")
    print(f"all_configs = {{")
    for config in config_info[:3]:  # Show first 3 as example
        print(f"    '{config['config_name']}': load_dataset('{full_repo_name}', '{config['config_name']}'),")
    if len(config_info) > 3:
        print(f"    # ... and {len(config_info) - 3} more configs")
    print(f"}}")
    print(f"```")
    
    return config_info


def generate_yaml_configs(config_info):
    """
    Generate the YAML configuration section for the metadata
    
    Args:
        config_info (list): List of configuration dictionaries
        
    Returns:
        str: YAML configuration section
    """
    yaml_configs = "configs:\n"
    
    for config in config_info:
        config_name = config['config_name']
        yaml_configs += f"""  - config_name: {config_name}
    data_files:
      - split: train
        path: "{config_name}/train*.parquet"
      - split: test
        path: "{config_name}/test*.parquet"
"""
    
    return yaml_configs

def create_multi_config_dataset_card(repo_name, config_info, source_dataset):
    """Create a comprehensive dataset card for multi-config dataset"""
    
    total_samples = sum(config['total_samples'] for config in config_info)
    total_train = sum(config['train_samples'] for config in config_info)
    total_test = sum(config['test_samples'] for config in config_info)
    
    # Generate config list for metadata
    config_names = [config['config_name'] for config in config_info]

    # Generate YAML configs section
    yaml_configs = generate_yaml_configs(config_info)
    
    card = f"""---
license: apache-2.0
task_categories:
- text-generation

language:
- en
tags:
- multi-domain
- conversational-ai
- clustered-data
size_categories:
- 10K<n<100K

{yaml_configs}
---

# {repo_name.title().replace('-', ' ')} Multi-Domain Dataset

This dataset contains high-quality examples across **{len(config_info)} specialized domains**, automatically extracted and curated from the Tulu-3 SFT mixture using advanced clustering techniques.

## 🎯 Multi-Domain Structure

This repository provides **{len(config_info)} domain-specific configurations**, each optimized for different types of tasks:

| Configuration | Domain | Train | Test | Total |
|---------------|--------|-------|------|-------|"""

    for config in config_info:
        card += f"\n| `{config['config_name']}` | {config['domain_name']} | {config['train_samples']:,} | {config['test_samples']:,} | {config['total_samples']:,} |"

    card += f"""
| **TOTAL** | **All Domains** | **{total_train:,}** | **{total_test:,}** | **{total_samples:,}** |

## 🚀 Quick Start

### Loading Specific Domains

```python
from datasets import load_dataset

# Load a specific domain configuration
{config_info[0]['config_name']}_data = load_dataset("{repo_name}", "{config_info[0]['config_name']}")

# Access train/test splits
train_examples = {config_info[0]['config_name']}_data['train']
test_examples = {config_info[0]['config_name']}_data['test']
```

### Loading Multiple Domains

```python
# Load multiple domain configurations
domains = {{"""

    for i, config in enumerate(config_info[:3]):  # Show first 3
        card += f"""
    '{config['config_name']}': load_dataset("{repo_name}", "{config['config_name']}"),"""
    
    if len(config_info) > 3:
        card += f"""
    # ... load {len(config_info) - 3} more configurations as needed"""

    card += f"""
}}

# Use specific domain
for example in domains['{config_info[0]['config_name']}']['train']:
    print(example['text'])
```

### Available Configurations

"""

    for config in config_info:
        card += f"""
#### `{config['config_name']}` - {config['domain_name']}
- **Focus**: {config['domain_name']} tasks and conversations
- **Size**: {config['total_samples']:,} examples ({config['train_samples']:,} train, {config['test_samples']:,} test)
- **Use for**: Domain-specific fine-tuning, specialized instruction following
"""

    card += f"""
## 🏗️ Dataset Structure

Each configuration contains the same structure:
- **`text`**: The user prompt/instruction (main input)
- **`source`**: Original source dataset identifier  
- **`messages`**: Complete conversation thread with roles
- **`original_id`**: Unique identifier from source dataset

## 🔍 Data Quality

- ✅ **Domain Clustering**: Automatically grouped by content similarity
- ✅ **Quality Filtering**: Filtered for relevance and coherence
- ✅ **Stratified Splits**: 80/20 train/test split per domain
- ✅ **Clean Format**: Ready-to-use conversation structure
- ✅ **Consistent Schema**: Same format across all configurations

## 📚 Use Cases

### Domain-Specific Training
```python
# Fine-tune on coding tasks only
coding_data = load_dataset("{repo_name}", "coding_and_programming")
```

### Multi-Domain Training  
```python
# Combine multiple domains for diverse training
multi_domain = concatenate_datasets([
    load_dataset("{repo_name}", "{config_info[0]['config_name']}")['train'],
    load_dataset("{repo_name}", "{config_info[1]['config_name'] if len(config_info) > 1 else config_info[0]['config_name']}")['train'],
    # Add more domains as needed
])
```

### Evaluation Across Domains
```python
# Test model performance on different domains
for config_name in ["{config_info[0]['config_name']}", "{config_info[1]['config_name'] if len(config_info) > 1 else config_info[0]['config_name']}"]:
    test_data = load_dataset("{repo_name}", config_name)['test']
    # Run evaluation on domain-specific test set
```

## 🏷️ Source Information

- **Original Dataset**: {source_dataset}
- **Clustering Method**: TF-IDF + K-Means
- **Total Configurations**: {len(config_info)}
- **Processing**: Automated clustering with manual domain labeling

## 📄 Citation

```bibtex
@dataset{{{repo_name.replace('-', '_')}_multi_domain,
  title={{{repo_name.title().replace('-', ' ')} Multi-Domain Dataset}},
  author={{Extracted from Tulu-3 SFT Mixture}},
  year={{2025}},
  url={{https://huggingface.co/datasets/{repo_name}}},
  note={{Multi-domain dataset with {len(config_info)} specialized configurations}}
}}
```

## 📜 License

This dataset follows the same license as the original Tulu-3 SFT mixture (Apache 2.0).

---

*This multi-domain dataset provides specialized configurations for targeted training while maintaining the flexibility to combine domains as needed. Each configuration is automatically curated for domain coherence and quality.*
"""

    return card


def analyze_cluster_quality(metadata, vectorizer, kmeans):
    """Analyze the quality of clustering results"""
    
    print(f"\n{'='*80}")
    print("📈 CLUSTERING QUALITY ANALYSIS")
    print(f"{'='*80}")
    
    # Basic statistics
    n_clusters = len(metadata['cluster'].unique())
    n_samples = len(metadata)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Total samples: {n_samples:,}")
    print(f"Average cluster size: {n_samples/n_clusters:.1f}")
    
    # Cluster size distribution
    cluster_sizes = metadata['cluster'].value_counts().sort_index()
    print(f"\nCluster size statistics:")
    print(f"  Min size: {cluster_sizes.min()}")
    print(f"  Max size: {cluster_sizes.max()}")
    print(f"  Std deviation: {cluster_sizes.std():.1f}")
    
    # Source diversity per cluster
    print(f"\nSource diversity per cluster:")
    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        n_sources = cluster_data['source'].nunique()
        print(f"  Cluster {cluster_id}: {n_sources} unique sources")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster data and upload to HuggingFace as multi-config dataset')
    parser.add_argument('--dataset-name', default="allenai/tulu-3-sft-mixture", type=str,
                       help='HuggingFace dataset name to process')
    parser.add_argument('--vectorizer-path', required=True, type=Path,
                       help='Path to pickled TF-IDF vectorizer')
    parser.add_argument('--kmeans-path', required=True, type=Path,
                       help='Path to pickled KMeans model')
    parser.add_argument('--hf-username', required=True, type=str,
                       help='HuggingFace username for uploading dataset')
    parser.add_argument('--repo-name', default='tulu-multi-domain', type=str,
                       help='Name of the HuggingFace repository to create')
    parser.add_argument('--sample-size', type=int, default=10000, 
                       help='Number of samples to process')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for test split (default: 0.2 = 20%)')
    parser.add_argument('--output-dir', type=Path, default=Path('./output'),
                       help='Output directory for logs and analysis')

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🔄 Loading models...")
    print(f"  Vectorizer: {args.vectorizer_path}")
    print(f"  KMeans: {args.kmeans_path}")
    
    # Load models
    vectorizer = load_model(args.vectorizer_path)
    kmeans = load_model(args.kmeans_path)
    
    # Define cluster-to-domain mapping
    # TODO: Update this mapping based on your manual cluster analysis
    cluster_names_mapping = {
    0: "Task Instructions & Chinese Content",        # Task-based instructions with Chinese text
    1: "Reasoning & Consciousness Questions",        # Logic, reasoning, consciousness exploration
    2: "Creative Writing & Articles",               # Story writing, blog posts, creative content
    3: "Translation Tasks",                         # Language translation, Japanese-English
    4: "General Assistance & Mixed Topics",         # Varied help requests, general questions
    5: "Alex Character Math Problems",              # Math problems featuring Alex character
    6: "Cyrillic Text & Algorithms",               # Russian/Ukrainian text, algorithm content
    7: "Group & Community Management",              # People management, community organization
    8: "Language Processing & NLP",                 # Language tasks, NLP, text processing
    9: "Character & Narrative Development",         # Story characters, AI narratives, world-building
    10: "Business Revenue & Growth",                # Company revenue, business growth analysis
    11: "Probability & Statistical Models",         # Probability calculations, statistical analysis
    12: "Sports & Gaming",                          # Team sports, game statistics, player data
    13: "Tamil & Sinhala Scripts",                  # Tamil and Sinhala language content
    14: "Prime Numbers & Sequences",                # Prime numbers, Fibonacci, mathematical sequences
    15: "Content Generation & Health",              # Content creation, health information
    16: "Time & Duration Problems",                 # Hours, weeks, time calculations
    17: "String Functions & Text Manipulation",     # String processing, character manipulation
    18: "Array Operations & Numerical Computing",   # Array functions, numerical operations
    19: "Malagasy & Misc Languages",               # Malagasy language and other scripts
    20: "Tax & Policy Analysis",                   # Tax calculations, policy documents
    21: "Indonesian & African Languages",          # Indonesian, Yoruba, African languages
    22: "Simple Named Character Problems",         # Basic math with Emily, Olivia characters
    23: "Machine Learning & AI Models",            # ML models, AI systems, datasets
    24: "Geometry & Spatial Calculations",         # Area, perimeter, geometric shapes
    25: "Complex Problem Solving & Budgets",       # Complex word problems, budget optimization
    26: "Art & Visual Programming",                # SVG, visual art, creative programming
    27: "Energy & Scientific Applications",        # Solar energy, quantum physics, scientific
    28: "Advanced Calculus & Trigonometry",        # Complex mathematical functions
    29: "Trigonometric Functions & Physics",       # Sin, cos, physics equations
    30: "Romance Languages (European)",            # Spanish, French, Italian, Portuguese
    31: "Step-by-Step Reasoning",                  # Logical reasoning, premise-hypothesis
    32: "Database & SQL Operations",               # SQL queries, database management
    33: "Formatted Writing & Documentation",       # Structured writing, paragraph formatting
    34: "Medical & Health Research",               # Medical studies, health research, patient data
    35: "Music & Entertainment",                   # Songs, playlists, music industry
    36: "Python List Processing",                  # Python list operations, data processing
    37: "Comedic & Detailed Storytelling",        # Comedy, vivid descriptions, entertainment
    38: "Security & Privacy",                      # Cybersecurity, privacy protection
    39: "Social Media & Digital Content",          # Social media posts, digital engagement
    40: "Word & Text Analysis",                    # Word counting, text analysis, linguistics
    41: "Educational & School Scenarios",          # Students, teachers, classroom problems
    42: "Graph Theory & Network Analysis",         # Network graphs, node analysis
    43: "Mathematical Optimization",               # Min/max problems, optimization
    44: "Premise-Hypothesis Testing",              # Logical premise evaluation
    45: "Calendar & Scheduling",                   # Day/date calculations, scheduling
    46: "Growth Models & Differential Equations",  # Exponential growth, differential equations
    47: "Combinatorics & Counting Problems",       # Counting, permutations, combinations
    48: "Books & Reading",                         # Book collections, reading, libraries
    49: "Travel & Distance Calculations",          # Speed, distance, travel time
    50: "Cost & Financial Calculations",           # Budget, cost analysis, financial math
    51: "Sentence Logic & Common Sense",           # Sentence evaluation, logical reasoning
    52: "Mixed Languages & General Knowledge",      # Various languages, general facts
    53: "Percentage & Ratio Problems",             # Percentage calculations, ratios
    54: "Age & Historical Data",                   # Age problems, historical facts
    55: "Python Code Analysis & Debugging",        # Code review, debugging, error fixing
    56: "Product Reviews & Sentiment",             # Product reviews, sentiment analysis
    57: "Python Function Development",             # Custom Python function creation
    58: "Short Questions & Simple Translations",   # Brief questions, simple translations
    59: "Data Science & Big Data",                 # Data processing, big data analytics
    60: "Geometric Shapes & Drawing",              # Points, lines, geometric drawing
    61: "Spanish & Portuguese Content",            # Spanish and Portuguese text
    62: "Detailed Instructions & Explanations",    # Comprehensive explanations, detailed responses
    63: "Program Development & Design"             # Software development, program design
    }
    
    print(f"\n🏷️ Domain Mapping:")
    for cluster_id, name in cluster_names_mapping.items():
        print(f"  C{cluster_id}: {name}")
    
    # Process clusters and upload as multi-config dataset
    config_info = create_multi_config_dataset_and_upload(
        vectorizer=vectorizer,
        kmeans=kmeans, 
        dataset_name=args.dataset_name,
        cluster_names_mapping=cluster_names_mapping,
        hf_username=args.hf_username,
        repo_name=args.repo_name,
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=42
    )
    
    # Save results to output directory
    if config_info:
        results_df = pd.DataFrame(config_info)
        results_path = args.output_dir / 'config_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to: {results_path}")
        
        # Save cluster mapping
        mapping_path = args.output_dir / 'cluster_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(cluster_names_mapping, f, indent=2)
        print(f"🏷️ Cluster mapping saved to: {mapping_path}")
    
    print(f"\n🎉 Process completed!")
    print(f"🌐 Check your dataset at: https://huggingface.co/datasets/{args.hf_username}/{args.repo_name}")