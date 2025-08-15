import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from emotionlinmult.preprocess.MOSEI import DB_PROCESSED


def load_distribution_data(json_path):
    """Load the sentiment distribution data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_sentiment_distribution(data, output_dir: Path):
    """Create histograms of sentiment values for train/valid/test splits."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle('Sentiment Distribution', fontsize=16, y=1.05)

    # Define colors for each split
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot each split
    for idx, (split, color) in enumerate(zip(['train', 'valid', 'test'], colors)):
        sentiments = data[split]['sentiment']
        total_samples = data[split]['total']

        # Create histogram
        n, bins, patches = axes[idx].hist(
            sentiments,
            bins=20,
            range=(-3, 3),  # Assuming sentiment scores are in [-3, 3]
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        # Add count labels on top of bars
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width() / 2
            y = patches[i].get_height()
            if y > 0:  # Only add label if count > 0
                axes[idx].text(
                    x, y, 
                    f'{int(y)}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        # Customize the plot
        axes[idx].set_title(f'{split.upper()} (n={total_samples})', fontsize=12, pad=10)
        axes[idx].set_xlabel('Sentiment Score', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].set_axisbelow(True)
        
        # Use integer y-axis ticks
        axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add vertical line at 0
        axes[idx].axvline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / 'sentiment_distribution.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved sentiment distribution plot to {output_path}")


def plot_sentiment_classes_7(data, output_dir: Path):
    """Create bar plots for 7 sentiment classes (-3 to 3) for train/valid/test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle('7-Class Sentiment Distribution', fontsize=16, y=1.05)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    classes = list(range(-3, 4))  # -3 to 3
    
    for idx, (split, color) in enumerate(zip(['train', 'valid', 'test'], colors)):
        # Round sentiment values to nearest integer and count occurrences
        sentiments = np.round(data[split]['sentiment']).astype(int)
        class_counts = defaultdict(int)
        
        for s in sentiments:
            # Ensure the rounded value is within -3 to 3
            s = max(min(int(round(s)), 3), -3)
            class_counts[s] += 1
        
        # Get counts in order from -3 to 3
        counts = [class_counts[c] for c in classes]
        total_samples = data[split]['total']
        
        # Create bar plot
        bars = axes[idx].bar(
            [str(c) for c in classes],
            counts,
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Customize the plot
        axes[idx].set_title(f'{split.upper()} (n={total_samples})', fontsize=12, pad=10)
        axes[idx].set_xlabel('Sentiment Class', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[idx].set_axisbelow(True)
        axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    output_path = output_dir / 'sentiment_classes_7.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved 7-class sentiment distribution plot to {output_path}")


def plot_sentiment_classes_3(data, output_dir: Path):
    """Create bar plots for 3 sentiment classes (negative, neutral, positive) for train/valid/test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('3-Class Sentiment Distribution', fontsize=16, y=1.05)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Negative', 'Neutral', 'Positive']
    
    for idx, (split, color) in enumerate(zip(['train', 'valid', 'test'], colors)):
        sentiments = np.array(data[split]['sentiment'])
        total_samples = data[split]['total']
        
        # Count negative, neutral, and positive
        negative = np.sum(sentiments < 0)
        neutral = np.sum(sentiments == 0)
        positive = np.sum(sentiments > 0)
        counts = [negative, neutral, positive]
        
        # Create bar plot
        bars = axes[idx].bar(
            class_names,
            counts,
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add count and percentage labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_samples) * 100
            axes[idx].text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Customize the plot
        axes[idx].set_title(f'{split.upper()} (n={total_samples})', fontsize=12, pad=10)
        axes[idx].set_xlabel('Sentiment Class', fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
        axes[idx].set_axisbelow(True)
        axes[idx].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    output_path = output_dir / 'sentiment_classes_3.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved 3-class sentiment distribution plot to {output_path}")


def main():
    # Path to the JSON file
    json_path = DB_PROCESSED / 'sentiment_distribution_avt.json'
    
    # Output directory for plots
    output_dir = DB_PROCESSED / 'plots_avt'
    
    # Load the data
    data = load_distribution_data(json_path)
    
    # Create and save the plots
    plot_sentiment_distribution(data, output_dir)  # Histogram plot
    plot_sentiment_classes_7(data, output_dir)     # 7-class bar plot (-3 to 3)
    plot_sentiment_classes_3(data, output_dir)     # 3-class bar plot (negative/neutral/positive)


if __name__ == "__main__":
    main()