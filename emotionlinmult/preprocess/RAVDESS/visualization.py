import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from emotionlinmult.preprocess import UNIFIED_EMOTION_CLASSES, UNIFIED_INTENSITY_CLASSES
from emotionlinmult.preprocess.RAVDESS import DB_PROCESSED


def load_distribution_data(json_path):
    """Load the class distribution data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_distribution(ax, data, title, colors=None):
    """Plot a single distribution bar chart with count labels."""
    categories = list(data.keys())
    counts = list(data.values())
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(categories)))
    
    bars = ax.bar(categories, counts, color=colors)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def create_distribution_plots(data, output_dir):
    """Create distribution plots for all tasks and splits."""
    # Define the tasks and their display names
    tasks = [
        ('dataset_emotion_class', 'Dataset Emotion Class'),
        ('unified_emotion_class', 'Unified Emotion Class'),
        ('dataset_intensity', 'Dataset Intensity'),
        ('unified_intensity', 'Unified Intensity')
    ]

    splits = ['train', 'valid', 'test']

    # Define all possible categories for unified classes
    unified_emotion_categories = [class_name.unified_name for class_name in UNIFIED_EMOTION_CLASSES]
    unified_intensity_categories = [intensity.unified_name for intensity in UNIFIED_INTENSITY_CLASSES]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure for each task
    for task_key, task_name in tasks:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(f'Distribution of {task_name}', fontsize=16)

        # Get all possible categories based on task type
        if 'unified_emotion' in task_key:
            all_categories = unified_emotion_categories
        elif 'unified_intensity' in task_key:
            all_categories = unified_intensity_categories
        else:
            # For dataset-specific categories, collect from data
            all_categories = set()
            for split in splits:
                if task_key in data[split]:
                    all_categories.update(data[split][task_key].keys())
            all_categories = sorted(all_categories)

        # Create a color map for consistent colors across plots
        color_map = {cat: plt.cm.viridis(i/len(all_categories)) 
                    for i, cat in enumerate(all_categories)}

        # Plot each split
        for i, split in enumerate(splits):
            if task_key in data[split]:
                # Prepare data for plotting
                plot_data = data[split].get(task_key, {})
                
                # Create ordered lists of categories and counts, including zero counts
                categories = []
                counts = []
                colors = []
                
                # For unified categories, ensure all are included even if not in data
                if 'unified_emotion' in task_key or 'unified_intensity' in task_key:
                    for cat in all_categories:
                        categories.append(cat)
                        count = plot_data.get(cat, 0)  # Get count or 0 if category doesn't exist
                        counts.append(count)
                        colors.append(color_map[cat])
                else:
                    # For dataset-specific categories, only include those present in data
                    for cat in all_categories:
                        if cat in plot_data:
                            categories.append(cat)
                            counts.append(plot_data[cat])
                            colors.append(color_map[cat])
                
                # Convert to dictionary for the plot function
                plot_dict = dict(zip(categories, counts))
                
                # Plot
                plot_distribution(axes[i], plot_dict, f'{split.title()} Split', colors)
                
                # Add total count to the title
                axes[i].set_title(f"{split.title()} (Total: {data[split]['subset_total']})")
            else:
                axes[i].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        output_path = os.path.join(output_dir, f'{task_key}_distribution.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {task_name} distribution plot to {output_path}")


def main():
    # Path to the class distribution JSON file
    json_path = DB_PROCESSED / "class_distribution.json"
    
    # Output directory for plots
    output_dir = DB_PROCESSED / "visualizations" / "class_distributions"
    
    # Load the data
    data = load_distribution_data(json_path)
    
    # Create and save the plots
    create_distribution_plots(data, output_dir)


if __name__ == "__main__":
    main()