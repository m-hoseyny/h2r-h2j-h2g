import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define the reasons
reasons = [
    'up-to-date information needed',
    'technical accuracy required',
    'risk of misinformation',
    'numerical data extraction',
    'specific product term',
    'ambiguous subject reference',
    'quantitative data needed',
    'Specific and niche topic',
    'Query unrelated to provided criteria',
    'No relevant passage context'
]

# Load the datasets
dataset1 = 'human_judgements/HumanJudgement1.csv'
dataset2 = 'human_judgements/HumanJudgement2.csv'

try:
    df1 = pd.read_csv(dataset1)
    print(f"Successfully loaded {dataset1}")
except FileNotFoundError:
    print(f"File {dataset1} not found. Please check the path.")
    df1 = None

try:
    df2 = pd.read_csv(dataset2)
    print(f"Successfully loaded {dataset2}")
except FileNotFoundError:
    print(f"File {dataset2} not found. Please check the path.")
    df2 = None

# Function to calculate accuracy metrics for each reason
def calculate_accuracy_metrics(df):
    """
    Calculate accuracy metrics for human annotations compared to the main data.
    
    Args:
        df: DataFrame containing the main data and annotations
    
    Returns:
        Dictionary with accuracy metrics for each reason
    """
    # Replace any non-numeric values
    df = df.replace({'FALSE': 0, 'TRUE': 1, np.nan: 0})
    
    # Initialize results dictionary
    results = {
        'accuracy': {},
        'precision': {},
        'recall': {},
        'f1': {},
        'confusion_matrices': {}
    }
    
    # Extract main rows and annotation rows
    main_rows = df[df['query'] != 'ANNOTATION:']
    annotation_rows = df[df['query'] == 'ANNOTATION:']
    
    # Ensure we have the same number of main rows and annotation rows
    if len(main_rows) != len(annotation_rows):
        print(f"Warning: Number of main rows ({len(main_rows)}) does not match number of annotation rows ({len(annotation_rows)})")
    
    # Calculate metrics for each reason
    for reason in reasons:
        # Extract true values (main data) and predicted values (annotations)
        y_true = []
        y_pred = []
        
        for i in range(0, len(df), 2):
            if i+1 < len(df):
                if df.iloc[i]['query'] != 'ANNOTATION:' and df.iloc[i+1]['query'] == 'ANNOTATION:':
                    y_true.append(int(df.iloc[i][reason]))
                    y_pred.append(int(df.iloc[i+1][reason]))
        
        # Calculate metrics
        if y_true and y_pred:
            results['accuracy'][reason] = accuracy_score(y_true, y_pred)
            results['precision'][reason] = precision_score(y_true, y_pred, zero_division=0)
            results['recall'][reason] = recall_score(y_true, y_pred, zero_division=0)
            results['f1'][reason] = f1_score(y_true, y_pred, zero_division=0)
            results['confusion_matrices'][reason] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return results

# Function to visualize the results
def visualize_results(results):
    """
    Visualize accuracy metrics for each reason.
    
    Args:
        results: Dictionary with accuracy metrics for each reason
    """
    # Create a DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1']
    })
    
    # Check if the DataFrame is empty
    if metrics_df.empty:
        print("No data available for visualization. Skipping plots.")
        return metrics_df
    
    # Plot the metrics
    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Human Annotation Accuracy Metrics by Reason')
    plt.xlabel('Reason')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('human_judgements/human_annotation_accuracy.png')
    plt.close()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, reason in enumerate(reasons):
        if reason in results['confusion_matrices']:
            cm = results['confusion_matrices'][reason]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{reason}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
            axes[i].set_xticklabels(['0', '1'])
            axes[i].set_yticklabels(['0', '1'])
        else:
            # If no data for this reason, display a message
            axes[i].text(0.5, 0.5, 'No data available', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[i].transAxes)
            axes[i].set_title(f'{reason}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('human_judgements/human_annotation_confusion_matrices.png')
    plt.close()
    
    return metrics_df

# Main execution
if __name__ == "__main__":
    # Process the first dataset if available
    if df1 is not None:
        print("\nAnalyzing dataset 1:")
        results1 = calculate_accuracy_metrics(df1)
        metrics_df1 = visualize_results(results1)
        
        # Print overall accuracy
        if results1['accuracy']:
            overall_accuracy1 = np.mean(list(results1['accuracy'].values()))
            overall_precision1 = np.mean(list(results1['precision'].values()))
            print(f"\nOverall accuracy for dataset 1: {overall_accuracy1:.4f}")
            print(f"Overall precision for dataset 1: {overall_precision1:.4f}")
            
            # Print accuracy for each reason
            print("\nAccuracy by reason for dataset 1:")
            for reason, acc in results1['accuracy'].items():
                print(f"{reason}: {acc:.4f}")
            
            # Save metrics to CSV
            metrics_df1.to_csv('human_judgements/human_annotation_metrics_dataset1.csv')
        else:
            print("No valid metrics could be calculated for dataset 1")
    
    # Process the second dataset if available
    if df2 is not None:
        print("\nAnalyzing dataset 2:")
        results2 = calculate_accuracy_metrics(df2)
        metrics_df2 = visualize_results(results2)
        
        # Print overall accuracy
        if results2['accuracy']:
            overall_accuracy2 = np.mean(list(results2['accuracy'].values()))
            overall_precision2 = np.mean(list(results2['precision'].values()))
            print(f"\nOverall accuracy for dataset 2: {overall_accuracy2:.4f}")
            print(f"Overall precision for dataset 2: {overall_precision2:.4f}")
            
            # Print accuracy for each reason
            print("\nAccuracy by reason for dataset 2:")
            for reason, acc in results2['accuracy'].items():
                print(f"{reason}: {acc:.4f}")
            
            # Save metrics to CSV
            metrics_df2.to_csv('human_judgements/human_annotation_metrics_dataset2.csv')
        else:
            print("No valid metrics could be calculated for dataset 2")
    
    # If both datasets are available, calculate combined metrics
    if df1 is not None and df2 is not None:
        print("\nAnalyzing combined datasets:")
        
        # Combine the datasets
        df_combined = pd.concat([df1, df2], ignore_index=True)
        
        # Calculate metrics for the combined dataset
        results_combined = calculate_accuracy_metrics(df_combined)
        metrics_df_combined = visualize_results(results_combined)
        
        if results_combined['accuracy']:
            # Calculate overall metrics across all reasons
            overall_accuracy_combined = np.mean(list(results_combined['accuracy'].values()))
            overall_precision_combined = np.mean(list(results_combined['precision'].values()))
            
            print(f"\nOverall accuracy for combined datasets: {overall_accuracy_combined:.4f}")
            print(f"Overall precision for combined datasets: {overall_precision_combined:.4f}")
            
            # Print accuracy for each reason
            print("\nAccuracy by reason for combined datasets:")
            for reason, acc in results_combined['accuracy'].items():
                print(f"{reason}: {acc:.4f}")
            
            # Save metrics to CSV
            metrics_df_combined.to_csv('human_judgements/human_annotation_metrics_combined.csv')
            
            # Calculate total metrics (all reasons combined)
            # Extract all true and predicted values across all reasons
            all_true = []
            all_pred = []
            
            for i in range(0, len(df_combined), 2):
                if i+1 < len(df_combined):
                    if df_combined.iloc[i]['query'] != 'ANNOTATION:' and df_combined.iloc[i+1]['query'] == 'ANNOTATION:':
                        for reason in reasons:
                            try:
                                all_true.append(int(df_combined.iloc[i][reason]))
                                all_pred.append(int(df_combined.iloc[i+1][reason]))
                            except (ValueError, TypeError):
                                pass
            
            if all_true and all_pred:
                # Calculate total metrics across all data points
                total_accuracy = accuracy_score(all_true, all_pred)
                total_precision = precision_score(all_true, all_pred, zero_division=0)
                total_recall = recall_score(all_true, all_pred, zero_division=0)
                total_f1 = f1_score(all_true, all_pred, zero_division=0)
                
                print("\n=== TOTAL METRICS (ALL REASONS COMBINED) ===")
                print(f"Total Accuracy: {total_accuracy:.4f}")
                print(f"Total Precision: {total_precision:.4f}")
                print(f"Total Recall: {total_recall:.4f}")
                print(f"Total F1 Score: {total_f1:.4f}")
                
                # Save total metrics
                total_metrics = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Value': [total_accuracy, total_precision, total_recall, total_f1]
                })
                total_metrics.to_csv('human_judgements/total_metrics.csv', index=False)
                
                # Create confusion matrix for total data
                total_cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
                
                # Plot total confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Total Confusion Matrix (All Reasons Combined)')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.xticks([0.5, 1.5], ['0', '1'])
                plt.yticks([0.5, 1.5], ['0', '1'])
                plt.tight_layout()
                plt.savefig('human_judgements/total_confusion_matrix.png')
                plt.close()
        else:
            print("No valid metrics could be calculated for combined datasets")