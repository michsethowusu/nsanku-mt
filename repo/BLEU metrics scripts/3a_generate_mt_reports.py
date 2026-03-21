"""
generate_mt_reports.py
Generate comprehensive reports for MT evaluation metrics: BLEU (corpus-level), chrF (corpus and sentence-level)
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import re
from collections import defaultdict
from tqdm import tqdm

# ------------------------------------------------------------------
# Configure PNG export settings
# ------------------------------------------------------------------
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800
pio.templates.default = "plotly_white"

# Load language mapping from CSV
def load_language_mapping():
    """Load language mapping from CSV file"""
    mapping_path = os.path.join("language_mapping.csv")
    language_mapping = {}
    
    try:
        if os.path.exists(mapping_path):
            df = pd.read_csv(mapping_path)
            for _, row in df.iterrows():
                language_mapping[row['language_code']] = row['language_name']
            print(f"Loaded language mapping for {len(language_mapping)} languages")
        else:
            print(f"Warning: Language mapping file not found at {mapping_path}")
    except Exception as e:
        print(f"Error loading language mapping: {e}")
    
    return language_mapping

LANGUAGE_MAPPING = load_language_mapping()

def get_language_name(lang_code):
    """Get language name from code"""
    return LANGUAGE_MAPPING.get(lang_code, lang_code)

def get_language_display_name(language_pair):
    """Convert language pair code to display name using SOURCE language"""
    if '-' not in language_pair:
        return get_language_name(language_pair)
    
    source_lang, target_lang = language_pair.split('-', 1)
    source_name = get_language_name(source_lang)
    
    return source_name if source_name != source_lang else language_pair

def get_available_recipes(recipes_dir="/home/owusus/Documents/GitHub/nsanku-mt/recipes"):
    """Get list of available recipe names"""
    recipes = []
    if os.path.exists(recipes_dir):
        for file in os.listdir(recipes_dir):
            if file.endswith(".py") and file != "__init__.py":
                recipes.append(file[:-3])
    return recipes

def extract_recipe_name_from_filename(filename, available_recipes):
    """Extract recipe/model name from filename"""
    name_without_ext = os.path.splitext(filename)[0]
    for recipe in available_recipes:
        if f"_{recipe}" in name_without_ext:
            return recipe
    match = re.search(r'_([^_]+)$', name_without_ext)
    if match:
        return match.group(1)
    return "unknown_recipe"

def combine_all_datasets(input_dir="/home/owusus/Documents/GitHub/nsanku-mt/output_combined"):
    """
    Combine all CSV files from all directories into one main dataset
    """
    all_data = []
    recipes = get_available_recipes()
    
    print("Combining all datasets...")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            
            src, tgt = folder.split("-", 1)
            recipe = extract_recipe_name_from_filename(file, recipes)
            
            try:
                df = pd.read_csv(os.path.join(root, file))
                
                # Check if we have the required metric columns (new column names)
                has_metrics = any(col in df.columns for col in ['corpus_bleu', 'corpus_chrf', 'chrf_sentence'])
                if not has_metrics:
                    continue
                
                # Add metadata columns
                df['language_pair'] = f"{src}-{tgt}"
                df['source_lang'] = src
                df['target_lang'] = tgt
                df['model'] = recipe
                df['file_path'] = os.path.join(root, file)
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    
    if not all_data:
        print("No data files found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Language pairs: {combined_df['language_pair'].nunique()}")
    print(f"Models: {combined_df['model'].nunique()}")
    
    return combined_df

def create_horizontal_bar_chart(data_dict, title, x_label, filename, output_dir):
    """Create horizontal bar chart for model/language comparison"""
    if not data_dict:
        return
    
    # Sort by score (highest first)
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=x_label)
        ),
        text=[f'{v:.2f}' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='',
        height=max(400, len(labels) * 30),
        showlegend=False,
        font=dict(size=12)
    )
    
    # Save as HTML and PNG
    html_path = os.path.join(output_dir, f'{filename}.html')
    png_path = os.path.join(output_dir, f'{filename}.png')
    
    fig.write_html(html_path)
    fig.write_image(png_path)

def create_metric_comparison_chart(model_metrics, title, filename, output_dir):
    """
    Create grouped bar chart comparing different metrics for each model
    model_metrics: dict of {model_name: {metric_name: score}}
    """
    if not model_metrics:
        return
    
    models = list(model_metrics.keys())
    metrics = ['Corpus BLEU', 'Corpus chrF', 'Mean Sentence chrF']
    
    fig = go.Figure()
    
    colors = {
        'Corpus BLEU': '#1f77b4',
        'Corpus chrF': '#ff7f0e',
        'Mean Sentence chrF': '#2ca02c'
    }
    
    for metric in metrics:
        values = [model_metrics[model].get(metric, 0) for model in models]
        
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            marker_color=colors.get(metric, '#333333'),
            text=[f'{v:.1f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=600,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save as HTML and PNG
    html_path = os.path.join(output_dir, f'{filename}.html')
    png_path = os.path.join(output_dir, f'{filename}.png')
    
    fig.write_html(html_path)
    fig.write_image(png_path)

def collect_results(input_dir="/home/owusus/Documents/GitHub/nsanku-mt/output_combined"):
    """
    Collect results for all models and language pairs
    Uses stored corpus_bleu and corpus_chrf from the metric calculation step
    """
    # Storage for aggregated results
    corpus_bleu_results = {}
    corpus_chrf_results = {}
    sentence_chrf_results = {}
    
    print("Collecting results from processed data...")
    
    recipes = get_available_recipes()
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            
            folder = os.path.basename(root)
            if "-" not in folder:
                continue
            
            src, tgt = folder.split("-", 1)
            language_pair = f"{src}-{tgt}"
            recipe = extract_recipe_name_from_filename(file, recipes)
            
            try:
                df = pd.read_csv(os.path.join(root, file))
                
                # Check for new metric columns
                has_corpus_bleu = 'corpus_bleu' in df.columns
                has_corpus_chrf = 'corpus_chrf' in df.columns
                has_sentence_chrf = 'chrf_sentence' in df.columns
                
                if not (has_corpus_bleu or has_corpus_chrf):
                    continue
                
                # Get valid rows (non-null sentence chrF indicates valid pairs)
                if has_sentence_chrf:
                    valid_mask = df['chrf_sentence'].notna()
                else:
                    # Fallback: check for valid translations
                    valid_mask = (
                        df['translated'].notna() & 
                        df['ref'].notna() & 
                        (df['translated'].astype(str).str.strip() != '') &
                        (df['ref'].astype(str).str.strip() != '')
                    )
                
                if not valid_mask.any():
                    continue
                
                # Initialize language pair dicts if needed
                if language_pair not in corpus_bleu_results:
                    corpus_bleu_results[language_pair] = {}
                    corpus_chrf_results[language_pair] = {}
                    sentence_chrf_results[language_pair] = {}
                
                # Collect corpus-level metrics (these are the same for all rows in a file)
                # We take the first valid row's value since they're all identical
                if has_corpus_bleu:
                    corpus_bleu_results[language_pair][recipe] = df.loc[valid_mask, 'corpus_bleu'].iloc[0]
                
                if has_corpus_chrf:
                    corpus_chrf_results[language_pair][recipe] = df.loc[valid_mask, 'corpus_chrf'].iloc[0]
                
                # Collect mean sentence-level chrF (average across valid sentences)
                if has_sentence_chrf:
                    sentence_chrf_results[language_pair][recipe] = df.loc[valid_mask, 'chrf_sentence'].mean()
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    # Combine results
    results = {
        'corpus_bleu': corpus_bleu_results,
        'corpus_chrf': corpus_chrf_results,
        'sentence_chrf': sentence_chrf_results
    }
    
    print(f"\n✓ Collected results for:")
    print(f"  - {len(corpus_bleu_results)} language pairs with corpus BLEU")
    print(f"  - {len(corpus_chrf_results)} language pairs with corpus chrF")
    print(f"  - {len(sentence_chrf_results)} language pairs with sentence chrF")
    
    # Note: source_breakdown is not implemented in this version
    source_breakdown = {}
    
    return results, source_breakdown

def generate_language_specific_reports(results, source_breakdown, 
                                       output_dir="/home/owusus/Documents/GitHub/nsanku-mt/reports_combined"):
    """
    Generate individual reports for each language pair with all metrics
    """
    print("Generating language-specific reports...")
    
    # Get all language pairs (from corpus_bleu results)
    language_pairs = results['corpus_bleu'].keys()
    
    for language_pair in language_pairs:
        print(f"Processing {language_pair}...")
        
        lang_output_dir = os.path.join(output_dir, language_pair)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        # Collect all metrics for this language pair
        model_metrics = {}
        
        # Get all models for this language pair
        all_models = set()
        for metric_dict in [results['corpus_bleu'], results['corpus_chrf'], results['sentence_chrf']]:
            if language_pair in metric_dict:
                all_models.update(metric_dict[language_pair].keys())
        
        for model in all_models:
            model_metrics[model] = {
                'Corpus BLEU': results['corpus_bleu'].get(language_pair, {}).get(model, 0),
                'Corpus chrF': results['corpus_chrf'].get(language_pair, {}).get(model, 0),
                'Mean Sentence chrF': results['sentence_chrf'].get(language_pair, {}).get(model, 0)
            }
        
        if not model_metrics:
            print(f"No model results found for {language_pair}")
            continue
        
        # Generate metric comparison chart
        create_metric_comparison_chart(
            model_metrics,
            f'MT Metrics Comparison for {language_pair}',
            'metrics_comparison',
            lang_output_dir
        )
        
        # Generate individual metric charts
        metric_configs = [
            ('Corpus BLEU', 'corpus_bleu', results['corpus_bleu']),
            ('Corpus chrF', 'corpus_chrf', results['corpus_chrf']),
            ('Mean Sentence chrF', 'sentence_chrf', results['sentence_chrf'])
        ]
        
        for metric_name, metric_key, metric_data in metric_configs:
            if language_pair in metric_data and metric_data[language_pair]:
                create_horizontal_bar_chart(
                    metric_data[language_pair],
                    f'{metric_name} Scores for {language_pair}',
                    f'{metric_name} Score',
                    f'{metric_key}_scores',
                    lang_output_dir
                )
        
        # Generate CSV report
        report_data = []
        # Sort by Corpus BLEU as primary metric
        for model, metrics in sorted(model_metrics.items(), 
                                    key=lambda x: x[1]['Corpus BLEU'], reverse=True):
            report_data.append({
                'Model': model,
                'Corpus BLEU': f"{metrics['Corpus BLEU']:.2f}",
                'Corpus chrF': f"{metrics['Corpus chrF']:.2f}",
                'Mean Sentence chrF': f"{metrics['Mean Sentence chrF']:.2f}",
                'Corpus BLEU_raw': metrics['Corpus BLEU'],
                'Corpus chrF_raw': metrics['Corpus chrF'],
                'Mean Sentence chrF_raw': metrics['Mean Sentence chrF']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(os.path.join(lang_output_dir, 'detailed_report.csv'), index=False)
        
        # Generate summary
        best_model_by_bleu = max(model_metrics.items(), key=lambda x: x[1]['Corpus BLEU'])
        
        summary = {
            'language_pair': language_pair,
            'timestamp': datetime.now().isoformat(),
            'models': {
                'corpus_bleu': results['corpus_bleu'].get(language_pair, {}),
                'corpus_chrf': results['corpus_chrf'].get(language_pair, {}),
                'sentence_chrf': results['sentence_chrf'].get(language_pair, {})
            },
            'best_model': best_model_by_bleu[0],
            'best_model_scores': best_model_by_bleu[1]
        }
        
        with open(os.path.join(lang_output_dir, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Text summary
        with open(os.path.join(lang_output_dir, 'summary.txt'), 'w') as f:
            f.write(f"MT Evaluation Results for {language_pair}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nModel Performance (sorted by Corpus BLEU score):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<30} {'Corpus BLEU':>12} {'Corpus chrF':>12} {'Mean Sent chrF':>15}\n")
            f.write("-" * 80 + "\n")
            for model, metrics in sorted(model_metrics.items(), 
                                        key=lambda x: x[1]['Corpus BLEU'], reverse=True):
                f.write(f"{model:<30} {metrics['Corpus BLEU']:>12.2f} {metrics['Corpus chrF']:>12.2f} "
                       f"{metrics['Mean Sentence chrF']:>15.2f}\n")
            f.write("-" * 80 + "\n")
            f.write(f"\nBest Model (by Corpus BLEU): {best_model_by_bleu[0]}\n")
            f.write(f"  Corpus BLEU: {best_model_by_bleu[1]['Corpus BLEU']:.2f}\n")
            f.write(f"  Corpus chrF: {best_model_by_bleu[1]['Corpus chrF']:.2f}\n")
            f.write(f"  Mean Sentence chrF: {best_model_by_bleu[1]['Mean Sentence chrF']:.2f}\n")

def generate_overall_summary(results, source_breakdown, 
                            output_dir="/home/owusus/Documents/GitHub/nsanku-mt/reports_combined"):
    """Generate overall summary across all language pairs and metrics"""
    if not results['corpus_bleu']:
        return
    
    print("Generating overall summary...")
    
    # Calculate overall model performance (average across all language pairs)
    all_models = set()
    for lang_results in results['corpus_bleu'].values():
        all_models.update(lang_results.keys())
    
    model_performance = {
        'corpus_bleu': {},
        'corpus_chrf': {},
        'sentence_chrf': {}
    }
    
    for model in all_models:
        for metric_key in ['corpus_bleu', 'corpus_chrf', 'sentence_chrf']:
            scores = []
            for lang_results in results[metric_key].values():
                if model in lang_results:
                    scores.append(lang_results[model])
            if scores:
                model_performance[metric_key][model] = np.mean(scores)
    
    # Generate overall model performance charts
    metric_configs = [
        ('Corpus BLEU', 'corpus_bleu'),
        ('Corpus chrF', 'corpus_chrf'),
        ('Mean Sentence chrF', 'sentence_chrf')
    ]
    
    for metric_name, metric_key in metric_configs:
        if model_performance[metric_key]:
            create_horizontal_bar_chart(
                model_performance[metric_key],
                f'Overall {metric_name} Performance Across All Languages',
                f'{metric_name} Score',
                f'overall_{metric_key}',
                output_dir
            )
    
    # Generate combined metrics chart for all models
    model_metrics_overall = {}
    for model in all_models:
        model_metrics_overall[model] = {
            'Corpus BLEU': model_performance['corpus_bleu'].get(model, 0),
            'Corpus chrF': model_performance['corpus_chrf'].get(model, 0),
            'Mean Sentence chrF': model_performance['sentence_chrf'].get(model, 0)
        }
    
    create_metric_comparison_chart(
        model_metrics_overall,
        'Overall MT Metrics Comparison Across All Languages',
        'overall_metrics_comparison',
        output_dir
    )
    
    # Calculate language performance (using corpus BLEU as primary metric)
    language_performance = {}
    for language_pair in results['corpus_bleu'].keys():
        scores = list(results['corpus_bleu'][language_pair].values())
        if scores:
            display_name = get_language_display_name(language_pair)
            language_performance[display_name] = np.mean(scores)
    
    create_horizontal_bar_chart(
        language_performance,
        'Language Performance Across Models (Corpus BLEU)',
        'Corpus BLEU Score',
        'language_performance',
        output_dir
    )
    
    # Save overall summary
    best_model = max(model_performance['corpus_bleu'].items(), 
                    key=lambda x: x[1]) if model_performance['corpus_bleu'] else ("none", 0)
    best_language = max(language_performance.items(), 
                       key=lambda x: x[1]) if language_performance else ("none", 0)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_language_pairs': len(results['corpus_bleu']),
        'total_models': len(all_models),
        'model_performance': model_performance,
        'language_performance': language_performance,
        'best_overall_model': best_model[0],
        'best_overall_scores': {
            'Corpus BLEU': model_performance['corpus_bleu'].get(best_model[0], 0),
            'Corpus chrF': model_performance['corpus_chrf'].get(best_model[0], 0),
            'Mean Sentence chrF': model_performance['sentence_chrf'].get(best_model[0], 0)
        },
        'best_language': best_language[0],
        'best_language_score': best_language[1]
    }
    
    with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create text summary
    with open(os.path.join(output_dir, 'overall_summary.txt'), 'w') as f:
        f.write("OVERALL MT EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Language Pairs: {summary['total_language_pairs']}\n")
        f.write(f"Total Models: {summary['total_models']}\n\n")
        
        f.write("BEST OVERALL MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {summary['best_overall_model']}\n")
        f.write(f"  Corpus BLEU: {summary['best_overall_scores']['Corpus BLEU']:.2f}\n")
        f.write(f"  Corpus chrF: {summary['best_overall_scores']['Corpus chrF']:.2f}\n")
        f.write(f"  Mean Sentence chrF: {summary['best_overall_scores']['Mean Sentence chrF']:.2f}\n\n")
        
        f.write("MODEL RANKINGS (by Corpus BLEU Score)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Model':<30} {'Corpus BLEU':>12} {'Corpus chrF':>12} {'Mean Sent chrF':>15}\n")
        f.write("-" * 80 + "\n")
        
        sorted_models = sorted(model_metrics_overall.items(), 
                             key=lambda x: x[1]['Corpus BLEU'], reverse=True)
        for rank, (model, metrics) in enumerate(sorted_models, 1):
            f.write(f"{rank:<6} {model:<30} {metrics['Corpus BLEU']:>12.2f} {metrics['Corpus chrF']:>12.2f} "
                   f"{metrics['Mean Sentence chrF']:>15.2f}\n")
    
    return summary

def generate_report(input_dir="/home/owusus/Documents/GitHub/nsanku-mt/output_combined", 
                   output_dir="/home/owusus/Documents/GitHub/nsanku-mt/reports_combined"):
    """Main function to generate all MT evaluation reports"""
    print("="*80)
    print("GENERATING MT EVALUATION REPORTS")
    print("="*80)
    print("Metrics: Corpus BLEU, Corpus chrF, Mean Sentence chrF")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect results
    results, source_breakdown = collect_results(input_dir)
    
    if not results['corpus_bleu']:
        print("No processed results found. Please run metric calculations first.")
        return
    
    # Generate reports
    generate_language_specific_reports(results, source_breakdown, output_dir)
    overall_summary = generate_overall_summary(results, source_breakdown, output_dir)
    
    print(f"\n{'='*80}")
    print("REPORTS GENERATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    if overall_summary:
        print(f"\nBest Overall Model: {overall_summary['best_overall_model']}")
        print(f"  Corpus BLEU: {overall_summary['best_overall_scores']['Corpus BLEU']:.2f}")
        print(f"  Corpus chrF: {overall_summary['best_overall_scores']['Corpus chrF']:.2f}")
        print(f"  Mean Sentence chrF: {overall_summary['best_overall_scores']['Mean Sentence chrF']:.2f}")
        print(f"\nBest Language: {overall_summary['best_language']} "
              f"(Corpus BLEU: {overall_summary['best_language_score']:.2f})")
    
    print(f"{'='*80}")
    
    return results, overall_summary

if __name__ == "__main__":
    generate_report()
