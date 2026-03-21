"""
calculate_mt_metrics.py
Calculate standard MT evaluation metrics: BLEU and chrF at corpus level,
chrF also at sentence level for per-sentence analysis
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import sacrebleu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_COMBINED_PATH = "/home/owusus/Documents/GitHub/nsanku-mt/output_combined"
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_COMBINED_PATH, "mt_metrics_summary.jsonl")
DEBUG_MODE = True
# ==============================================================================

print("="*70)
print("MT METRICS CALCULATOR (Corpus-level BLEU & chrF, Sentence-level chrF)")
print("="*70)


def calculate_sentence_chrf(hypothesis: str, reference: str) -> float:
    """Calculate chrF score for a single sentence pair"""
    if not hypothesis or not reference:
        return 0.0
    try:
        # sacrebleu expects list of references
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference])
        return chrf.score
    except Exception as e:
        if DEBUG_MODE:
            print(f"    chrF error: {e}")
        return 0.0


def calculate_corpus_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate corpus-level metrics (the proper way to evaluate MT)
    
    Returns dict with corpus_bleu and corpus_chrf scores
    """
    if not hypotheses or not references or len(hypotheses) != len(references):
        return {'corpus_bleu': 0.0, 'corpus_chrf': 0.0}
    
    try:
        # Format references as list of lists for sacrebleu
        refs_for_bleu = [[ref] for ref in references]
        
        # Corpus BLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, refs_for_bleu)
        corpus_bleu = bleu.score
        
        # Corpus chrF (more stable than averaging sentence chrF)
        chrf = sacrebleu.corpus_chrf(hypotheses, references)
        corpus_chrf = chrf.score
        
        return {
            'corpus_bleu': corpus_bleu,
            'corpus_chrf': corpus_chrf
        }
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"    Corpus metrics error: {e}")
        return {'corpus_bleu': 0.0, 'corpus_chrf': 0.0}


def get_valid_pairs(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for valid translation-reference pairs"""
    required_cols = ['translated', 'ref']
    if not all(col in df.columns for col in required_cols):
        return pd.Series([False] * len(df), index=df.index)
    
    return (
        df['translated'].notna() & 
        df['ref'].notna() & 
        (df['translated'].astype(str).str.strip() != '') &
        (df['ref'].astype(str).str.strip() != '') &
        (df['translated'].astype(str).str.lower() != 'nan') &
        (df['ref'].astype(str).str.lower() != 'nan')
    )


def process_file_metrics(file_path: str) -> Optional[Dict]:
    """
    Process a single CSV file:
    1. Calculate sentence-level chrF for each row (for per-sentence analysis)
    2. Calculate corpus-level BLEU and chrF for the entire file
    3. Store corpus metrics as file-level summary (NOT per-row)
    
    Returns summary dict or None if failed
    """
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        if 'translated' not in df.columns or 'ref' not in df.columns:
            if DEBUG_MODE:
                print(f"  ⚠ {filename}: Missing 'translated' or 'ref' columns")
            return None
        
        # Get valid pairs
        valid_mask = get_valid_pairs(df)
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            if DEBUG_MODE:
                print(f"  ⚠ {filename}: No valid translation pairs found")
            return None
        
        # Extract valid pairs
        hypotheses = df.loc[valid_mask, 'translated'].astype(str).str.strip().tolist()
        references = df.loc[valid_mask, 'ref'].astype(str).str.strip().tolist()
        
        # --- SENTENCE-LEVEL chrF (for per-sentence analysis) ---
        # Only calculate if column doesn't exist or has missing values
        if 'chrf_sentence' not in df.columns:
            df['chrf_sentence'] = np.nan
        
        needs_sentence_chrf = df.loc[valid_mask, 'chrf_sentence'].isna().any()
        
        if needs_sentence_chrf:
            chrf_scores = []
            for hyp, ref in zip(hypotheses, references):
                score = calculate_sentence_chrf(hyp, ref)
                chrf_scores.append(score)
            
            df.loc[valid_mask, 'chrf_sentence'] = chrf_scores
        
        # Set invalid rows to NaN (not 0, to distinguish from actual zero scores)
        df.loc[~valid_mask, 'chrf_sentence'] = np.nan
        
        # --- CORPUS-LEVEL METRICS (file-level, not per-row) ---
        corpus_metrics = calculate_corpus_metrics(hypotheses, references)
        
        # Store corpus metrics as metadata (same value for all rows - indicates it's file-level)
        df['corpus_bleu'] = corpus_metrics['corpus_bleu']
        df['corpus_chrf'] = corpus_metrics['corpus_chrf']
        
        # Save updated file
        df.to_csv(file_path, index=False)
        
        # Calculate mean sentence chrF for reporting
        mean_sentence_chrf = df.loc[valid_mask, 'chrf_sentence'].mean()
        
        if DEBUG_MODE:
            print(f"  ✓ {filename}:")
            print(f"      Corpus BLEU:  {corpus_metrics['corpus_bleu']:.2f}")
            print(f"      Corpus chrF:  {corpus_metrics['corpus_chrf']:.2f}")
            print(f"      Mean sent chrF: {mean_sentence_chrf:.2f}")
            print(f"      Sentences: {valid_count}")
        
        # Return file-level summary
        return {
            'file': filename,
            'filepath': file_path,
            'corpus_bleu': corpus_metrics['corpus_bleu'],
            'corpus_chrf': corpus_metrics['corpus_chrf'],
            'mean_sentence_chrf': mean_sentence_chrf,
            'num_sentences': valid_count,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ✗ Error processing {filename}: {e}")
        return None


def find_csv_files(folder_path: str) -> List[str]:
    """Find all CSV files recursively"""
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)


def save_summary(summaries: List[Dict]):
    """Save file-level summaries to JSONL for easy analysis"""
    try:
        with open(SUMMARY_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            for summary in summaries:
                f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        print(f"\n  Summary saved to: {SUMMARY_OUTPUT_PATH}")
    except Exception as e:
        print(f"\n  ⚠ Could not save summary: {e}")


def print_final_report(summaries: List[Dict]):
    """Print overall statistics across all files"""
    if not summaries:
        return
    
    print("\n" + "="*70)
    print("CORPUS-LEVEL SUMMARY ACROSS ALL FILES")
    print("="*70)
    
    corpus_bleus = [s['corpus_bleu'] for s in summaries]
    corpus_chrfs = [s['corpus_chrf'] for s in summaries]
    mean_sent_chrfs = [s['mean_sentence_chrf'] for s in summaries]
    
    print(f"Files processed: {len(summaries)}")
    print(f"Total sentences: {sum(s['num_sentences'] for s in summaries)}")
    print()
    print("CORPUS BLEU (per file):")
    print(f"  Mean: {np.mean(corpus_bleus):.2f}")
    print(f"  Std:  {np.std(corpus_bleus):.2f}")
    print(f"  Min:  {np.min(corpus_bleus):.2f}")
    print(f"  Max:  {np.max(corpus_bleus):.2f}")
    print()
    print("CORPUS chrF (per file):")
    print(f"  Mean: {np.mean(corpus_chrfs):.2f}")
    print(f"  Std:  {np.std(corpus_chrfs):.2f}")
    print(f"  Min:  {np.min(corpus_chrfs):.2f}")
    print(f"  Max:  {np.max(corpus_chrfs):.2f}")
    print()
    print("Mean Sentence chrF (per file):")
    print(f"  Mean: {np.mean(mean_sent_chrfs):.2f}")
    print(f"  Std:  {np.std(mean_sent_chrfs):.2f}")
    print(f"  Min:  {np.min(mean_sent_chrfs):.2f}")
    print(f"  Max:  {np.max(mean_sent_chrfs):.2f}")
    
    # Top and bottom performers
    print()
    print("Top 3 by Corpus BLEU:")
    top_bleu = sorted(summaries, key=lambda x: x['corpus_bleu'], reverse=True)[:3]
    for i, s in enumerate(top_bleu, 1):
        print(f"  {i}. {s['file']}: {s['corpus_bleu']:.2f}")
    
    print()
    print("Bottom 3 by Corpus BLEU:")
    bottom_bleu = sorted(summaries, key=lambda x: x['corpus_bleu'])[:3]
    for i, s in enumerate(bottom_bleu, 1):
        print(f"  {i}. {s['file']}: {s['corpus_bleu']:.2f}")


def main():
    print(f"\nOutput folder: {OUTPUT_COMBINED_PATH}")
    print(f"Debug mode: {DEBUG_MODE}")
    print("="*70)
    print("\nMETRIC EXPLANATION:")
    print("  • corpus_bleu:  Corpus-level BLEU (proper MT evaluation)")
    print("  • corpus_chrf:  Corpus-level chrF (character n-grams)")
    print("  • chrf_sentence: Sentence-level chrF (per-row analysis)")
    print("\nCorpus metrics are stored per-row for convenience but represent")
    print("the ENTIRE FILE's score, not individual sentence scores.")
    print("="*70)
    
    if not os.path.exists(OUTPUT_COMBINED_PATH):
        print(f"\n✗ Error: Path does not exist: {OUTPUT_COMBINED_PATH}")
        return
    
    # Find all CSV files
    print("\nScanning for CSV files...")
    csv_files = find_csv_files(OUTPUT_COMBINED_PATH)
    
    if not csv_files:
        print("⚠ No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each file
    print("\n" + "-"*70)
    print("PROCESSING FILES")
    print("-"*70)
    
    summaries = []
    
    for file_path in tqdm(csv_files, desc="Processing"):
        summary = process_file_metrics(file_path)
        if summary:
            summaries.append(summary)
    
    # Save and report
    if summaries:
        save_summary(summaries)
        print_final_report(summaries)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Files processed successfully: {len(summaries)}")
    print(f"Files with errors: {len(csv_files) - len(summaries)}")
    print(f"Total files: {len(csv_files)}")
    print()
    print("Output columns added to each CSV:")
    print("  - chrf_sentence:  Sentence-level chrF (0-100, per row)")
    print("  - corpus_bleu:    File-level BLEU (same for all rows)")
    print("  - corpus_chrf:    File-level chrF (same for all rows)")
    print()
    print("Note: corpus_bleu and corpus_chrf are FILE-LEVEL metrics.")
    print("      They are duplicated across rows for convenience only.")
    print("✓ Processing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
