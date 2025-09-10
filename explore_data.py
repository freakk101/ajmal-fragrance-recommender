#!/usr/bin/env python3

import pandas as pd
import numpy as np
from data_preprocessor import FragranceDataPreprocessor

def explore_fragrance_data():
    """Comprehensive data exploration for fragrance dataset"""
    print("🔍 FRAGRANCE DATA EXPLORATION")
    print("=" * 50)
    
    # Load and process data
    print("📊 Loading data...")
    preprocessor = FragranceDataPreprocessor("chrome_ajmal_fragrance_data_20250909_202423.csv")
    df, vectors, notes_list = preprocessor.get_processed_data()
    
    print(f"✅ Data loaded: {len(df)} fragrances, {len(notes_list)} unique notes\n")
    
    # 1. Basic Dataset Info
    print("📋 DATASET OVERVIEW")
    print("-" * 30)
    print(f"Total fragrances: {len(df)}")
    print(f"Total unique notes: {len(notes_list)}")
    print(f"Feature vector dimensions: {vectors.shape}")
    print()
    
    # 2. Sample fragrances table
    print("📝 SAMPLE FRAGRANCES")
    print("-" * 30)
    display_df = df[['product_name', 'Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']].head(10)
    print(display_df.to_string(index=False, max_colwidth=50))
    print()
    
    # 3. Note frequency analysis
    print("🌿 FRAGRANCE NOTES ANALYSIS")
    print("-" * 30)
    
    all_notes_count = {}
    for idx, row in df.iterrows():
        for note_type in ['Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']:
            notes_str = row[note_type]
            if notes_str and pd.notna(notes_str):
                notes = preprocessor._parse_notes(notes_str)
                for note in notes:
                    all_notes_count[note] = all_notes_count.get(note, 0) + 1
    
    # Top 15 most common notes
    top_notes = sorted(all_notes_count.items(), key=lambda x: x[1], reverse=True)[:15]
    print("Top 15 most common fragrance notes:")
    for i, (note, count) in enumerate(top_notes, 1):
        print(f"{i:2d}. {note.capitalize():<15} ({count} fragrances)")
    print()
    
    # 4. Note distribution by position
    print("📊 NOTE POSITION ANALYSIS")
    print("-" * 30)
    top_notes_count = 0
    heart_notes_count = 0
    base_notes_count = 0
    
    for idx, row in df.iterrows():
        if row['Chrome_Top_Notes'] and pd.notna(row['Chrome_Top_Notes']):
            top_notes_count += len(preprocessor._parse_notes(row['Chrome_Top_Notes']))
        if row['Chrome_Heart_Notes'] and pd.notna(row['Chrome_Heart_Notes']):
            heart_notes_count += len(preprocessor._parse_notes(row['Chrome_Heart_Notes']))
        if row['Chrome_Base_Notes'] and pd.notna(row['Chrome_Base_Notes']):
            base_notes_count += len(preprocessor._parse_notes(row['Chrome_Base_Notes']))
    
    print(f"Top notes total: {top_notes_count}")
    print(f"Heart notes total: {heart_notes_count}")
    print(f"Base notes total: {base_notes_count}")
    print()
    
    # 5. Data quality check
    print("🔍 DATA QUALITY CHECK")
    print("-" * 30)
    empty_top = df['Chrome_Top_Notes'].isna().sum()
    empty_heart = df['Chrome_Heart_Notes'].isna().sum()
    empty_base = df['Chrome_Base_Notes'].isna().sum()
    
    print(f"Missing top notes: {empty_top}/{len(df)} ({empty_top/len(df)*100:.1f}%)")
    print(f"Missing heart notes: {empty_heart}/{len(df)} ({empty_heart/len(df)*100:.1f}%)")
    print(f"Missing base notes: {empty_base}/{len(df)} ({empty_base/len(df)*100:.1f}%)")
    
    # Fragrances with all three note types
    complete_fragrances = df[
        (df['Chrome_Top_Notes'].notna()) & 
        (df['Chrome_Heart_Notes'].notna()) & 
        (df['Chrome_Base_Notes'].notna())
    ]
    print(f"Complete fragrances (all 3 note types): {len(complete_fragrances)}/{len(df)} ({len(complete_fragrances)/len(df)*100:.1f}%)")
    print()
    
    # 6. Feature vector statistics
    print("📈 FEATURE VECTOR STATISTICS")
    print("-" * 30)
    print(f"Vector shape: {vectors.shape}")
    print(f"Vector mean: {vectors.mean():.3f}")
    print(f"Vector std: {vectors.std():.3f}")
    print(f"Vector min: {vectors.min():.3f}")
    print(f"Vector max: {vectors.max():.3f}")
    print(f"Sparsity (zero values): {(vectors == 0).sum() / vectors.size * 100:.1f}%")
    print()
    
    # 7. Individual fragrance analysis
    print("🌸 DETAILED FRAGRANCE EXAMPLES")
    print("-" * 30)
    
    # Show 3 detailed examples
    example_indices = [0, len(df)//2, -1]
    for i, idx in enumerate(example_indices):
        if idx == -1:
            idx = len(df) - 1
        
        fragrance = df.iloc[idx]
        print(f"\n{i+1}. {fragrance['product_name']}")
        print(f"   Top Notes: {fragrance['Chrome_Top_Notes']}")
        print(f"   Heart Notes: {fragrance['Chrome_Heart_Notes']}")
        print(f"   Base Notes: {fragrance['Chrome_Base_Notes']}")
        
        # Show parsed notes
        top_parsed = preprocessor._parse_notes(fragrance['Chrome_Top_Notes'])
        heart_parsed = preprocessor._parse_notes(fragrance['Chrome_Heart_Notes'])
        base_parsed = preprocessor._parse_notes(fragrance['Chrome_Base_Notes'])
        
        print(f"   Parsed - Top: {top_parsed}")
        print(f"   Parsed - Heart: {heart_parsed}")
        print(f"   Parsed - Base: {base_parsed}")
        
        # Show feature vector stats for this fragrance
        frag_vector = vectors[idx]
        non_zero = np.count_nonzero(frag_vector)
        print(f"   Feature vector: {non_zero} non-zero features out of {len(frag_vector)}")
    
    print("\n" + "=" * 50)
    print("✅ Data exploration complete!")
    
    return df, vectors, notes_list, all_notes_count

def create_data_visualizations(df, vectors, notes_list, all_notes_count):
    """Create visualizations for the data"""
    print("\n📊 Creating data visualizations...")
    
    # Save comprehensive data summary
    with open("data_summary.txt", "w") as f:
        f.write("AJMAL FRAGRANCE DATA SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total fragrances: {len(df)}\n")
        f.write(f"Unique notes: {len(notes_list)}\n")
        f.write(f"Feature dimensions: {vectors.shape}\n\n")
        
        f.write("TOP 20 MOST COMMON NOTES:\n")
        f.write("-" * 30 + "\n")
        top_notes = sorted(all_notes_count.items(), key=lambda x: x[1], reverse=True)[:20]
        for i, (note, count) in enumerate(top_notes, 1):
            f.write(f"{i:2d}. {note.capitalize():<20} {count} fragrances\n")
        
        f.write("\nALL FRAGRANCES:\n")
        f.write("-" * 30 + "\n")
        for idx, row in df.iterrows():
            f.write(f"{idx+1:2d}. {row['product_name']}\n")
            f.write(f"    Top: {row['Chrome_Top_Notes']}\n")
            f.write(f"    Heart: {row['Chrome_Heart_Notes']}\n")
            f.write(f"    Base: {row['Chrome_Base_Notes']}\n\n")
    
    print("💾 Saved detailed summary to: data_summary.txt")
    print("📋 Saved processed data to: processed_fragrances.pkl")
    print("🔢 Saved feature vectors to: fragrance_vectors.npy")
    print("📝 Saved notes list to: fragrance_notes.npy")

if __name__ == "__main__":
    df, vectors, notes_list, all_notes_count = explore_fragrance_data()
    create_data_visualizations(df, vectors, notes_list, all_notes_count)