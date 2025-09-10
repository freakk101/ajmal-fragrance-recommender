import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FragranceDataPreprocessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.fragrance_notes = set()
        self.note_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        self.df = pd.read_csv(self.csv_path)
        
        # Extract relevant columns
        self.df = self.df[['product-card__title', 'Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']].copy()
        
        # Clean product names
        self.df['product_name'] = self.df['product-card__title'].str.replace(r'Non-Alcoholic Attar.*', '', regex=True).str.strip()
        
        # Fill NaN values
        self.df['Chrome_Top_Notes'] = self.df['Chrome_Top_Notes'].fillna('')
        self.df['Chrome_Heart_Notes'] = self.df['Chrome_Heart_Notes'].fillna('')
        self.df['Chrome_Base_Notes'] = self.df['Chrome_Base_Notes'].fillna('')
        
        # Remove rows where all note columns are empty
        self.df = self.df[
            (self.df['Chrome_Top_Notes'] != '') | 
            (self.df['Chrome_Heart_Notes'] != '') | 
            (self.df['Chrome_Base_Notes'] != '')
        ].reset_index(drop=True)
        
        return self.df
    
    def extract_fragrance_notes(self) -> Dict[str, List[str]]:
        """Extract and standardize fragrance notes"""
        all_notes = {}
        
        for idx, row in self.df.iterrows():
            product_notes = {
                'top': self._parse_notes(row['Chrome_Top_Notes']),
                'heart': self._parse_notes(row['Chrome_Heart_Notes']),
                'base': self._parse_notes(row['Chrome_Base_Notes'])
            }
            all_notes[idx] = product_notes
            
            # Collect all unique notes
            for note_type in product_notes:
                self.fragrance_notes.update(product_notes[note_type])
        
        return all_notes
    
    def _parse_notes(self, notes_str: str) -> List[str]:
        """Parse fragrance notes from string"""
        if not notes_str or pd.isna(notes_str):
            return []
        
        # Clean the string
        notes_str = re.sub(r'[(),-]', ' ', notes_str)
        notes_str = re.sub(r'\s+', ' ', notes_str).strip()
        
        # Split and clean individual notes
        notes = []
        for note in notes_str.split():
            note = note.strip().lower()
            if len(note) > 2:  # Filter out very short words
                notes.append(note)
        
        return notes
    
    def create_note_vectors(self) -> np.ndarray:
        """Create binary vectors for fragrance notes"""
        notes_list = sorted(list(self.fragrance_notes))
        note_to_idx = {note: idx for idx, note in enumerate(notes_list)}
        
        vectors = []
        for idx, row in self.df.iterrows():
            vector = np.zeros(len(notes_list))
            
            # Parse notes for this fragrance
            top_notes = self._parse_notes(row['Chrome_Top_Notes'])
            heart_notes = self._parse_notes(row['Chrome_Heart_Notes'])
            base_notes = self._parse_notes(row['Chrome_Base_Notes'])
            
            # Set binary flags with different weights for note positions
            for note in top_notes:
                if note in note_to_idx:
                    vector[note_to_idx[note]] += 3  # Top notes get higher weight
            
            for note in heart_notes:
                if note in note_to_idx:
                    vector[note_to_idx[note]] += 2  # Heart notes get medium weight
            
            for note in base_notes:
                if note in note_to_idx:
                    vector[note_to_idx[note]] += 1  # Base notes get lower weight
            
            vectors.append(vector)
        
        return np.array(vectors), notes_list
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Get fully processed data"""
        self.load_and_clean_data()
        self.extract_fragrance_notes()
        vectors, notes_list = self.create_note_vectors()
        
        # Normalize vectors
        vectors = self.scaler.fit_transform(vectors)
        
        return self.df, vectors, notes_list