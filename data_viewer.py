#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="🔍 Fragrance Data Explorer",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_processed_data():
    """Load all processed data files"""
    try:
        df = pd.read_pickle("processed_fragrances.pkl")
        vectors = np.load("fragrance_vectors.npy")
        notes_list = np.load("fragrance_notes.npy", allow_pickle=True)
        return df, vectors, notes_list
    except FileNotFoundError as e:
        st.error(f"Data files not found: {str(e)}")
        st.info("Please run train_model.py first to generate the processed data.")
        return None, None, None

def main():
    st.title("🔍 Fragrance Data Explorer")
    st.markdown("### Explore your processed Ajmal fragrance dataset")
    
    # Load data
    df, vectors, notes_list = load_processed_data()
    
    if df is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.header("📊 Data Views")
    view_option = st.sidebar.selectbox(
        "Choose what to explore:",
        ["Dataset Overview", "Fragrance Table", "Notes Analysis", "Feature Vectors", "Individual Fragrances"]
    )
    
    if view_option == "Dataset Overview":
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fragrances", len(df))
        with col2:
            st.metric("Unique Notes", len(notes_list))
        with col3:
            st.metric("Feature Dimensions", f"{vectors.shape[0]}×{vectors.shape[1]}")
        with col4:
            st.metric("Data Quality", "100%")
        
        # Data completeness
        st.subheader("🎯 Data Completeness")
        completeness_data = {
            'Note Type': ['Top Notes', 'Heart Notes', 'Base Notes'],
            'Missing': [
                df['Chrome_Top_Notes'].isna().sum(),
                df['Chrome_Heart_Notes'].isna().sum(), 
                df['Chrome_Base_Notes'].isna().sum()
            ],
            'Available': [
                len(df) - df['Chrome_Top_Notes'].isna().sum(),
                len(df) - df['Chrome_Heart_Notes'].isna().sum(),
                len(df) - df['Chrome_Base_Notes'].isna().sum()
            ]
        }
        
        completeness_df = pd.DataFrame(completeness_data)
        fig = px.bar(completeness_df, x='Note Type', y='Available', 
                    title='Data Availability by Note Type',
                    color='Note Type')
        st.plotly_chart(fig, use_container_width=True)
        
    elif view_option == "Fragrance Table":
        st.header("📝 Fragrance Collection Table")
        
        # Search functionality
        search_term = st.text_input("🔍 Search fragrances:", placeholder="Type fragrance name...")
        
        # Filter data based on search
        if search_term:
            filtered_df = df[df['product_name'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df
        
        st.subheader(f"Showing {len(filtered_df)} fragrances")
        
        # Display table with formatting
        display_df = filtered_df[['product_name', 'Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']].copy()
        display_df.columns = ['Fragrance Name', 'Top Notes', 'Heart Notes', 'Base Notes']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="ajmal_fragrances.csv",
            mime="text/csv"
        )
        
    elif view_option == "Notes Analysis":
        st.header("🌿 Fragrance Notes Analysis")
        
        # Calculate note frequencies
        all_notes_count = {}
        for idx, row in df.iterrows():
            for note_type in ['Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']:
                notes_str = row[note_type]
                if notes_str and pd.notna(notes_str):
                    # Simple parsing (split by common separators)
                    notes = notes_str.lower().replace('(', '').replace(')', '').replace(',', ' ').split()
                    for note in notes:
                        note = note.strip()
                        if len(note) > 2:
                            all_notes_count[note] = all_notes_count.get(note, 0) + 1
        
        # Top notes chart
        top_notes = sorted(all_notes_count.items(), key=lambda x: x[1], reverse=True)[:20]
        
        notes_df = pd.DataFrame(top_notes, columns=['Note', 'Frequency'])
        
        fig = px.bar(notes_df, x='Frequency', y='Note', orientation='h',
                    title='Top 20 Most Common Fragrance Notes',
                    labels={'Frequency': 'Number of Fragrances'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Notes by position
        st.subheader("📊 Notes Distribution by Position")
        
        position_counts = {
            'Top Notes': sum(1 for _, row in df.iterrows() if row['Chrome_Top_Notes'] and pd.notna(row['Chrome_Top_Notes'])),
            'Heart Notes': sum(1 for _, row in df.iterrows() if row['Chrome_Heart_Notes'] and pd.notna(row['Chrome_Heart_Notes'])),
            'Base Notes': sum(1 for _, row in df.iterrows() if row['Chrome_Base_Notes'] and pd.notna(row['Chrome_Base_Notes']))
        }
        
        fig_pie = px.pie(values=list(position_counts.values()), 
                        names=list(position_counts.keys()),
                        title='Distribution of Note Positions')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed notes table
        st.subheader("📋 Complete Notes List")
        all_notes_df = pd.DataFrame(list(all_notes_count.items()), 
                                   columns=['Note', 'Frequency']).sort_values('Frequency', ascending=False)
        st.dataframe(all_notes_df, use_container_width=True, hide_index=True)
        
    elif view_option == "Feature Vectors":
        st.header("📈 Feature Vector Analysis")
        
        st.subheader("🔢 Vector Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shape", f"{vectors.shape[0]}×{vectors.shape[1]}")
        with col2:
            st.metric("Mean", f"{vectors.mean():.3f}")
        with col3:
            st.metric("Std Dev", f"{vectors.std():.3f}")
        with col4:
            st.metric("Sparsity", f"{(vectors == 0).sum() / vectors.size * 100:.1f}%")
        
        # Heatmap of feature vectors
        st.subheader("🎨 Feature Vector Heatmap")
        
        # Show a sample of vectors (first 20 fragrances, first 30 features)
        sample_vectors = vectors[:20, :30]
        sample_names = [df.iloc[i]['product_name'][:20] for i in range(min(20, len(df)))]
        sample_features = [f"F{i+1}" for i in range(min(30, vectors.shape[1]))]
        
        fig_heatmap = px.imshow(sample_vectors, 
                               x=sample_features,
                               y=sample_names,
                               title="Feature Vectors Heatmap (Sample: 20 fragrances × 30 features)",
                               aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        feature_importance = np.abs(vectors).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': [f"Feature_{i+1}" for i in range(len(feature_importance))],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(20)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                              title='Top 20 Most Important Features')
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        
    elif view_option == "Individual Fragrances":
        st.header("🌸 Individual Fragrance Explorer")
        
        # Fragrance selector
        fragrance_names = [f"{row['product_name']} (#{idx})" for idx, row in df.iterrows()]
        selected_fragrance = st.selectbox("Choose a fragrance:", fragrance_names)
        
        if selected_fragrance:
            # Extract index
            idx = int(selected_fragrance.split('#')[-1].replace(')', ''))
            fragrance = df.iloc[idx]
            
            # Display fragrance details
            st.subheader(f"🌸 {fragrance['product_name']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Fragrance Notes:**")
                st.markdown(f"🔝 **Top Notes:** {fragrance['Chrome_Top_Notes']}")
                st.markdown(f"💗 **Heart Notes:** {fragrance['Chrome_Heart_Notes']}")
                st.markdown(f"🌿 **Base Notes:** {fragrance['Chrome_Base_Notes']}")
            
            with col2:
                # Feature vector for this fragrance
                frag_vector = vectors[idx]
                non_zero_count = np.count_nonzero(frag_vector)
                
                st.markdown("**Feature Vector Stats:**")
                st.metric("Non-zero Features", f"{non_zero_count}/{len(frag_vector)}")
                st.metric("Vector Norm", f"{np.linalg.norm(frag_vector):.3f}")
                st.metric("Max Value", f"{frag_vector.max():.3f}")
                st.metric("Min Value", f"{frag_vector.min():.3f}")
            
            # Vector visualization
            st.subheader("📊 Feature Vector Visualization")
            fig_vector = px.bar(x=range(len(frag_vector)), y=frag_vector,
                              title=f"Feature Vector for {fragrance['product_name']}")
            fig_vector.update_layout(
                xaxis_title="Feature Index",
                yaxis_title="Feature Value",
                showlegend=False
            )
            st.plotly_chart(fig_vector, use_container_width=True)

if __name__ == "__main__":
    main()