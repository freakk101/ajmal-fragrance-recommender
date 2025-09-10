import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import List
from sklearn_recommender import FragranceRecommender
from data_preprocessor import FragranceDataPreprocessor
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🌸 Ajmal Fragrance Recommender",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def initialize_data():
    """Initialize and process data if not already done"""
    csv_path = "chrome_ajmal_fragrance_data_20250909_202423.csv"
    
    if not os.path.exists(csv_path):
        st.error("❌ CSV data file not found!")
        st.info("Please ensure chrome_ajmal_fragrance_data_20250909_202423.csv is in the app directory.")
        return None, None, None
    
    # Check if processed files exist
    if (os.path.exists("processed_fragrances.pkl") and 
        os.path.exists("fragrance_vectors.npy") and 
        os.path.exists("fragrance_notes.npy")):
        
        # Load existing processed data
        df = pd.read_pickle("processed_fragrances.pkl")
        vectors = np.load("fragrance_vectors.npy")
        notes_list = np.load("fragrance_notes.npy", allow_pickle=True)
        return df, vectors, notes_list
    else:
        # Process data for the first time
        with st.spinner("🔄 Processing fragrance data for the first time..."):
            preprocessor = FragranceDataPreprocessor(csv_path)
            df, vectors, notes_list = preprocessor.get_processed_data()
            
            # Save processed data
            df.to_pickle("processed_fragrances.pkl")
            np.save("fragrance_vectors.npy", vectors)
            np.save("fragrance_notes.npy", np.array(notes_list, dtype=object))
            
            return df, vectors, notes_list

@st.cache_resource
def initialize_model(_vectors):
    """Initialize and train the recommendation model"""
    if os.path.exists("fragrance_model.pkl"):
        # Load existing model
        recommender = FragranceRecommender()
        recommender.load_model("fragrance_model.pkl")
        return recommender
    else:
        # Train model for the first time
        with st.spinner("🧠 Training AI model for the first time (this may take a moment)..."):
            recommender = FragranceRecommender(embedding_dim=32)
            recommender.train(_vectors, epochs=200)  # Reduced epochs for faster deployment
            
            # Save the trained model
            recommender.save_model("fragrance_model.pkl")
            return recommender

def display_fragrance_card(fragrance_info, index, score=None):
    """Display a fragrance card"""
    score_badge = ""
    if score is not None:
        score_badge = f"""
        <div style="margin-bottom: 5px;">
            <span style="
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 12px;
                font-weight: bold;
            ">
                Match: {score:.1%}
            </span>
        </div>
        """
    
    with st.container():
        st.markdown(f"""
        {score_badge}
        <div style="
            border: 2px solid #e1e1e1; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        ">
            <h4 style="color: #2c3e50; margin-bottom: 10px;">
                {fragrance_info['product_name']}
            </h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1;">
                    <p style="margin: 5px 0;"><strong>🔝 Top Notes:</strong> {fragrance_info.get('Chrome_Top_Notes', 'N/A')}</p>
                    <p style="margin: 5px 0;"><strong>💗 Heart Notes:</strong> {fragrance_info.get('Chrome_Heart_Notes', 'N/A')}</p>
                    <p style="margin: 5px 0;"><strong>🌿 Base Notes:</strong> {fragrance_info.get('Chrome_Base_Notes', 'N/A')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_fragrance_profile_chart(selected_fragrances, df):
    """Create a radar chart for fragrance notes"""
    if not selected_fragrances:
        return None
    
    # Extract notes from selected fragrances
    all_notes = set()
    for idx in selected_fragrances:
        fragrance = df.iloc[idx]
        for note_type in ['Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']:
            notes_str = fragrance.get(note_type, '')
            if notes_str and pd.notna(notes_str):
                notes = notes_str.lower().split()
                all_notes.update([note.strip() for note in notes if len(note.strip()) > 2])
    
    if len(all_notes) < 3:
        return None
    
    # Count note frequencies
    note_counts = {}
    for note in all_notes:
        note_counts[note] = 0
        for idx in selected_fragrances:
            fragrance = df.iloc[idx]
            for note_type in ['Chrome_Top_Notes', 'Chrome_Heart_Notes', 'Chrome_Base_Notes']:
                notes_str = fragrance.get(note_type, '')
                if notes_str and pd.notna(notes_str) and note in notes_str.lower():
                    note_counts[note] += 1
    
    # Create radar chart
    top_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[count for _, count in top_notes],
        theta=[note.title() for note, _ in top_notes],
        fill='toself',
        name='Your Fragrance Profile',
        line_color='rgb(229, 134, 6)',
        fillcolor='rgba(229, 134, 6, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([count for _, count in top_notes]) + 1]
            )),
        showlegend=True,
        title="Your Fragrance Profile",
        height=400
    )
    
    return fig

def main():
    """Main application"""
    st.title("🌸 Ajmal Fragrance Recommender")
    st.markdown("### Discover your perfect fragrance with AI-powered recommendations")
    
    # Initialize data and model
    data_result = initialize_data()
    if data_result is None:
        st.stop()
    
    df, vectors, notes_list = data_result
    
    if df is None or vectors is None:
        st.error("❌ Failed to load or process data")
        st.stop()
    
    recommender = initialize_model(vectors)
    
    if recommender is None:
        st.error("❌ Failed to load or train model")
        st.stop()
    
    st.success(f"✅ Loaded {len(df)} fragrances from Ajmal collection")
    
    # Sidebar for fragrance selection
    with st.sidebar:
        st.header("🎯 Select Your Favorite Fragrances")
        st.markdown("Choose 1-3 fragrances you like to get personalized recommendations:")
        
        # Search functionality
        search_term = st.text_input("🔍 Search fragrances:", placeholder="Type fragrance name...")
        
        # Filter fragrances based on search
        if search_term:
            filtered_df = df[df['product_name'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df
        
        # Fragrance selection
        selected_fragrances = []
        if len(filtered_df) > 0:
            fragrance_options = {f"{row['product_name']} (#{idx})": idx 
                               for idx, row in filtered_df.iterrows()}
            
            selected_names = st.multiselect(
                "Choose fragrances:",
                options=list(fragrance_options.keys()),
                max_selections=3,
                help="Select 1-3 fragrances you love"
            )
            
            selected_fragrances = [fragrance_options[name] for name in selected_names]
        
        # Recommendation settings
        st.header("⚙️ Recommendation Settings")
        num_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("👤 Your Selected Fragrances")
        
        if selected_fragrances:
            for idx in selected_fragrances:
                fragrance_info = df.iloc[idx]
                display_fragrance_card(fragrance_info, idx)
            
            # Show fragrance profile chart
            profile_chart = create_fragrance_profile_chart(selected_fragrances, df)
            if profile_chart:
                st.plotly_chart(profile_chart, use_container_width=True)
        else:
            st.info("👈 Please select fragrances from the sidebar to get started!")
    
    with col2:
        st.header("🎁 Recommended Fragrances")
        
        if selected_fragrances:
            try:
                with st.spinner("🧠 Generating AI-powered recommendations..."):
                    recommendations = recommender.recommend_fragrances(
                        selected_fragrances, 
                        top_k=num_recommendations
                    )
                
                if recommendations:
                    st.success(f"✨ Found {len(recommendations)} perfect matches for you!")
                    
                    for i, (idx, score) in enumerate(recommendations, 1):
                        fragrance_info = df.iloc[idx]
                        display_fragrance_card(fragrance_info, idx, score)
                else:
                    st.warning("No recommendations found. Try selecting different fragrances.")
                    
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
        else:
            st.info("Select your favorite fragrances to see AI-powered recommendations!")
    
    # Additional information
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This AI-powered fragrance recommender uses **machine learning** trained on Ajmal's fragrance collection to understand scent profiles and preferences.
        
        **Features:**
        - 🧠 **Machine Learning**: Uses scikit-learn autoencoder with PCA
        - 🎯 **Personalized**: Learns from your selected fragrances
        - 🔄 **Real-time**: Generates recommendations instantly
        - 📊 **Smart Matching**: Combines multiple similarity metrics
        
        **How to use:**
        1. Search and select 1-3 fragrances you love
        2. View your fragrance profile analysis  
        3. Get personalized AI recommendations
        4. Discover new scents that match your taste!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and scikit-learn • Data from Ajmal Perfumes")

if __name__ == "__main__":
    main()