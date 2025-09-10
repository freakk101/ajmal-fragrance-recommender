# 🌸 Ajmal Fragrance Recommender

An AI-powered fragrance recommendation system that uses deep learning to suggest perfumes based on your preferences. Built with PyTorch and Streamlit for real-time recommendations from Ajmal's fragrance collection.

## ✨ Features

- **🧠 Neural Network**: Custom autoencoder with contrastive learning
- **🎯 Personalized Recommendations**: Based on 1-3 selected fragrances  
- **📊 Fragrance Profile Analysis**: Visual breakdown of your scent preferences
- **🔍 Smart Search**: Find fragrances by name
- **⚡ Real-time**: Instant AI-powered recommendations
- **🎨 Beautiful UI**: Interactive Streamlit interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The app will automatically process data and train the model on first run.

## 🌐 Live Demo

**Streamlit Cloud:** [Visit Live App](https://your-app-url.streamlit.app)

## 📱 Local Development

### Optional: Manual Training
```bash
python train_model.py
```

## 🏗️ Architecture

### Neural Network Design
- **Autoencoder Architecture**: Learns compressed fragrance representations
- **Contrastive Learning**: Distinguishes similar vs dissimilar fragrances
- **Multi-layer Encoder**: 128 → 96 → 64 dimensional embeddings
- **Combined Loss**: Reconstruction + contrastive loss

### Data Processing
- **Feature Extraction**: Fragrance notes → weighted vectors
- **Note Weighting**: Top notes (3x), Heart notes (2x), Base notes (1x)
- **Normalization**: StandardScaler for consistent training

### Recommendation Engine
- **User Profile**: Average embeddings of selected fragrances
- **Similarity Metrics**: Cosine similarity + Euclidean distance
- **Smart Filtering**: Excludes already selected fragrances

## 📁 Project Structure

```
frag-recommender/
├── chrome_ajmal_fragrance_data_20250909_202423.csv  # Raw fragrance data
├── data_preprocessor.py                             # Data cleaning & feature extraction
├── neural_recommender.py                            # Neural network & training logic
├── train_model.py                                   # Model training script
├── streamlit_app.py                                 # Web application
├── requirements.txt                                 # Dependencies
├── CLAUDE.md                                        # Development guide
└── README.md                                        # This file
```

## 🎯 How It Works

1. **Data Preprocessing**: Extract and vectorize fragrance notes
2. **Neural Training**: Learn embeddings using autoencoder + contrastive loss
3. **User Selection**: Choose 1-3 favorite fragrances
4. **Profile Creation**: Average selected fragrance embeddings
5. **Recommendation**: Find most similar fragrances using learned embeddings

## 🔧 Technical Details

- **Framework**: PyTorch for neural networks, Streamlit for UI
- **Model**: Custom autoencoder with 64-dimensional embeddings
- **Training**: Contrastive learning with positive/negative fragrance pairs
- **Similarity**: Hybrid cosine similarity + Euclidean distance scoring
- **Real-time**: Pre-computed embeddings for instant recommendations

## 📊 Performance

- **Training**: ~50 epochs on fragrance note vectors
- **Inference**: Real-time recommendations (<100ms)
- **Accuracy**: Learns meaningful fragrance relationships from note compositions

## 🎨 Usage

1. **Search**: Use the sidebar to find fragrances by name
2. **Select**: Choose 1-3 fragrances you love  
3. **Analyze**: View your fragrance profile radar chart
4. **Discover**: Get AI-powered recommendations with match scores

## 🔄 Future Enhancements

- [ ] Add more fragrance attributes (longevity, sillage, season)
- [ ] Implement user feedback learning
- [ ] Add fragrance category clustering
- [ ] Include price-based filtering
- [ ] Mobile-responsive design improvements