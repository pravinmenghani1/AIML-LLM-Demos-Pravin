import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Page config
st.set_page_config(
    page_title="🧠 Encoder Architecture Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        return self.transformer(x)

@st.cache_data
def create_attention_heatmap(attention_weights):
    fig = px.imshow(
        attention_weights,
        color_continuous_scale='Viridis',
        title="Attention Weights Visualization"
    )
    fig.update_layout(height=400)
    return fig

def main():
    st.markdown('<h1 class="main-header">🧠 Encoder Architecture Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Model Configuration")
    vocab_size = st.sidebar.slider("Vocabulary Size", 100, 10000, 1000)
    d_model = st.sidebar.selectbox("Model Dimension", [128, 256, 512], index=1)
    nhead = st.sidebar.selectbox("Number of Heads", [4, 8, 16], index=1)
    num_layers = st.sidebar.slider("Number of Layers", 1, 6, 2)
    
    # Create model
    model = SimpleEncoder(vocab_size, d_model, nhead, num_layers)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>Vocabulary Size</h3><h2>{:,}</h2></div>'.format(vocab_size), unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>Model Dimension</h3><h2>{}</h2></div>'.format(d_model), unsafe_allow_html=True)
    
    with col3:
        total_params = sum(p.numel() for p in model.parameters())
        st.markdown('<div class="metric-card"><h3>Total Parameters</h3><h2>{:,}</h2></div>'.format(total_params), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input section
    st.markdown('<h2 class="section-header">📝 Input Processing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_text = st.text_area("Enter your text:", "The quick brown fox jumps over the lazy dog", height=100)
        seq_length = st.slider("Sequence Length", 5, 50, len(input_text.split()))
    
    with col2:
        # Tokenization visualization
        tokens = input_text.split()[:seq_length]
        token_ids = [hash(token) % vocab_size for token in tokens]
        
        df = pd.DataFrame({
            'Token': tokens,
            'Token ID': token_ids,
            'Position': range(len(tokens))
        })
        
        st.dataframe(df, use_container_width=True)
    
    # Model processing
    if st.button("🚀 Process with Encoder", type="primary"):
        with st.spinner("Processing..."):
            # Create input tensor
            input_ids = torch.tensor([token_ids])
            
            # Forward pass
            with torch.no_grad():
                output = model(input_ids)
            
            # Visualizations
            st.markdown('<h2 class="section-header">📊 Model Output Visualization</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Output embeddings heatmap
                fig = px.imshow(
                    output[0].numpy(),
                    title="Output Embeddings",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Embedding distribution
                flat_embeddings = output[0].numpy().flatten()
                fig = px.histogram(
                    x=flat_embeddings,
                    nbins=50,
                    title="Embedding Value Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Attention visualization (simulated)
            st.markdown('<h2 class="section-header">👁️ Attention Patterns</h2>', unsafe_allow_html=True)
            
            # Create simulated attention weights
            attention_weights = torch.softmax(torch.randn(len(tokens), len(tokens)), dim=-1).numpy()
            
            fig = create_attention_heatmap(attention_weights)
            fig.update_xaxes(ticktext=tokens, tickvals=list(range(len(tokens))))
            fig.update_yaxes(ticktext=tokens, tickvals=list(range(len(tokens))))
            st.plotly_chart(fig, use_container_width=True)
    
    # Architecture explanation
    st.markdown("---")
    st.markdown('<h2 class="section-header">🏗️ Architecture Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Components:**
        - **Embedding Layer**: Converts tokens to dense vectors
        - **Positional Encoding**: Adds position information
        - **Multi-Head Attention**: Captures relationships between tokens
        - **Feed-Forward Networks**: Processes attention outputs
        - **Layer Normalization**: Stabilizes training
        """)
    
    with col2:
        # Architecture diagram (simplified)
        layers = ['Input', 'Embedding', 'Pos Encoding', f'{num_layers}x Encoder', 'Output']
        y_pos = list(range(len(layers)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0.5] * len(layers),
            y=y_pos,
            mode='markers+text',
            marker=dict(size=60, color='lightblue'),
            text=layers,
            textposition='middle center',
            textfont=dict(size=12)
        ))
        
        # Add arrows
        for i in range(len(layers)-1):
            fig.add_annotation(
                x=0.5, y=i+0.3,
                ax=0.5, ay=i+0.7,
                arrowhead=2, arrowsize=1, arrowwidth=2
            )
        
        fig.update_layout(
            title="Encoder Architecture Flow",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
