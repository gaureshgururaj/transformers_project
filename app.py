import streamlit as st
from app_utils import (
    load_model, process_text, plot_gradients, plot_attention_heatmap,
    plot_attention_heads, plot_hidden_states, plot_activation_distribution
)

@st.cache_resource
def get_model(model_name):
    return load_model(model_name)

def main():
    st.title("Transformer Model Visualization")
    
    # Dropdown for model selection
    model_name = st.selectbox(
        "Choose a Transformer Model:",
        ["bert-base-uncased", "distilbert-base-uncased", "roberta-base", "albert-base-v2"]
    )
    
    # Load selected model
    st.write(f"Loading {model_name}...")
    model, tokenizer = get_model(model_name)
    
    # Text input
    text = st.text_input("Enter text to analyze:", "A cow jumped over the moon")
    
    if st.button("Analyze"):
        with st.spinner(f"Analyzing text with {model_name}..."):
            # Process text
            inputs, outputs, embeddings = process_text(text, model, tokenizer)
            
            # Display visualizations
            st.subheader("Gradient Analysis")
            st.pyplot(plot_gradients(embeddings, inputs, tokenizer))
            
            st.subheader("Attention Heatmap")
            st.pyplot(plot_attention_heatmap(outputs, inputs, tokenizer))
            
            st.subheader("Attention Heads")
            st.pyplot(plot_attention_heads(outputs, inputs, tokenizer))
            
            st.subheader("Hidden States Analysis")
            token_index = st.slider("Select token index to analyze:", 
                                  0, len(inputs['input_ids'][0])-1, 0)
            st.pyplot(plot_hidden_states(outputs, token_index))
            
            st.subheader("Layer-Wise Activation Distribution")
            st.pyplot(plot_activation_distribution(outputs))

if __name__ == "__main__":
    main()