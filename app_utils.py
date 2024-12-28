import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_name):
    """Load the specified transformer model and tokenizer"""
    model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def process_text(text, model, tokenizer):
    """Process input text through the specified transformer model"""
    inputs = tokenizer(text, return_tensors="pt")
    
    embeddings = model.embeddings.word_embeddings(inputs['input_ids'])
    embeddings.retain_grad()
    
    outputs = model(inputs_embeds=embeddings)
    
    loss = outputs.last_hidden_state.sum()
    loss.backward()
    
    return inputs, outputs, embeddings

def plot_gradients(embeddings, inputs, tokenizer):
    """Plot gradient analysis with corrected token display."""
    gradients = embeddings.grad
    average_gradients = gradients[0].mean(dim=1).detach().numpy()
    
    # Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Ensure tokens are a flat list of strings
    if isinstance(tokens[0], list):  # Flatten if nested
        tokens = [item for sublist in tokens for item in sublist]
    
    # Clean tokens
    cleaned_tokens = [str(token).replace("##", "").lstrip("G") for token in tokens]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(average_gradients, marker='o')
    ax.set_title("Averaged Gradients for Input Tokens")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Average Gradient Value")
    plt.xticks(ticks=range(len(average_gradients)), labels=cleaned_tokens, rotation=45)
    plt.grid(True)
    return fig


def plot_attention_heatmap(outputs, inputs, tokenizer):
    """Plot attention heatmap"""
    attention = outputs.attentions
    attention_matrix = attention[0][0][0].detach().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_matrix, 
                xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), 
                cmap="viridis")
    plt.title("Attention Weights")
    return fig

def plot_attention_heads(outputs, inputs, tokenizer):
    """Plot attention heads"""
    attention = outputs.attentions
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        sns.heatmap(attention[0][0][i].detach().numpy(), 
                    ax=ax, 
                    cmap="viridis",
                    xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                    yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        ax.set_title(f"Head {i+1}")
    plt.tight_layout()
    return fig

def plot_hidden_states(outputs, token_index=0):
    """Plot hidden states analysis"""
    hidden_states = outputs.hidden_states
    cls_hidden_states = [state[:, token_index, :].detach().numpy() for state in hidden_states]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot([state.mean() for state in cls_hidden_states])
    plt.title(f"Mean Hidden State of Token {token_index} Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Mean Activation")
    return fig

def plot_activation_distribution(outputs):
    """Plot layer-wise activation distribution"""
    hidden_states = outputs.hidden_states
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, layer in enumerate(hidden_states):
        sns.kdeplot(layer[0].detach().numpy().flatten(), ax=ax, label=f"Layer {i+1}", bw_adjust=0.5)
    
    plt.title("Activation Distribution Across Layers")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")
    plt.legend()
    return fig
