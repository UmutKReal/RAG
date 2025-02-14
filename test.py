import streamlit as st
from transformers import pipeline
import torch

def main():
    # PyTorch ile basit bir tensor oluşturun
    x = torch.rand(5, 3)
    st.write(f"PyTorch Tensor: {x}")
    
    # HuggingFace modeli yükleyin ve metin oluşturun
    model = pipeline("text-generation", model="gpt2")
    generated_text = model("Merhaba, nasılsınız?")
    st.write(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    main()

