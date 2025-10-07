import os, sys
import numpy as np

from transformers import pipeline

pipe = pipeline("text-generation", model="GroNLP/gpt2-medium-italian-embeddings")
from transformers import AutoTokenizer, AutoModel, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-medium-italian-embeddings")
model = AutoModel.from_pretrained("GroNLP/gpt2-medium-italian-embeddings")  # PyTorch
model = TFAutoModel.from_pretrained("GroNLP/gpt2-medium-italian-embeddings")  # Tensorflow

