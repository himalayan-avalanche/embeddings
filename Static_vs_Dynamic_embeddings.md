# Static and Dynamic Embeddings

### 1. Static Embeddings

Static embeddings are fixed vector representations of words. Once a word’s embedding is learned, it is the same regardless of the context in which the word appears. These embeddings are pre-trained and do not change based on sentence context.

#### Characteristics:

	•	Context-agnostic: Each word has a single vector representation, no matter where or how it appears in different sentences.
	•	Pretrained and Fixed: These embeddings are usually pre-trained on large corpora and remain unchanged during downstream tasks.
	•	Efficiency: They are computationally less expensive because you don’t need to recompute embeddings every time the word appears in a new context.

#### Examples of Static Embeddings:

	•	Word2Vec: Word2Vec generates fixed embeddings for each word by predicting words that appear in the same context.
	•	GloVe: GloVe embeddings are also context-agnostic. They are obtained by factorizing the word co-occurrence matrix from a corpus, meaning each word has the same embedding everywhere.
	•	FastText: FastText provides embeddings for subwords, but still, each word’s embedding remains fixed across all contexts.

#### Limitations:

	•	Lack of context: Since a word like “bank” has the same representation whether it’s used in “river bank” or “bank account,” static embeddings can’t disambiguate words based on their meaning in different contexts.
	•	Outdated information: Static embeddings can’t be updated dynamically to reflect newer usage or trends in language.

#### Example with Word2Vec:

In Word2Vec, the word “apple” will always have the same vector representation whether it’s used in the sentence “Apple Inc. is a tech company” or “I love eating apples.”

	•	Type: Dynamic Embedding
	•	Architecture: Transformer-based, multi-modal
	•	Description: VilBERT is an extension of BERT that handles both vision and language inputs, learning to align image regions with corresponding text tokens. It is used in tasks like visual question answering and image captioning.
	•	Use Case: Multimodal tasks, visual question answering, image captioning.

 ```python
from gensim.models import Word2Vec

# Load a pre-trained Word2Vec model
model = Word2Vec.load("path_to_word2vec_model")

# Get the embedding for the word "apple"
apple_embedding = model.wv['apple']  # The same embedding across all contexts
```

### 2. Dynamic (Contextual) Embeddings

Dynamic embeddings, also called contextual embeddings, provide word or token embeddings that change based on the surrounding context. The embedding for a word varies depending on how it’s used in a sentence or passage, capturing different meanings or nuances based on the context.

#### Characteristics:

	•	Context-aware: Dynamic embeddings take into account the context in which a word appears. The same word can have different embeddings depending on the sentence or phrase it’s part of.
	•	Generated on the fly: These embeddings are generated dynamically during model inference, usually using deep neural network architectures like transformers.
	•	More expressive: Because they are sensitive to context, dynamic embeddings can capture nuances in meaning and syntax, improving the performance of tasks like machine translation, named entity recognition, and question answering.

#### Examples of Dynamic Embeddings:

	•	ELMo (Embeddings from Language Models): ELMo generates different embeddings for a word based on the context in which it appears by using bidirectional LSTMs.
	•	BERT (Bidirectional Encoder Representations from Transformers): BERT generates contextual embeddings based on the input sentence, capturing the word’s meaning relative to other words in the sentence.


```python
from transformers import BertTokenizer, BertModel
import torch

# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Two different sentences with the word "apple"
sentence_1 = "I love eating apples."
sentence_2 = "Apple Inc. is a tech giant."

# Tokenize the sentences
inputs_1 = tokenizer(sentence_1, return_tensors="pt")
inputs_2 = tokenizer(sentence_2, return_tensors="pt")

# Get the embeddings from BERT
outputs_1 = model(**inputs_1)
outputs_2 = model(**inputs_2)

# Extract the embeddings for the word "apple"
# For simplicity, we take the embeddings of the first occurrence of "apple"
apple_embedding_1 = outputs_1.last_hidden_state[0][3]  # Embedding in the context of "eating apples"
apple_embedding_2 = outputs_2.last_hidden_state[0][1]  # Embedding in the context of "Apple Inc."

# The embeddings for "apple" will be different in each case
```
#### Limitations:

	•	Computationally expensive: Since dynamic embeddings are generated on the fly during inference, they require more computation and memory.
	•	Complexity: They involve more complex architectures (transformers) and training processes, which might not be necessary for simpler tasks.

### Summary Table: Static vs. Dynamic Embeddings

| Feature                      | Static Embeddings                         | Dynamic (Contextual) Embeddings            |
|------------------------------|-------------------------------------------|--------------------------------------------|
| **Context Sensitivity**       | Context-agnostic (same for all contexts)  | Context-aware (varies by context)          |
| **Computational Cost**        | Low (precomputed, no need to re-generate) | High (generated on-the-fly during inference)|
| **Models**                    | Word2Vec, GloVe, FastText                 | BERT, GPT, ELMo, T5                        |
| **Representation Flexibility**| Fixed representations                     | Varies depending on surrounding words      |
| **Usage**                     | Simpler tasks (e.g., word similarity)     | More complex tasks (e.g., QA, summarization)|
| **Example**                   | `model.wv['apple']`                       | `model(**inputs).last_hidden_state[0][i]`  |


#### When to Use Static vs. Dynamic Embeddings:

	•	Static embeddings are useful for simpler tasks where the context doesn’t play a major role, such as basic text classification or when computational efficiency is a concern.
	•	Dynamic embeddings are better suited for more complex NLP tasks like question answering, machine translation, and context-dependent text generation, where the meaning of words changes depending on context.
