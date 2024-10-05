# embeddings
#### Overview of embedding systems and various use cases.

Embeddings are dense, low-dimensional representations of data (such as words, sentences, or even images) in a continuous vector space. Different types of embeddings are used depending on the task and the data being processed. Below are some of the most common types of embeddings, primarily focusing on text embeddings, but also extending to images and graphs.

1. Static and Dynamic Embeddings:
   Static embeddings and dynamic embeddings refer to different ways in which text (usually words or tokens) is transformed into vector representations. Their distinction comes from how the embeddings are generated and whether they change based on the context or remain fixed across all contexts.

3. Various Types of Embeddings: https://github.com/himalayan-avalanche/embeddings/blob/main/Static_vs_Dynamic_embeddings.md



### 1. Word Embeddings

Word embeddings represent words in a continuous vector space where semantically similar words are located close to each other. Word embeddings capture semantic relationships and are widely used in NLP tasks.

#### a) Word2Vec

	•	Type: Static Embedding
	•	Architecture: Skip-gram and Continuous Bag of Words (CBOW)
	•	Description: Word2Vec generates embeddings where words with similar contexts appear close together in the vector space. It’s trained using two main models:
	•	Skip-gram: Predicts the surrounding words given a word.
	•	CBOW: Predicts a word based on its context (surrounding words).
	•	Use Case: Text classification, sentiment analysis, semantic similarity.

#### b) GloVe (Global Vectors for Word Representation)

	•	Type: Static Embedding
	•	Architecture: Matrix factorization
	•	Description: GloVe builds word embeddings by factoring the co-occurrence matrix of words. Unlike Word2Vec, it captures both global (document-level) and local (window-level) statistical information about words.
	•	Use Case: Similar to Word2Vec, but GloVe is particularly useful when global co-occurrence information is critical.

#### c) FastText

	•	Type: Static Embedding
	•	Architecture: Word2Vec variant with subword information
	•	Description: FastText improves on Word2Vec by considering character n-grams (subwords) when building word embeddings. This helps handle out-of-vocabulary (OOV) words, such as misspelled words or new words.
	•	Use Case: Handling morphologically rich languages, OOV issues.

### 2. Sentence and Document Embeddings

Sentence or document embeddings represent entire sentences, paragraphs, or documents as vectors, rather than individual words.

#### a) Doc2Vec

	•	Type: Static Embedding
	•	Architecture: Extension of Word2Vec
	•	Description: Doc2Vec learns vector representations for entire documents (or sentences) rather than individual words. It builds on Word2Vec and adds document-level context during training.
	•	Use Case: Document classification, document clustering.

#### b) Universal Sentence Encoder (USE)

	•	Type: Dynamic Embedding
	•	Architecture: Transformer-based or Deep Averaging Network (DAN)
	•	Description: USE encodes sentences into fixed-length embeddings and is pre-trained on various tasks to capture general sentence semantics. It uses either a Transformer or a deep averaging network (DAN).
	•	Use Case: Sentence-level tasks like semantic similarity, sentence classification.

#### c) Sentence-BERT (SBERT)

	•	Type: Dynamic Embedding
	•	Architecture: BERT-based
	•	Description: SBERT is a variant of BERT fine-tuned to produce meaningful sentence embeddings that work well for tasks like sentence similarity. It uses BERT or other transformer models, and introduces a pooling mechanism to convert word embeddings into sentence embeddings.
	•	Use Case: Sentence similarity, text clustering, information retrieval.

### 3. Contextual Word Embeddings

Unlike static embeddings like Word2Vec or GloVe, contextual embeddings generate different embeddings for the same word depending on the surrounding context. These embeddings capture word meaning in various contexts and are typically generated using transformer-based models.

#### a) ELMo (Embeddings from Language Models)

	•	Type: Dynamic Embedding
	•	Architecture: Bidirectional LSTM
	•	Description: ELMo generates context-dependent embeddings for words. It uses a bidirectional LSTM to model word context, providing different embeddings for the same word in different sentences.
	•	Use Case: Named Entity Recognition (NER), sentiment analysis, machine translation.

#### b) BERT (Bidirectional Encoder Representations from Transformers)

	•	Type: Dynamic Embedding
	•	Architecture: Transformer-based, bidirectional attention
	•	Description: BERT generates context-dependent word and sentence embeddings by processing the entire input sequence bidirectionally. BERT is pre-trained on large corpora and fine-tuned for various tasks.
	•	Use Case: Text classification, question answering, named entity recognition.

#### c) GPT (Generative Pre-trained Transformer)

	•	Type: Dynamic Embedding
	•	Architecture: Transformer-based, autoregressive
	•	Description: GPT generates embeddings based on an autoregressive model (predicting the next word based on the previous context). Unlike BERT, which processes text bidirectionally, GPT processes text from left to right.
	•	Use Case: Text generation, dialogue systems, conversational agents.

### 4. Image Embeddings

Image embeddings represent images as vectors, capturing the visual content of the image in a low-dimensional space. These embeddings are typically learned using convolutional neural networks (CNNs).

#### a) CNN-based Embeddings

	•	Type: Static/Dynamic Embedding
	•	Architecture: Convolutional Neural Networks (e.g., ResNet, VGG)
	•	Description: CNNs are used to encode images into dense vector representations. Pre-trained models like ResNet, VGG, and InceptionNet provide feature embeddings for images, which can be used for tasks like classification or similarity detection.
	•	Use Case: Image classification, object detection, image retrieval.

#### b) CLIP (Contrastive Language-Image Pretraining)

	•	Type: Dynamic Embedding
	•	Architecture: Transformer-based
	•	Description: CLIP is a vision-language model trained to associate images and textual descriptions. It learns embeddings for both images and text and can retrieve images based on text and vice versa.
	•	Use Case: Multimodal tasks, image captioning, image retrieval.

### 5. Graph Embeddings

Graph embeddings represent nodes, edges, or entire graphs as vectors, capturing the structural and relational properties of the graph.

#### a) Node2Vec

	•	Type: Static Embedding
	•	Architecture: Random walks and Skip-gram
	•	Description: Node2Vec learns embeddings for graph nodes by simulating random walks and applying the Skip-gram model to those walks. Nodes that appear in similar neighborhoods are placed closer together in the embedding space.
	•	Use Case: Node classification, link prediction, graph clustering.

#### b) GraphSAGE

	•	Type: Dynamic Embedding
	•	Architecture: Graph Convolutional Network (GCN)
	•	Description: GraphSAGE generates embeddings for nodes by sampling and aggregating features from neighboring nodes. It can generate embeddings for previously unseen nodes (inductive learning).
	•	Use Case: Node classification, inductive learning on graphs.

### 6. Knowledge Graph Embeddings

These embeddings represent entities and relations in a knowledge graph as vectors in a continuous vector space. They aim to capture the relationships between entities and can be used for tasks like link prediction.

#### a) TransE (Translational Embedding)

	•	Type: Static Embedding
	•	Architecture: Translation-based model
	•	Description: TransE embeds entities and relations of a knowledge graph in a vector space such that the relationship between entities is modeled as a translation in that space.
	•	Use Case: Link prediction, knowledge graph completion.

#### b) ComplEx

	•	Type: Static Embedding
	•	Architecture: Complex numbers
	•	Description: ComplEx embeds both entities and relations in a complex vector space. It can handle asymmetric relations better than TransE.
	•	Use Case: Link prediction, knowledge graph reasoning.

### 7. Multimodal Embeddings

These embeddings represent data from multiple modalities (e.g., text, image, audio) in a common vector space, enabling tasks like cross-modal retrieval or alignment.

#### a) VilBERT (Vision-and-Language BERT)

	•	Type: Dynamic Embedding
	•	Architecture: Transformer-based, multi-modal
	•	Description: VilBERT is an extension of BERT that handles both vision and language inputs, learning to align image regions with corresponding text tokens. It is used in tasks like visual question answering and image captioning.
	•	Use Case: Multimodal tasks, visual question answering, image captioning.
