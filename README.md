# Prompt Engineering

This project explores the effectiveness of zero-shot and few-shot prompting strategies in both text-to-text and image-to-text tasks. The system integrates Flan-T5 (for text generation), BLIP-2 (for multimodal vision-language modeling), and SentenceTransformer (for evaluation) to generate and assess AI responses across different modalities.
 
 # Development Process
 The codebase was developed using Python, utilizing the following libraries:
 
 Torch
 
 Transformers
 
 Sentence-Transformers
 
 PIL (Python Imaging Library)
 
 Development and testing were conducted locally. Code implementation was carried out using Jupyter Notebook and PyCharm, depending on device compatibility. Collaborative analysis and discussion took place via a shared Google Doc.
 
 # Model Selection
 Initial experiments were conducted using:
 
 BLIP
 
 Flan-T5 Small and Base
 
 These configurations often produced inconsistent and incoherent outputs. To enhance quality and reliability, the system was upgraded to:
 
 Flan-T5 Large, an instruction-optimized LLM developed by Google
 
 BLIP-2, a multimodal transformer suitable for image captioning and visual reasoning tasks
 
 Each model configuration was evaluated with five examples for consistency and comparison.
 
 # Hyperparameter Tuning
 Flan-T5 Parameters:
 max_length = 300
 
 do_sample = True
 
 early_stopping = False
 
 num_beams = 2
 
 top_p = 0.9
 
 temperature = 1.0
 
 These parameters were found to balance originality, accuracy, and response length.
 
 BLIP-2 Parameters:
 max_new_tokens = 25
 
 temperature = 0.2
 
 num_beams = 5
 
 top_p = 0.6
 
 repetition_penalty = 1.2
 
 This configuration yielded concise, structured, and non-repetitive outputs, especially in few-shot settings.
 
 # System Design
 The following considerations were prioritized in the system architecture:
 
 Hyperparameter interaction effects were carefully managed to avoid suboptimal combinations.
 
 Stopping criteria for generation were tuned to balance informativeness and verbosity.
 
 Prompt phrasing and parameter configurations were refined through iterative testing.
 
 Model performance was evaluated using a structured approach:
 
 Semantic similarity was measured using the evaluate_similarity function with SentenceTransformer embeddings.
 
 Output length and relevance were also considered as part of the evaluation strategy.
