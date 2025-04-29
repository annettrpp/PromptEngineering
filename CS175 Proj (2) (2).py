import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration
)

# Initialize models for text-to-text and image-to-text
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Text-to-text model (Flan-T5)
model_name = "google/flan-t5-large"
t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
t5_tokenizer = AutoTokenizer.from_pretrained(model_name)

# BLIP-2 for image captioning
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# Text-to-text prompts
zero_shot_text_prompts = [
    "Describe the impact of climate change on marine life in detail.",
    "Mention a detail of photosynthesis.",
    "Describe an application of the Pythagorean theorem.",
    "Explain a key feature of database indexing.",
    "State an important principle of cybersecurity."
]

few_shot_text_prompts = [
    ("Climate change affects marine life in several ways. Here are examples:\n"
     "1. Rising temperatures lead to coral bleaching, harming marine ecosystems.\n"
     "2. Ocean acidification reduces marine biodiversity, affecting fish populations.\n"
     "3. Rising sea levels destroy coastal habitats, threatening many marine species.\n"
     "4. Another impact is"
     ),
    (
        "Here are the key details involved in the process of photosynthesis:\n"
        "Chlorophyll in the plant's leaves absorbs light energy from the sun.\n"
        "The Plantae's energy is then used to convert carbon dioxide from the air and water from the soil into glucose and oxygen.\n"
        "A byproduct of the chemical reaction is how oxygen being released.\n"
        "The process takes place in two stages: the light-dependent reactions and the Calvin cycle.\n\n"
        "An important detail of photosynthesis is"
    ),

    (
    "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides (a² + b² = c²).\n"
    "This theorem is widely used in geometry, trigonometry, and real-world applications like construction and navigation.\n"
    "It helps in determining distances, designing structures, and solving mathematical problems.\n\n"
    "An application of the Pythagorean theorem is"
    ),


    (
    "Database indexing improves the speed of data retrieval operations by creating a structure that allows the database to find records quickly.\n"
    "Indexes are commonly created on columns that are frequently used in search queries.\n"
    "There are different types of indexes, such as primary indexes, clustered indexes, and non-clustered indexes.\n\n"
    "A key feature of database indexing is"
    ),


    (
    "Cybersecurity is the practice of protecting systems, networks, and data from digital attacks.\n"
    "It involves techniques such as encryption, firewalls, multi-factor authentication, and intrusion detection.\n"
    "A strong cybersecurity framework prevents unauthorized access and data breaches.\n\n"
    "An important principle of cybersecurity is"
    )
]

#Image Paths
image_paths = [
    "fieldflower.jpg",
    "elephant.jpg",
    "test44.jpg",
]

few_shot_text_examples = [
    "A golden retriever runs joyfully across a green field, its fur glowing in the sunlight.",
    "A herd of elephants slowly walks through the savannah under a golden sunset, their trunks swaying.",
    "A powerful lion rests on a sunlit rock, its golden mane flowing in the warm breeze."
]


#TEXT-TO-TEXT HELPER

def text_to_text_prompt(prompt):
    formatted_prompt = f"Continue this pattern and provide a detailed new impact:\n{prompt}"
    inputs = t5_tokenizer(formatted_prompt, return_tensors="pt")
    output = t5_model.generate(
        inputs.input_ids,
        max_length=300,
        num_return_sequences=1,
        do_sample=True,
        early_stopping=False,
        num_beams=2,
        top_p=0.9,
        temperature=1.0
    )
    response = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n debug - Generated Text:", response)
    return response


#IMAGE-TO-TEXT HELPER(BLIP-2)

def process_image(image_path):
    """Loads an image for BLIP-2 processing."""
    return Image.open(image_path).convert("RGB")


def image_to_text_prompt(image_path, mode="zero-shot", few_shot_text_examples=None):
    """Generates a caption for an image using BLIP-2.

      Modes:
      - 'zero-shot': Uses only the image without extra context.
      - 'few-shot': Includes text-based few-shot examples to guide the model.
      """
    """Generates a caption for an image using BLIP-2.

       Modes:
       - 'zero-shot': Uses only the image without extra context.
       - 'few-shot': Includes text-based few-shot examples to guide the model.
       """
    image = process_image(image_path)

    if mode == "few-shot" and few_shot_text_examples:
        few_shot_prompt = "\n".join(f"Example {i + 1}: {desc}" for i, desc in enumerate(few_shot_text_examples))
        prompt = f"{few_shot_prompt}\n---\nQuestion: What is happening in this new image? Answer:"
    else:
        prompt = "Question: What is happening in this image? Answer:"
    #process input for BLIP-2
    inputs = blip2_processor(images=image, text=prompt, return_tensors="pt")


    output = blip2_model.generate(
        **inputs,
        max_new_tokens=25,  #shorter for concise output
        num_return_sequences=1,
        do_sample=True,  #removes randomness
        temperature=0.2,  #more structured response
        num_beams=5,  #helps with coherence
        top_p=0.6,  #prevents irrelevant sentence completions
        repetition_penalty=1.2  #reduces weird repetitive phrasing)
    )

    #decode return 
    caption = blip2_processor.tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n debug - Generated Image Caption:", caption)
    return caption

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score


def evaluate_length(text):
    """Return the number of words in a text."""
    return len(text.split())


def evaluate_results(text_output, image_caption):
    """Evaluate length and semantic similarity of text outputs."""
    length_text = evaluate_length(text_output)
    length_image = evaluate_length(image_caption)
    similarity = evaluate_similarity(text_output, image_caption)
    return {
        "Text Output Length": length_text,
        "Image Caption Length": length_image,
        "Semantic Similarity": similarity
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    #zero shot
    zero_shot_results = []
    for prompt in zero_shot_text_prompts:
        text_output_zero_shot = text_to_text_prompt(prompt)
        eval_zero_shot = evaluate_results(text_output_zero_shot, prompt)
        zero_shot_results.append((text_output_zero_shot, eval_zero_shot))

        print(f"\nZero-Shot Prompt: {prompt}")
        print("Text-to-Text Output (Zero-Shot):\n", text_output_zero_shot)
        print("Evaluation Metrics (Zero-Shot):\n", eval_zero_shot)

    #few shot
    few_shot_results = []
    for prompt in few_shot_text_prompts:
        text_output_few_shot = text_to_text_prompt(prompt)
        eval_few_shot = evaluate_results(text_output_few_shot, prompt)
        few_shot_results.append((text_output_few_shot, eval_few_shot))

        print("\nFew-Shot Prompt:")
        print("Text-to-Text Output (Few-Shot):\n", text_output_few_shot)
        print("Evaluation Metrics (Few-Shot):\n", eval_few_shot)

    for i, image_path in enumerate(image_paths):
        print(f"\n Processing Image {i + 1}: {image_path}")

        #zero-shot image captioning
        image_caption_zero_shot = image_to_text_prompt(image_path, mode="zero-shot")
        print("\n Zero-Shot Image Caption:")
        print(image_caption_zero_shot)

        #few shot image captioning
        image_caption_few_shot = image_to_text_prompt(image_path, mode="few-shot",
                                                      few_shot_text_examples=few_shot_text_examples)
        print("\n Few-Shot Image Caption:")
        print(image_caption_few_shot)

        print("-" * 80)
