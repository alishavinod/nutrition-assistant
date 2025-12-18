from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import pipeline

# Vision model for image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-large")

from PIL import Image
img = Image.open("food.jpg")

inputs = processor(images=img, return_tensors="pt")
out = model.generate(**inputs)
description = processor.decode(out[0], skip_special_tokens=True)

print("Description:", description)

# Free LLM for recipe generation
recipe_prompt = f"""
You are a chef. Given: {description}
Generate a recipe with:
• dish
• ingredients
• steps
"""

# Use a free model like “tiiuae/falcon-7b-instruct”
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
llm = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")

input_ids = tokenizer(recipe_prompt, return_tensors="pt").input_ids
output = llm.generate(input_ids, max_new_tokens=300)
recipe = tokenizer.decode(output[0], skip_special_tokens=True)

print(recipe)
