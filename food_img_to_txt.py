from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

image = Image.open("food.jpg").convert("RGB")

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=50)

description = processor.decode(out[0], skip_special_tokens=True)
print(description)


# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.


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
