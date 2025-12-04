from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained('..//clipmodel')
tokenizer = AutoTokenizer.from_pretrained("..//clipmodel")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
text_embeds = outputs.text_embeds
print(text_embeds.shape)