import torch
from transformers import GPT2Tokenizer
from model import GPT3

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the trained model
model = GPT3(n_vocab=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("model.pth"))

# Set the model to evaluation mode
model.eval()

# Set the prompt text
prompt = "The quick brown fox"

# Set the number of tokens to generate
length = 50

# Tokenize the prompt text
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
output = model.generate(
    input_ids=input_ids,
    max_length=length + len(input_ids[0]),
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    num_return_sequences=1,
)

# Decode the generated tokens back to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
