from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Function that generates a response from a given context using GPT-2
def generate_response(context, model_name="gpt2"):
    
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Convert the input text (context) into tokens that the model can understand
    input_ids = tokenizer.encode(context, return_tensors='pt')  # 'pt' for PyTorch

    # Generate the response based on the input tokens
    # 'max_length=200' defines the maximum length of the generated sequence
    # 'num_return_sequences=1' specifies that we want only one generated sequence
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    
    # Decode the generated tokens into human-readable text
    # 'skip_special_tokens=True' removes special tokens like <|endoftext|>
    return tokenizer.decode(output[0], skip_special_tokens=True)
