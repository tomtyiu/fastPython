# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local references
AutoTok = AutoTokenizer
AutoModel = AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTok.from_pretrained("EpistemeAI/metatune-gpt20b-R1.2")
model = AutoModel.from_pretrained("EpistemeAI/metatune-gpt20b-R1.2")
device = model.device

# Prepare messages
messages = [{"role": "user", "content": "Who are you?"}]

# Local references to methods
apply_chat_template = tokenizer.apply_chat_template
decode = tokenizer.decode
generate = model.generate

# Tokenize input with local variables
inputs = apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}  # move tensors to device

# Generate output
outputs = generate(**inputs, max_new_tokens=40)

# Decode only the new tokens
input_len = inputs["input_ids"].shape[-1]
print(decode(outputs[0][input_len:]))
