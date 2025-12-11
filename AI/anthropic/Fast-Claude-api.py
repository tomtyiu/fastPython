import anthropic

# Create the client once; reuse it if this code runs repeatedly
client = anthropic.Anthropic()
create = client.messages.create  # local binding to reduce attribute lookups

# Prebuild the static user message so it isn't reconstructed each call
USER_MSG = [
    {
        "role": "user",
        "content": (
            "What should I search for to find the latest developments "
            "in renewable energy?"
        )
    }
]

# Faster call
msg = create(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    messages=USER_MSG
)

print(msg.content[0].text)
