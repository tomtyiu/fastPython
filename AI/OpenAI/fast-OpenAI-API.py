import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def OpenAIAPI(prompt):
    create = client.responses.create
    responses = create(
        model="gpt-5.2",
        instructions="You are a coding assistant that talks like a pirate.",
        input=prompt,
    )

#print out response
# responses=response.output_text
# print(responses)
