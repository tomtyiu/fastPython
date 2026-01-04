#To install , pip install -q -U google-genai
from google import genai

client = genai.Client()
#reduce or avoid multiple dot notation for speed up
create=client.models.generate_content
response = create(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)
