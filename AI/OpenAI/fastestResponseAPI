from openai import OpenAI
client = OpenAI()
create = client.responses.create

#Must declare model and input variables to work
resp = create(model="gpt-5-nano", input="Write a one-sentence bedtime story about a unicorn.")
print(resp.output_text)

# time_taken = timeit.timeit(my_code, number=2)
#Total time taken: 8.661138200000096
