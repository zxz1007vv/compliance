from openai import OpenAI

client = OpenAI(
    api_key="sk-R7d1k4cT5xeHP1wnXs9bynYfj8cFmyXlJXlGmPABugDqFG3O",
    base_url="http://10.1.6.27/v1"
)

response = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
