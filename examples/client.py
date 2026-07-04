"""Call Proton 1 through the gateway with the OpenAI SDK — zero code change but
base_url and api_key. Demonstrates that Proton 1 is a drop-in OpenAI-style API.

    pip install openai
    python3 examples/client.py
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",   # the Node/Express gateway
    api_key="sk-proton-demo-key",
)

resp = client.chat.completions.create(
    model="proton-1",
    messages=[{"role": "user", "content": "Write a TypeScript function to add two numbers."}],
)
print(resp.choices[0].message.content)
