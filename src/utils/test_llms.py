import openai
from src.settings import settings

client = openai.OpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

model_ids = [
    settings.llm_model,
]

for model_id in model_ids:
    print("-" * 20)
    print(f"Model: {model_id}")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "hi, what is your name"}],
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
