import openai


async def get_chat_response(
    system_message: str, user_request: str, seed: int = None, temperature: float = 0.7
):
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request},
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5",
            messages=messages,
            seed=seed,
            max_tokens=200,
            temperature=temperature,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None