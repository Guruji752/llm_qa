import asyncio
from huggingface_hub import InferenceClient
from app.core.config import settings

_client = InferenceClient(token=settings.hf_token)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    "If the answer is not in the context, say so."
)


def _call_hf(question: str, context: str) -> str:
    response = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_hf, question, context)
