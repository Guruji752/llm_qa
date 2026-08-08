import asyncio
from huggingface_hub import InferenceClient
from langfuse import get_client, observe
from app.core.config import settings

_client = InferenceClient(token=settings.hf_token)

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about a resume. "
    "Use only the provided context to answer. "
    "You MAY reason and calculate from information present — for example, computing total experience from date ranges. "
    "Be concise and specific. Do not pad your answer. "
    "If the context does not contain enough information to answer, say exactly: 'The context does not contain enough information to answer this question.'"
)


def _call_hf(question: str, context: str) -> str:
    response = _client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content


@observe(name="generate_answer", as_type="generation")
async def generate_answer(question: str, context_chunks: list[str]) -> str:
    get_client().update_current_generation(model=settings.chat_model)
    context = "\n\n---\n\n".join(context_chunks)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_hf, question, context)
