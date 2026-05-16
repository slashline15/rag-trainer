from telegram.ext import CallbackContext
from loguru import logger

from rag_sqlite.core.database import get_db
from rag_sqlite.tg import flows


async def save_rating(
    context: CallbackContext,
    resp_msg_id: str,
    rating: int,
    correction_type: str | None = None,
    correction_text: str | None = None,
) -> None:
    """Persiste avaliação no banco."""
    data = flows.get_registered_response(context, int(resp_msg_id))
    if not data:
        logger.warning(f"Resposta {resp_msg_id} não registrada para avaliação")
        return

    db = get_db()
    db.insert_rating({
        "user_id": data.get("user_id"),
        "message_id": data.get("message_id"),
        "response_id": resp_msg_id,
        "rating": rating,
        "correction_type": correction_type,
        "correction_text": correction_text,
        "response_time_ms": data.get("response_time_ms"),
        "tokens_input": data.get("tokens_input"),
        "tokens_output": data.get("tokens_output"),
        "model_used": data.get("model_used"),
        "self_consistency_score": data.get("self_consistency_score"),
        "retrieval_score": data.get("retrieval_score"),
        "final_confidence": data.get("final_confidence"),
    })
    logger.info(f"Rating saved: {rating} for response {resp_msg_id}")
