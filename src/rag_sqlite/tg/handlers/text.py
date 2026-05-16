from typing import Optional

from telegram import Update
from telegram.ext import CallbackContext
from loguru import logger

from rag_sqlite.pipeline.ingest import ingest_text
from rag_sqlite.core.database import get_db
from rag_sqlite.rag.engine import RAGEngine
from rag_sqlite.tg import flows
from rag_sqlite.tg.helpers import send_long_message


engine = RAGEngine()


async def process_rag_text(user_id: int, chat_id: int, text: str,
                           context: CallbackContext,
                           original_message_id: Optional[int] = None) -> None:
    """Processa um texto qualquer via RAG (usado por texto, áudio transcrito, etc)."""
    if not text:
        return

    db = get_db()

    # Indexar o texto
    await ingest_text(user_id, text)

    # Buscar histórico recente
    history = db.get_recent_history(user_id, limit=10)

    # Processar via RAG Engine
    result = await engine.process(user_id, text, history=history)

    # Montar resposta final
    response_text = result["response"]

    # Tag curta para esta resposta do bot
    bot_tag = db.generate_next_tag(user_id)

    # Enviar resposta (dividida se necessário, sem markdown)
    sent_messages = await send_long_message(
        context.bot, chat_id, response_text
    )

    # A última mensagem recebe a tag e os botões
    last_sent = sent_messages[-1]

    # Adicionar tag na última parte
    tag_text = f"\n\n{bot_tag}"
    try:
        current = last_sent.text or ""
        if len(current + tag_text) <= 4096:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=last_sent.message_id,
                text=current + tag_text,
            )
    except Exception:
        pass  # Se falhar, envia separado
        await context.bot.send_message(chat_id=chat_id, text=bot_tag)

    # Registrar resposta do bot no histórico
    db.insert_message(
        msg_id=f"bot_{last_sent.message_id}",
        user_id=user_id,
        raw_content=result["response"],
        media_type="text",
        role="assistant",
        tag=bot_tag,
    )

    # Botões de avaliação na última mensagem
    msg_id = str(original_message_id) if original_message_id else str(last_sent.message_id)
    try:
        await context.bot.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=last_sent.message_id,
            reply_markup=flows.build_rating_buttons(msg_id, str(last_sent.message_id)),
        )
    except Exception as e:
        logger.debug(f"Não foi possível adicionar botões de avaliação: {e}")

    # Registrar resposta para avaliação futura
    flows.register_response(context, last_sent.message_id, {
        "user_id": user_id,
        "message_id": msg_id,
        "response_id": str(last_sent.message_id),
        "model_used": result.get("model_used"),
        "response_time_ms": result.get("response_time_ms"),
        "tokens_input": result.get("tokens_input"),
        "tokens_output": result.get("tokens_output"),
        "self_consistency_score": result.get("self_consistency_score"),
        "retrieval_score": result.get("retrieval_score"),
    })
    logger.info(f"Text processed for user {user_id}: {len(text)} chars, {len(sent_messages)} msg(s)")


async def process_text(update: Update, context: CallbackContext, user_id: int,
                       chat_id: int, status_msg_id: int) -> None:
    text = update.message.text if update.message else ""
    await process_rag_text(
        user_id, chat_id, text, context,
        original_message_id=update.message.message_id if update.message else None,
    )
