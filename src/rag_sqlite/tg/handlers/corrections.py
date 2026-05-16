from telegram import Update
from telegram.ext import CallbackContext
from loguru import logger

from rag_sqlite.tg import flows
from rag_sqlite.tg.handlers.ratings import save_rating


async def handle_correction_input(update: Update, context: CallbackContext, pending: dict) -> None:
    """Processa a mensagem de texto enviada como correção/comentário."""
    text = update.message.text if update.message else ""
    corr_type = pending["type"]
    resp_msg_id = pending["response_id"]

    await save_rating(
        context, resp_msg_id, pending["rating"],
        correction_type=corr_type, correction_text=text
    )

    flows.clear_pending_correction(context)

    label = "Ajuste salvo" if corr_type == "ajuste" else "Comentário salvo"
    await update.effective_chat.send_message(f"✅ {label}! Obrigado pela contribuição.")
    logger.info(f"Correction saved for response {resp_msg_id}: type={corr_type}")
