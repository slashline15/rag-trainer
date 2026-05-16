from telegram import Update
from telegram.ext import CallbackContext
from loguru import logger

from rag_sqlite.tg import flows
from rag_sqlite.tg.handlers.corrections import handle_correction_input
from rag_sqlite.tg.handlers.text import process_text
from rag_sqlite.tg.handlers.media import process_audio, process_photo, process_document
from rag_sqlite.tg.handlers.callbacks import handle_callback as _handle_callback
from rag_sqlite.tg.helpers import send_typing_continuously, send_status_message, delete_message


async def handle_message(update: Update, context: CallbackContext) -> None:
    """Roteia mensagens de texto, áudio, imagem e documento."""

    pending = flows.get_pending_correction(context)
    if pending:
        await handle_correction_input(update, context, pending)
        return

    user = update.effective_user
    if user is None:
        return
    user_id = user.id
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    typing_task = await send_typing_continuously(update, context)
    status_msg_id = await send_status_message(update, context, "⚡ Processando sua mensagem...")

    try:
        if update.message and update.message.text:
            await process_text(update, context, user_id, chat_id, status_msg_id)
        elif update.message and (update.message.voice or update.message.audio):
            await process_audio(update, context, user_id, chat_id, status_msg_id)
        elif update.message and update.message.photo:
            await process_photo(update, context, user_id, chat_id, status_msg_id)
        elif update.message and update.message.document:
            await process_document(update, context, user_id, chat_id, status_msg_id)
        else:
            await update.effective_chat.send_message("Tipo de mensagem não suportado ainda.")
    except Exception as e:
        logger.exception(f"Erro processando mensagem: {e}")
        await update.effective_chat.send_message(
            "❌ Ocorreu um erro ao processar sua mensagem. Tente novamente."
        )
    finally:
        typing_task.cancel()
        try:
            await delete_message(update, context, status_msg_id)
        except Exception:
            pass


async def handle_callback(update: Update, context: CallbackContext) -> None:
    await _handle_callback(update, context)
