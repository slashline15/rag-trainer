from telegram import Update
from telegram.ext import CallbackContext
from loguru import logger

from rag_sqlite.tg import flows
from rag_sqlite.tg.helpers import edit_or_delete_message
from rag_sqlite.tg.handlers.ratings import save_rating


async def handle_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split(":")
    action = parts[0]

    if action == "rate":
        await _handle_rate(update, context, parts)
    elif action == "corr":
        await _handle_correction(update, context, parts)
    else:
        logger.warning(f"Callback desconhecido: {data}")


async def _handle_rate(update: Update, context: CallbackContext, parts: list) -> None:
    if len(parts) < 4:
        return
    rating = int(parts[1])
    orig_msg_id = parts[2]
    resp_msg_id = parts[3]
    chat_id = update.effective_chat.id

    await edit_or_delete_message(update, context, int(resp_msg_id))

    emoji = flows.RATING_EMOJIS.get(rating, "⭐")

    if rating >= 4:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(resp_msg_id),
            text=f"{update.callback_query.message.text}\n\n✅ Você avaliou com {emoji}",
        )
        await save_rating(context, resp_msg_id, rating)
        return

    if rating == 3:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(resp_msg_id),
            text=f"{update.callback_query.message.text}\n\n{emoji} Avaliação neutra registrada.",
        )
        await save_rating(context, resp_msg_id, rating)
        return

    # 1 ou 2: perguntar se deseja corrigir
    await context.bot.edit_message_text(
        chat_id=chat_id,
        message_id=int(resp_msg_id),
        text=f"{update.callback_query.message.text}\n\n{emoji} Você avaliou negativamente. Deseja corrigir?",
        reply_markup=flows.build_correction_prompt_buttons(orig_msg_id, resp_msg_id),
    )


async def _handle_correction(update: Update, context: CallbackContext, parts: list) -> None:
    if len(parts) < 4:
        return
    sub_action = parts[1]
    orig_msg_id = parts[2]
    resp_msg_id = parts[3]
    chat_id = update.effective_chat.id

    if sub_action == "no":
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(resp_msg_id),
            text=f"{update.callback_query.message.text}\n\nObrigado pelo feedback!",
            reply_markup=None,
        )
        await save_rating(context, resp_msg_id, 1)
        return

    if sub_action in ("ajuste", "comentario"):
        label = (
            "Envie a resposta corrigida:"
            if sub_action == "ajuste"
            else "Envie seu comentário:"
        )
        flows.set_pending_correction(
            context, orig_msg_id, resp_msg_id, rating=1, corr_type=sub_action
        )
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(resp_msg_id),
            text=f"{update.callback_query.message.text}\n\n📝 {label}",
            reply_markup=None,
        )
        return

    if sub_action == "yes":
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(resp_msg_id),
            text=f"{update.callback_query.message.text}\n\nComo deseja corrigir?",
            reply_markup=flows.build_correction_type_buttons(orig_msg_id, resp_msg_id),
        )
        return
