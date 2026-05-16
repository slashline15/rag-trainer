from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from loguru import logger

from rag_sqlite.core.config import settings
from rag_sqlite.tg.handlers import handle_message, handle_callback


async def start_command(update: Update, context) -> None:
    await update.effective_chat.send_message(
        "👋 Olá! Sou seu assistente de conhecimento pessoal.\n\n"
        "Envie mensagens de texto, áudios, fotos ou documentos que eu organizo e "
        "respondo com base na sua base de conhecimento.\n\n"
        "Tudo é privado — seus dados nunca são misturados com os de outros usuários.\n\n"
        "Use #hashtags para categorizar informações."
    )


async def error_handler(update: Update, context) -> None:
    logger.error(f"Erro no handler: {context.error}", exc_info=context.error)
    if update and update.effective_chat:
        await update.effective_chat.send_message(
            "⚠️ Ocorreu um erro inesperado. Já registrei o problema. Tente novamente em instantes."
        )


def main() -> None:
    if not settings.telegram.bot_token:
        logger.error("TELEGRAM_BOT_TOKEN não configurado no .env")
        raise SystemExit(1)

    application = Application.builder().token(settings.telegram.bot_token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(
        filters.TEXT | filters.VOICE | filters.AUDIO | filters.PHOTO | filters.Document.ALL,
        handle_message,
    ))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_error_handler(error_handler)

    logger.info("Bot iniciado. Aguardando mensagens...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
