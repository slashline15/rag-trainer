import tempfile
from pathlib import Path
from telegram import Update
from telegram.ext import CallbackContext
from loguru import logger

from rag_sqlite.pipeline.ingest import ingest_audio, ingest_image, ingest_document
from rag_sqlite.tg.handlers.text import process_rag_text
from rag_sqlite.tg.helpers import delete_message


async def process_audio(update: Update, context: CallbackContext, user_id: int,
                        chat_id: int, status_msg_id: int) -> None:
    voice = update.message.voice or update.message.audio
    if voice is None:
        logger.warning("process_audio called but no voice/audio in message")
        return
    file = await voice.get_file()
    logger.info(f"Audio file received: {file.file_id}, size={voice.file_size}")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        file_path = tmp.name
        logger.info(f"Audio downloaded to {file_path}, size={Path(file_path).stat().st_size} bytes")

    await context.bot.edit_message_text(
        chat_id=chat_id,
        message_id=status_msg_id,
        text="🎙️ Transcrevendo áudio...",
    )

    try:
        logger.info(f"Calling ingest_audio for user {user_id}...")
        result = await ingest_audio(user_id, Path(file_path).name, Path(file_path).read_bytes())
        logger.info(f"ingest_audio result keys: {list(result.keys())}")
        transcription = result.get("corrected_transcription", "")
        raw = result.get("raw_transcription", "")
        logger.info(f"Raw transcription: {raw[:200] if raw else 'EMPTY'}")
        logger.info(f"Corrected transcription: {transcription[:200] if transcription else 'EMPTY'}")
        if transcription:
            await update.effective_chat.send_message(
                f"📝 *Transcrição:*\n{transcription}",
                parse_mode="Markdown",
            )
            # Processar a transcrição como se fosse uma mensagem de texto
            await process_rag_text(user_id, chat_id, transcription, context)
        else:
            await update.effective_chat.send_message("❌ Não consegui transcrever o áudio.")
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        await update.effective_chat.send_message(f"❌ Erro ao processar áudio: {e}")
    finally:
        Path(file_path).unlink(missing_ok=True)


async def process_photo(update: Update, context: CallbackContext, user_id: int,
                        chat_id: int, status_msg_id: int) -> None:
    if not update.message.photo:
        return
    photo = update.message.photo[-1]
    file = await photo.get_file()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        file_path = tmp.name

    await context.bot.edit_message_text(
        chat_id=chat_id,
        message_id=status_msg_id,
        text="📸 Analisando imagem...",
    )

    try:
        result = await ingest_image(user_id, Path(file_path).name, Path(file_path).read_bytes())
        description = result.get("description", "")
        if description:
            await update.effective_chat.send_message(
                f"🖼️ *Descrição da imagem:*\n{description}",
                parse_mode="Markdown",
            )
            # Processar a descrição como texto para o RAG
            await process_rag_text(user_id, chat_id, description, context)
    finally:
        Path(file_path).unlink(missing_ok=True)


async def process_document(update: Update, context: CallbackContext, user_id: int,
                           chat_id: int, status_msg_id: int) -> None:
    doc = update.message.document
    if doc is None:
        return
    file = await doc.get_file()

    suffix = f".{doc.file_name.split('.')[-1]}" if '.' in doc.file_name else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        file_path = tmp.name

    await context.bot.edit_message_text(
        chat_id=chat_id,
        message_id=status_msg_id,
        text="📄 Processando documento...",
    )

    try:
        result = await ingest_document(user_id, doc.file_name, Path(file_path).read_bytes())
        chunks = result.get("chunks", [])
        await update.effective_chat.send_message(
            f"📑 Documento recebido! {len(chunks)} trechos indexados.",
        )
    finally:
        Path(file_path).unlink(missing_ok=True)
