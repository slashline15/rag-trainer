import asyncio
import re
from loguru import logger
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CallbackContext

# Telegram max message length
MAX_MESSAGE_LENGTH = 4096


def markdown_to_html(text: str) -> str:
    """
    Converte markdown limitado para HTML que o Telegram aceita.
    Telegram HTML suporta: <b>, <i>, <code>, <pre>, <a>.
    """
    # Primeiro, escapar caracteres HTML no texto normal
    # Mas precisamos proteger os blocos de código antes
    code_blocks = []

    def _save_code_block(m):
        idx = len(code_blocks)
        lang = m.group(1) or ""
        code = m.group(2)
        # Escapar HTML dentro do código
        code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if lang:
            code_blocks.append(f'<pre><code class="language-{lang}">{code}</code></pre>')
        else:
            code_blocks.append(f"<pre>{code}</pre>")
        return f"__CODEBLOCK_{idx}__"

    # Proteger blocos de código ```lang\ncode```
    text = re.sub(r'```(\w*)\n?([\s\S]*?)```', _save_code_block, text)

    # Proteger inline code `code`
    inline_codes = []

    def _save_inline(m):
        idx = len(inline_codes)
        code = m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        inline_codes.append(f"<code>{code}</code>")
        return f"__INLINE_{idx}__"

    text = re.sub(r'`([^`]+)`', _save_inline, text)

    # Agora escapar HTML no texto restante
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Converter markdown para HTML
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text)
    # Headers -> bold
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Restaurar blocos de código
    for idx, block in enumerate(code_blocks):
        text = text.replace(f"__CODEBLOCK_{idx}__", block)
    for idx, inline in enumerate(inline_codes):
        text = text.replace(f"__INLINE_{idx}__", inline)

    return text.strip()


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Divide mensagem em partes que o Telegram aceita (max 4096 chars)."""
    if len(text) <= max_length:
        return [text]

    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break

        cut = max_length
        for sep in ['\n\n', '</pre>\n', '\n', '. ', ' ']:
            pos = text.rfind(sep, 0, max_length)
            if pos > max_length // 2:
                cut = pos + len(sep)
                break

        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()

    return parts


async def send_long_message(bot, chat_id: int, text: str, **kwargs) -> list:
    """Envia mensagem convertida para HTML, dividida se necessário."""
    html = markdown_to_html(text)
    parts = split_message(html)
    sent_messages = []
    for part in parts:
        try:
            msg = await bot.send_message(
                chat_id=chat_id, text=part, parse_mode="HTML", **kwargs,
            )
        except Exception:
            # Fallback: enviar sem parse_mode se HTML inválido
            msg = await bot.send_message(chat_id=chat_id, text=part, **kwargs)
        sent_messages.append(msg)
    return sent_messages


async def send_typing_continuously(update: Update, context: CallbackContext, interval: float = 4.0) -> asyncio.Task:
    """
    Mantém o indicador de 'digitando' ativo enquanto processa.
    Retorna uma Task que deve ser cancelada ao final.
    """
    async def _loop() -> None:
        while True:
            try:
                await update.effective_chat.send_action(ChatAction.TYPING)
            except Exception:
                break
            await asyncio.sleep(interval)
    return asyncio.create_task(_loop())


async def send_status_message(update: Update, context: CallbackContext, text: str) -> int:
    """Envia uma mensagem de status temporária. Retorna message_id."""
    msg = await update.effective_chat.send_message(text)
    return msg.message_id


async def delete_message(update: Update, context: CallbackContext, message_id: int) -> None:
    """Deleta uma mensagem específica. Silencia erros."""
    try:
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message_id)
    except Exception:
        pass


async def edit_or_delete_message(
    update: Update, context: CallbackContext, message_id: int, new_text: str | None = None
) -> None:
    """Edita mensagem removendo botões ou alterando texto."""
    try:
        chat_id = update.effective_chat.id
        if new_text is not None:
            if len(new_text) > MAX_MESSAGE_LENGTH:
                new_text = new_text[:MAX_MESSAGE_LENGTH - 3] + "..."
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=message_id,
                text=new_text, reply_markup=None,
            )
        else:
            await context.bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=message_id, reply_markup=None,
            )
    except Exception as e:
        logger.debug(f"edit_or_delete_message {message_id}: {e}")
