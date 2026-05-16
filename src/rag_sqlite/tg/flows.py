from typing import Any
from telegram import InlineKeyboardMarkup, InlineKeyboardButton


PENDING_CORRECTION = "pending_correction"
RESPONSE_REGISTRY = "response_registry"


RATING_EMOJIS = {
    1: "😡",
    2: "😠",
    3: "😐",
    4: "🙂",
    5: "😁",
}


def build_rating_buttons(message_id: str, response_id: str) -> InlineKeyboardMarkup:
    """Botões inline 1-5 para avaliação. Layout: 3 em cima, 2 embaixo."""
    row1 = [
        InlineKeyboardButton(RATING_EMOJIS[1], callback_data=f"rate:1:{message_id}:{response_id}"),
        InlineKeyboardButton(RATING_EMOJIS[2], callback_data=f"rate:2:{message_id}:{response_id}"),
        InlineKeyboardButton(RATING_EMOJIS[3], callback_data=f"rate:3:{message_id}:{response_id}"),
    ]
    row2 = [
        InlineKeyboardButton(RATING_EMOJIS[4], callback_data=f"rate:4:{message_id}:{response_id}"),
        InlineKeyboardButton(RATING_EMOJIS[5], callback_data=f"rate:5:{message_id}:{response_id}"),
    ]
    return InlineKeyboardMarkup([row1, row2])


def build_correction_prompt_buttons(message_id: str, response_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Sim", callback_data=f"corr:yes:{message_id}:{response_id}"),
            InlineKeyboardButton("Não", callback_data=f"corr:no:{message_id}:{response_id}"),
        ]
    ])


def build_correction_type_buttons(message_id: str, response_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Ajustar resposta", callback_data=f"corr:ajuste:{message_id}:{response_id}"),
            InlineKeyboardButton("Adicionar comentário", callback_data=f"corr:comentario:{message_id}:{response_id}"),
        ]
    ])


def set_pending_correction(
    context: Any, message_id: str, response_id: str, rating: int, corr_type: str
) -> None:
    context.user_data[PENDING_CORRECTION] = {
        "message_id": message_id,
        "response_id": response_id,
        "rating": rating,
        "type": corr_type,
    }


def get_pending_correction(context: Any) -> dict | None:
    return context.user_data.get(PENDING_CORRECTION)


def clear_pending_correction(context: Any) -> None:
    context.user_data.pop(PENDING_CORRECTION, None)


def register_response(context: Any, response_msg_id: int, data: dict) -> None:
    if RESPONSE_REGISTRY not in context.user_data:
        context.user_data[RESPONSE_REGISTRY] = {}
    context.user_data[RESPONSE_REGISTRY][str(response_msg_id)] = data


def get_registered_response(context: Any, response_msg_id: int) -> dict | None:
    reg = context.user_data.get(RESPONSE_REGISTRY, {})
    return reg.get(str(response_msg_id))
