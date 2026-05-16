import time
from typing import Dict, Any, Optional, List
from loguru import logger

from rag_sqlite.rag.intent import classify_intent
from rag_sqlite.rag.retrieval import HybridRetriever
from rag_sqlite.rag.prompt_builder import build_rag_prompt, build_ingest_prompt
from rag_sqlite.rag.generator import generate_response
from rag_sqlite.core.database import get_db
from rag_sqlite.core.config import settings


class RAGEngine:
    def __init__(self):
        self.retriever = HybridRetriever(top_k=10)

    async def process(self, user_id: int, message: str, context: str = None,
                      history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Fluxo completo de uma interação:
        1. Classificar intenção.
        2. Se ingest → confirmar.
        3. Se query/mixed → recuperar contexto híbrido, montar prompt, gerar resposta.
        4. Calcular métricas (tempo, tokens estimados).
        5. Retornar tudo para o Telegram bot montar a resposta + botões.
        """
        start_time = time.time()

        # Resolver contexto padrão se não especificado
        if context is None:
            context = settings.ollama.default_context or "local"

        # 1. Classificar intenção
        intent_result = await classify_intent(message, context)
        intent = intent_result["intent"]
        target_module = intent_result.get("target_module", "general")

        logger.info(f"Processing message from user {user_id}: intent={intent}, module={target_module}")

        # 2. Ingestão pura: confirmar
        if intent == "ingest":
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "intent": "ingest",
                "response": "Anotei! Posso ajudar com mais alguma coisa?",
                "target_module": target_module,
                "entities": intent_result.get("entities", {}),
                "response_time_ms": elapsed,
                "model_used": "",
                "retrieval_score": 0.0,
                "chunks_used": [],
            }

        # 3. Query / Mixed: recuperar contexto
        chunks = await self.retriever.retrieve(user_id, message, module=target_module if target_module != "general" else None)

        # Buscar dados estruturados se o módulo for específico
        structured_data = None
        if target_module == "finance":
            structured_data = self.retriever.retrieve_structured(
                user_id,
                "SELECT * FROM module_finance WHERE user_id = ? ORDER BY transaction_date DESC LIMIT 10",
                (user_id,),
            )

        # Recuperar avaliações recentes para RAG contrastivo
        ratings = self.retriever.get_recent_ratings_for_context(user_id, limit=5)

        # 4. Montar prompt (com histórico de conversa)
        prompt, chunk_aliases = build_rag_prompt(
            message, chunks,
            ratings_context=ratings,
            structured_data=structured_data,
            history=history,
        )

        # 5. Gerar resposta
        try:
            gen_result = await generate_response(prompt, context)
            response_text = gen_result["response"]
            model_used = gen_result["model_used"]
            sc_score = gen_result.get("self_consistency_score")
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            response_text = (
                "Não consegui gerar uma resposta no momento. "
                "O modelo de IA pode estar sem memória disponível ou offline. "
                "Tente novamente em instantes ou use um modelo menor."
            )
            model_used = ""
            sc_score = None

        # 6. Calcular retrieval score (sem expor confiança na resposta)
        retrieval_score = 0.0
        if chunks:
            retrieval_score = sum(c.get("combined_score", 0) for c in chunks) / len(chunks)

        elapsed = int((time.time() - start_time) * 1000)

        # Estimar tokens (aproximado: chars / 4)
        tokens_input = len(prompt) // 4
        tokens_output = len(response_text) // 4

        return {
            "intent": intent,
            "response": response_text,
            "target_module": target_module,
            "entities": intent_result.get("entities", {}),
            "response_time_ms": elapsed,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "model_used": model_used,
            "self_consistency_score": sc_score,
            "retrieval_score": round(retrieval_score, 4),
            "chunks_used": chunks,
            "chunk_aliases": chunk_aliases,
        }

    async def ingest_and_respond(self, user_id: int, message: str,
                                 source_id: str, context: str = "local") -> Dict[str, Any]:
        """
        Fluxo misto: usuário pode estar informando E perguntando ao mesmo tempo.
        A ingestão já ocorreu (source_id fornecido). Aqui apenas processamos a consulta
        e montamos a resposta final, possivelmente confirmando o que foi anotado.
        """
        result = await self.process(user_id, message, context)
        # Se mixed, podemos adicionar uma confirmação de ingestão no início da resposta
        if result["intent"] == "mixed":
            confirm = f"Anotei a informação. "
            result["response"] = confirm + result["response"]
        return result
