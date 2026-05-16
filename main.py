import asyncio
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Log em arquivo para o dashboard de logs
Path("logs").mkdir(exist_ok=True)
logger.add("logs/rag_sqlite.log", rotation="10 MB", retention=5, encoding="utf-8")


def check_dependencies() -> bool:
    """Verifica se dependências críticas estão instaladas."""
    required = [
        ("rag_sqlite", "pacote principal (pip install -e .)"),
        ("telegram", "python-telegram-bot"),
        ("httpx", "httpx"),
        ("pydantic", "pydantic"),
        ("sqlite_vec", "sqlite-vec"),
        ("numpy", "numpy"),
        ("dotenv", "python-dotenv"),
        ("fitz", "PyMuPDF"),
        ("PIL", "pillow"),
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("loguru", "loguru"),
    ]
    missing = []
    for mod, name in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(name)
    if missing:
        logger.error(f"Dependências faltando: {', '.join(missing)}")
        logger.error("Rode: pip install -r requirements.txt && pip install -e .")
        return False
    logger.info("Todas as dependências estão OK.")
    return True


def run_bot():
    return subprocess.Popen(
        [sys.executable, "-m", "rag_sqlite.tg.bot"],
        cwd=str(Path(__file__).parent),
    )


def run_dashboard():
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/rag_sqlite/dashboard/app.py"],
        cwd=str(Path(__file__).parent),
    )


def main():
    logger.info("=== RAG SQLite Launcher ===")
    if not check_dependencies():
        sys.exit(1)

    bot_proc = run_bot()
    dash_proc = run_dashboard()

    logger.info(f"Bot PID: {bot_proc.pid} | Dashboard PID: {dash_proc.pid}")
    logger.info("Pressione Ctrl+C para parar ambos.")

    try:
        while True:
            # Verificar se algum processo morreu
            bot_code = bot_proc.poll()
            dash_code = dash_proc.poll()

            if bot_code is not None:
                logger.warning(f"Bot saiu com código {bot_code}. Reiniciando...")
                bot_proc = run_bot()

            if dash_code is not None:
                logger.warning(f"Dashboard saiu com código {dash_code}. Reiniciando...")
                dash_proc = run_dashboard()

            # Monitorar .env para mudanças (a cada 3s)
            # Simplificado: só verifica se processos estão vivos
            import time
            time.sleep(3)
    except KeyboardInterrupt:
        logger.info("Encerrando...")
        bot_proc.terminate()
        dash_proc.terminate()
        bot_proc.wait(timeout=5)
        dash_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
