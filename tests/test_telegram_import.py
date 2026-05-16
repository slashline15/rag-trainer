import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    from rag_sqlite.tg.bot import main, start_command, error_handler
    from rag_sqlite.tg.handlers import handle_message, handle_callback
    from rag_sqlite.tg.flows import build_rating_buttons
    from rag_sqlite.tg.helpers import send_typing_continuously
    print("All tg imports OK")


if __name__ == "__main__":
    test_imports()
