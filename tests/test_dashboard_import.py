import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    from rag_sqlite.dashboard.app import main
    from rag_sqlite.dashboard.pages import overview, tables, chunks, ratings, metrics
    print("All dashboard imports OK")


if __name__ == "__main__":
    test_imports()
