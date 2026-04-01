from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_memory import ConversationMemoryEngine, MemoryConfig


def main() -> None:
    engine = ConversationMemoryEngine(
        config=MemoryConfig(
            window_size=6,
            stride=3,
            top_k=3,
            prefer_hf_embeddings=False,
        )
    )

    conversation = [
        ("user", "I prefer dark mode for coding UIs."),
        ("assistant", "Noted, dark mode preference recorded."),
        ("user", "My backend stack is FastAPI + Postgres."),
        ("assistant", "Great, FastAPI and Postgres."),
        ("user", "I need low-latency retrieval for memory."),
        ("assistant", "We should optimize retrieval and cache paths."),
        ("user", "I usually deploy on a single A100 instance."),
        ("assistant", "Understood: single GPU deployment constraints."),
        ("user", "What UI settings did I mention earlier?"),
    ]

    for role, text in conversation:
        engine.add_turn(role, text)

    engine.rebuild_memory()

    query = "What does the user prefer for UI theme?"
    hits = engine.retrieve(query)

    print(f"Query: {query}\n")
    for index, unit in enumerate(hits, start=1):
        print(f"[{index}] turns={unit.turn_start}-{unit.turn_end}")
        print(unit.text)
        print("-" * 60)


if __name__ == "__main__":
    main()
