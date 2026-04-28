import sys

from rich.console import Console
from rich.panel import Panel

from app.services.rag_service import build_assistant

console = Console()


def main() -> None:
    if len(sys.argv) < 2:
        console.print('Usage: python -m app.scripts.ask "Your question here"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    assistant = build_assistant()
    answer = assistant.answer(question)

    console.print(Panel(question, title="Question"))
    console.print(Panel(answer, title="Assistant Answer"))


if __name__ == "__main__":
    main()
