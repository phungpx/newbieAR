from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table


def display_rag_results(
    console: Console,
    query: str = None,
    contexts: list[str] = None,
    citations: list[str] = None,
    response: str = None,
):
    if query:
        console.print(
            f"\n[bold magenta]Query:[/bold magenta] [italic]{query}[/italic]\n"
        )

    if contexts:
        table = Table(
            title="Retrieved Facts from Knowledge Graph", show_lines=True, expand=True
        )
        table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")

        for i, context in enumerate(contexts):
            table.add_row(f"Context {i + 1}", context)

        console.print(table)

    if response:
        console.print(
            Panel(
                Markdown(response),
                title="[bold green]LLM Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    if citations:
        console.print(
            Panel(
                Markdown(f"- {'\n- '.join(citations)}"),
                title="[bold magenta]Citations[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
