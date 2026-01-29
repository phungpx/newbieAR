import os
import wikipediaapi


def save_wikipedia_article(url, output_format="md", folder="docs"):
    """
    Crawls a Wikipedia page and saves it as a .md or .txt file.
    """
    # Extract the page title from the URL
    # Example: https://en.wikipedia.org/wiki/Albert_Einstein -> Albert_Einstein
    page_name = url.split("/")[-1]

    # Initialize Wikipedia API with a descriptive User-Agent (Required by Wikimedia policy)
    wiki = wikipediaapi.Wikipedia(
        user_agent="WikiCrawlerBot/1.0 (contact: your@email.com)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    page = wiki.page(page_name)

    if not page.exists():
        print(f"Error: The page '{page_name}' does not exist.")
        return

    # Create output directory
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Clean the filename
    filename = f"{page_name.replace(' ', '_')}.{output_format}"
    file_path = os.path.join(folder, filename)

    if output_format == "md":
        # Create a basic Markdown structure
        content = f"# {page.title}\n\n"
        content += f"Source: {page.fullurl}\n\n"
        content += page.text
    else:
        # Plain text
        content = f"Title: {page.title}\n"
        content += f"URL: {page.fullurl}\n\n"
        content += page.text

    # Write to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully saved: {file_path}")


# --- Execution ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--output_format", type=str, default="md")
    parser.add_argument("--folder", type=str, default="RAG/data/wikipedia")
    args = parser.parse_args()

    save_wikipedia_article(args.url, args.output_format, args.folder)
