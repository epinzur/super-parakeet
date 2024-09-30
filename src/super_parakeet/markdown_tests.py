import json
from markdownify import MarkdownConverter
import mistune
from mistune.plugins import url

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from markdown_it.token import Token


import re
from typing import List
from markdown import markdown
from bs4 import BeautifulSoup

from typing import List, Optional


with open("datasets/crag/task_1/first_html_document.json") as f:
    html = json.load(f)["page_result"]

# with open("datasets/crag/task_1/first_html_document.html", "w") as f:
#     f.write(html)

markdown_converter = MarkdownConverter(heading_style="atx")

markdown_text = markdown_converter.convert(html=html)

with open("datasets/crag/task_1/first_html_document.md", "w") as f:
    f.write(markdown_text)

# # Initialize the markdown parser with the URL plugin
# markdown = mistune.create_markdown(plugins=[url])

# # Parse the markdown text and get all URLs
# links = markdown.renderer.urls(markdown_text)

# print(links)

# Initialize the parser
md = MarkdownIt()
ast = SyntaxTreeNode(md.parse(markdown_text))

# Extract links using the tree walker
markdown_it_links = list(set([node.attrs['href'] for node in ast.walk(include_self=False) if node.type == 'link']))


def extract_links_from_markdown(markdown_text: str) -> List[str]:
    # Convert Markdown to HTML
    html = markdown(markdown_text)

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract links from anchor tags
    links = [a['href'] for a in soup.find_all('a', href=True)]

    # Regex patterns for additional Markdown links (e.g., images, references)
    markdown_link_pattern = re.compile(r'\[.*?\]\(([^"\)]+)(?:\s*".*?")?\)')
    reference_link_pattern = re.compile(r'\[.*?\]:\s*(\S+)(?:\s*".*?")?')

    # Find all markdown-style links and references
    links.extend(markdown_link_pattern.findall(markdown_text))
    links.extend(reference_link_pattern.findall(markdown_text))

    # Remove duplicates by converting to a set, then back to a list
    return list(set([l.strip() for l in links]))

re_links = extract_links_from_markdown(markdown_text=markdown_text)

missing_links = []

for re_link in re_links:
    if re_link not in markdown_it_links:
        missing_links.append(re_link)
    if "kGLyX" in re_link:
        print(re_link)

print(len(markdown_it_links))

for markdown_it_link in markdown_it_links:
    if markdown_it_link not in re_links:
        missing_links.append(markdown_it_link)
    if "kGLyX" in markdown_it_link:
        print(markdown_it_link)

# print(missing_links)

print(len(missing_links))

# node_types = list(set([node.type for node in ast.walk(include_self=True)]))
# print(node_types)

# for node in ast.walk(include_self=True):
#     if node.type == "heading":
#         print(f"tag: {node.tag} children[0].content: {node.children[0].content} node.parent.type: {node.parent.type}")

class Chunk:
    def __init__(self, parent: Optional["Chunk"], markdown: str, name: str, level: int):
        self.parent: Optional["Chunk"] = parent
        self.children: List["Chunk"] = []
        self.previous_sibling: Optional["Chunk"] = None
        self.next_sibling: Optional["Chunk"] = None
        self.markdown: str = markdown
        self.name: str = name
        self.level: int = level


# def build_tree(node:SyntaxTreeNode, current_level: int) -> NodeType:
#     new_node = NodeType()
#     new_node.children = []
#     new_node.previous_sibling = None
#     new_node.next_sibling = None
#     new_node.inner_markdown = node.to_tokens()[0].

#     for node in ast.walk(include_self=False):
#         if node.type == "heading":
#             level = int(node.tag[1])

#             new_node = NodeType()
#             new_node.parent =

def parse_markdown_to_chunks(markdown_text: str) -> Chunk:
    md = MarkdownIt()
    tokens = md.parse(markdown_text)

    root = Chunk(parent=None, markdown="", name="", level=0)
    current_chunk = root
    chunk_stack = [root]

    for token in tokens:
        if token.type == 'heading_open':
            # Determine the level of the heading
            level = int(token.tag[1])
            # Get the heading content (name)
            heading_token = tokens[tokens.index(token) + 1]
            name = heading_token.content

            # Create a new chunk
            new_chunk = Chunk(parent=None, markdown="", name=name, level=level)

            # Find the correct parent based on level
            while chunk_stack and chunk_stack[-1].level >= level:
                chunk_stack.pop()

            parent_chunk = chunk_stack[-1]
            new_chunk.parent = parent_chunk
            parent_chunk.children.append(new_chunk)

            # Handle siblings
            if parent_chunk.children:
                if len(parent_chunk.children) > 1:
                    new_chunk.previous_sibling = parent_chunk.children[-2]
                    parent_chunk.children[-2].next_sibling = new_chunk

            # Push the new chunk onto the stack
            chunk_stack.append(new_chunk)
            current_chunk = new_chunk

        elif token.type == 'heading_close':
            # Do nothing on heading close
            continue

        else:
            # Add content to the current chunk
            current_chunk.markdown += token.content + '\n'

    return root



# print(chunk.markdown)


# Example usage
markdown_content = """
This is text before the 1st chapter

# Chapter 1

This is the first chapter.

## Section 1.1

This is the first section.

## Section 1.2

This is the second section.

# Chapter 2

This is the second chapter.

## Section 2.1

This is the first section of the second chapter.
"""

root_chunk = parse_markdown_to_chunks(markdown_content)

# Helper function to display the tree
def print_chunk_tree(chunk: Chunk, indent: int = 0):
    print(' ' * indent + f"Level {chunk.level}: {chunk.name}")
    print(' ' * indent + f"Content:\n{chunk.markdown}")
    for child in chunk.children:
        print_chunk_tree(child, indent + 2)

# print_chunk_tree(root_chunk)

chunk = parse_markdown_to_chunks(markdown_text=markdown_text)

print_chunk_tree(chunk=chunk)