from markdownify import MarkdownConverter, chomp
from urllib.parse import urldefrag
from langchain_core.graph_vectorstores.links import Link

from typing import Tuple, Set

class LinkExtractionConverter(MarkdownConverter):
    links = {}
    link_index = 0
    """
    Create a custom MarkdownConverter extracts html links during the conversion process
    """

    def get_link_placeholder(self, text: str) -> str:
        index = self.link_index
        self.link_index += 1
        return f"[[[{index}: {text}]]]"


    def convert_a(self, el, text:str, convert_as_inline):
        _, _, text = chomp(text)
        if not text:
            return ''
        text = text.replace("\n", "")
        href = el.get('href')

        link_placeholder = self.get_link_placeholder(text)

        self.links[link_placeholder] = (text, href)

        return link_placeholder

    def extract_links(self, chunk: str) -> Tuple[Set[Link], str]:
        links: Set[Link] = set()
        for link_placeholder, (text, url) in self.links.items():
            if link_placeholder in chunk:
                url = urldefrag(url).url
                if url:
                    links.add(Link.outgoing(kind="hyperlink", tag=url))
                chunk = chunk.replace(link_placeholder, text)

        return (links, chunk)

    def reset(self) -> None:
        self.links = {}
        self.link_index = 0
