from typing import Optional

from pyjsparser import parse
from graphviz import Digraph


class JSAbstractSyntaxTree:
    def __init__(self, js_source: str, file_name: Optional[str] = None):
        self.file_name = file_name
        self.js_source = js_source
        self.ast = parse(self.js_source)

    def visualize_ast(self):
        g = Digraph()
        for
