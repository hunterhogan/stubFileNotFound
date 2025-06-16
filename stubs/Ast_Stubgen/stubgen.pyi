import ast

HAVE_ORDERED_SET: bool

def generate_stub(source_file_path: str, output_file_path: str, text_only: bool = False) -> str | None:
    class StubGenerator(ast.NodeVisitor): ...

def generate_text_stub(source_file_path: str) -> str: ...
