"""DSL-to-platform-code compiler that parses .gui files and generates output markup."""

#!/usr/bin/env python
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import json
from classes.Node import Node


class Compiler:
    """Compiles a .gui DSL file into platform-specific code.

    Reads a DSL mapping JSON file that defines how each DSL token maps to a
    platform-specific markup template. Parses the input .gui file into a tree
    of Node objects, then renders the tree using the mapping.
    """

    def __init__(self, dsl_mapping_file_path):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

    def compile(self, input_file_path, output_file_path, rendering_function=None):
        """Parse a .gui file and write the rendered output to a file.

        Args:
            input_file_path: Path to the input .gui DSL file.
            output_file_path: Path where the rendered output will be written.
            rendering_function: Optional callable(key, value) -> value for
                post-processing rendered template values.
        """
        self.root = Node("body", None, self.content_holder)
        current_parent = self.root

        with open(input_file_path) as dsl_file:
            for token in dsl_file:
                token = token.replace(" ", "").replace("\n", "")

                if token.find(self.opening_tag) != -1:
                    token = token.replace(self.opening_tag, "")

                    element = Node(token, current_parent, self.content_holder)
                    current_parent.add_child(element)
                    current_parent = element
                elif token.find(self.closing_tag) != -1:
                    current_parent = current_parent.parent
                else:
                    tokens = token.split(",")
                    for t in tokens:
                        element = Node(t, current_parent, self.content_holder)
                        current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)
