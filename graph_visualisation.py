import os

import matplotlib.pyplot as plt
from graphviz import Digraph, Source

"""
Edited version accoridng to my Value implementation
"""
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            # I changed the _prev from the original to previous_values
            for child in v.previous_values:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.gradient), shape='record')
        if n.operator_symbol:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n.operator_symbol, label=n.operator_symbol)
            # and connect this node to it
            dot.edge(uid + n.operator_symbol, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.operator_symbol)

    import webbrowser
    # Render the DOT graph to an image file (e.g., "graph.png")
    image_file_path = "rendered_graphviz_imgs/latest_graph"
    dot.render(image_file_path, format='png', cleanup=True)
    webbrowser.open(f"{ROOT_DIR}/rendered_graphviz_imgs/latest_graph.png")
    return dot
