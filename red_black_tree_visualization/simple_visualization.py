"""
Author: Sai Sundeep Rayidi
Date: 7/28/2024

Description:
[Description of what the file does, its purpose, etc.]

Additional Notes:
[Any additional notes or information you want to include.]

License: 
MIT License

Copyright (c) 2024 Sai Sundeep Rayidi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contact:
[Optional: How to reach you for questions or collaboration.]

"""

from red_black_tree_node_implementation import RedBlackTree


import plotly.graph_objects as go
import networkx as nx


def plot_graph(G):
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='blue')
    ))

    # Add nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        marker=dict(size=25, color=node_colors, line=dict(width=2, color='black')),
        textposition='top center'
    ))

    fig.update_layout(
        title="Red-Black Tree Visualization",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    fig.show()


def run_test_client():
    rb_tree = RedBlackTree()
    rb_tree.insert(2)
    rb_tree.insert(3)
    rb_tree.insert(4)
    rb_tree.insert(7)
    rb_tree.insert(11)
    rb_tree.insert(9)
    rb_tree.insert(6)
    rb_tree.insert(14)
    rb_tree.insert(18)
    rb_tree.insert(20)
    rb_tree.insert(23)
    rb_tree.insert(8)
    graph = rb_tree.tree_to_networkx_graph(rb_tree.root)
    plot_graph(graph)


if __name__ == "__main__":
    run_test_client()