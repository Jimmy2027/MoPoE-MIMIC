import networkx as nx
from matplotlib import pyplot as plt
import pygraphviz as pgv  # pygraphviz should be available


def draw_text_word_workflow_graph():
    G = nx.DiGraph()
    G.add_edge('SentDataset', 'text_enc', label='[bs, 1024]')
    G.add_edge('text_enc', 'sampling', label='[bs, 640, 1]')
    G.add_edge('sampling', 'feature_gen', label='[10,64]')
    G.add_edge('feature_gen', 'text_gen', label='[bs,640,1]')
    G.add_edge('text_gen', 'text_dec', label='[bs,vs,1024]')
    G.add_edge('text_dec', 'lhood_text', label='[bs,1024,vs]')
    G.add_edge('lhood_text', 'output', label='OneHotCategorical')

    G.add_nodes_from(
        ['SentDataset', 'text_enc', 'sampling', 'text_gen', 'text_dec', 'lhood_text', 'output', 'feature_gen'])
    A = nx.drawing.nx_agraph.to_agraph(G)
    print('drawing grapt to utils/text_word_workflow.dot')
    A.draw('utils/text_word_workflow.dot', prog='dot')
    plt.show()


def draw_training_graph():
    G = nx.DiGraph()
    G.add_edge('Volume', 'Model')
    G.add_edge('Model', 'Mask', label='Prediction')
    G.add_edge('Mask', 'Error')
    G.add_edge('Template Mask', 'Error')
    G.add_edge('Error', 'Model', label='Update')

    A = nx.drawing.nx_agraph.to_agraph(G)
    one = A.add_subgraph(rank='same')
    one.add_node('Volume',
                 label='<Volume\n\n\n<BR /> <FONT POINT-SIZE="10">\n\n\n registered by the Generic workflow</FONT>>')
    one.add_node('Model')
    one.add_node('Mask', color='springgreen3')
    two = A.add_subgraph(rank='same')
    two.add_node('Template Mask', color='springgreen3')
    two.add_node('Error', color='red2')

    A.draw('training.dot', prog='dot')
    # nx.draw(A, with_labels=True)
    # plt.show()


def other_example():
    G = nx.Graph()
    G.add_edges_from(
        [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
         ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

    val_map = {'A': 1.0,
               'D': 0.5714285714285714,
               'H': 0.0}

    values = [val_map.get(node, 0.25) for node in G.nodes()]

    nx.draw(G, cmap=plt.get_cmap('jet'), node_color=values, with_labels=True)
    plt.show()

# draw_text_word_workflow_graph()
# draw_training_graph()
# other_example()
