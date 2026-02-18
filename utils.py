import matplotlib.pyplot as plt
import networkx as nx

def visualize_medical_graph(G, output_path="medical_graph.png", max_nodes=50):
    """Generates a visual representation of the Bipartite Graph."""
    plt.figure(figsize=(12, 8))
    
    # Subsample nodes so the plot isn't a mess
    nodes_to_draw = list(G.nodes())[:max_nodes]
    sub_G = G.subgraph(nodes_to_draw)
    
    pos = nx.spring_layout(sub_G, k=0.5)
    
    # Differentiate Hubs (Entities) and Satellites (Chunks)
    hubs = [n for n in sub_G.nodes() if not str(n).startswith('chunk_')]
    chunks = [n for n in sub_G.nodes() if str(n).startswith('chunk_')]
    
    nx.draw_networkx_nodes(sub_G, pos, nodelist=hubs, node_color='orange', node_size=800, label='Entities')
    nx.draw_networkx_nodes(sub_G, pos, nodelist=chunks, node_color='lightblue', node_size=300, label='Chunks')
    nx.draw_networkx_edges(sub_G, pos, alpha=0.3)
    nx.draw_networkx_labels(sub_G, pos, font_size=8)
    
    plt.title("EAC-RAG: Bipartite Clinical Knowledge Graph")
    plt.legend()
    plt.savefig(output_path)
    print(f"📈 Graph visualization saved to {output_path}")
    plt.close()
