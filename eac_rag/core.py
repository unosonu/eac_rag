import os
import re
import pickle
import networkx as nx
import faiss
import numpy as np
import fitz  # PyMuPDF
import ollama
from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER
from tqdm import tqdm

class EACRAG:
    def __init__(self, embed_model='all-MiniLM-L6-v2', ner_model="Ihor/gliner-biomed-small-v1.0"):
        print("🚀 Initializing EAC-RAG Engines...")
        self.embedder = SentenceTransformer(embed_model)
        self.ner = GLiNER.from_pretrained(ner_model)
        self.G = nx.Graph()
        self.chunks = []
        self.index = None
        self.ensemble_index = None

    def avt_segmentation(self, text, k_sentences=5):
        """Adaptive Value Thresholding to find semantic valleys."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s) > 10]
        if len(sentences) <= k_sentences: return [text]
        
        windows = [" ".join(sentences[i:i+k_sentences]) for i in range(len(sentences)-k_sentences+1)]
        window_embs = self.embedder.encode(windows, convert_to_tensor=True)
        
        sims = [util.cos_sim(window_embs[i], window_embs[i+1]).item() for i in range(len(window_embs)-1)]
        tau = np.mean(sims) - (1.2 * np.std(sims))
        
        chunks, current = [], []
        for i, sim in enumerate(sims):
            current.append(sentences[i])
            if sim < tau:
                chunks.append(" ".join(current))
                current = []
        current.extend(sentences[len(sims):])
        if current: chunks.append(" ".join(current))
        return chunks

    def ingest_pdfs(self, folder_path="pdfs"):
        """Extracts text from PDFs and runs the EAC pipeline."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return "Folder created. Add PDFs."

        raw_data = []
        for fn in os.listdir(folder_path):
            if fn.endswith(".pdf"):
                doc = fitz.open(os.path.join(folder_path, fn))
                for i, page in enumerate(doc):
                    raw_data.append({"text": page.get_text(), "source": f"{fn}_p{i}"})

        print(f"📦 Processing {len(raw_data)} pages...")
        for entry in tqdm(raw_data):
            segments = self.avt_segmentation(entry['text'])
            for seg in segments:
                entities = self.ner.predict_entities(seg, ["Drug", "Disease"], threshold=0.55)
                hubs = {e['text'].title() for e in entities}
                
                c_id = f"chunk_{len(self.chunks)}"
                self.chunks.append({"text": seg, "source": entry['source'], "hubs": list(hubs)})
                
                for h in hubs:
                    self.G.add_edge(h, c_id)

        self._apply_weights_and_build_indices()
        return "Ingestion complete."

    def _apply_weights_and_build_indices(self):
        """Concept 1: Table Trap Weighting & FAISS setup."""
        # Weighted Graph
        for u, v in self.G.edges():
            # Inversely proportional to degree (specificity)
            deg = self.G.degree(v) if v.startswith('chunk_') else self.G.degree(u)
            self.G[u][v]['weight'] = 1.0 / (deg + 1)

        # Vector Index
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedder.encode(texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

        # Ensemble Index
        ens_texts = [" ".join(c['hubs']) for c in self.chunks]
        ens_embs = self.embedder.encode(ens_texts)
        self.ensemble_index = faiss.IndexFlatIP(ens_embs.shape[1])
        faiss.normalize_L2(ens_embs.astype('float32'))
        self.ensemble_index.add(ens_embs.astype('float32'))

    def query(self, question, model="llama3.2:latest"):
        """Multi-hop retrieval funnel."""
        ents = [e['text'].title() for e in self.ner.predict_entities(question, ["Drug"], threshold=0.55)]
        context_chunks = set()

        # 1. Graph Traversal (Prioritizing specific edges)
        for ent in ents:
            if ent in self.G:
                neighbors = sorted(self.G.neighbors(ent), 
                                  key=lambda x: self.G[ent][x].get('weight', 0), 
                                  reverse=True)
                for n in neighbors[:3]:
                    idx = int(n.split('_')[-1])
                    context_chunks.add(self.chunks[idx]['text'])

        # 2. Semantic Fallback
        if len(context_chunks) < 2:
            q_vec = self.embedder.encode([question])
            _, I = self.index.search(q_vec.astype('float32'), 3)
            for i in I[0]: context_chunks.add(self.chunks[i]['text'])

        context_str = "\n---\n".join(list(context_chunks))
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
        return ollama.generate(model=model, prompt=prompt)['response']

    def save_state(self, filepath="eac_state.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'G': self.G, 'index': self.index, 'ens': self.ensemble_index}, f)

    def load_state(self, filepath="eac_state.pkl"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.chunks, self.G, self.index, self.ensemble_index = data['chunks'], data['G'], data['index'], data['ens']

    def plot_graph(self, filename="my_graph.png"):
        from .utils import visualize_medical_graph
        visualize_medical_graph(self.G, output_path=filename)
