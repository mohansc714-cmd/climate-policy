import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch

print("[init] Starting Climate Policy RAG Assistant...")

# ─── 1. Load Dataset ────────────────────────────────────────────────────────
DATA_PATH = "climate_change_dataset.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"[init] Dataset loaded: {len(df)} rows.")

# ─── 2. Data Preprocessing ──────────────────────────────────────────────────
def row_to_text(row):
    return (
        f"In {int(row['Year'])}, {row['Country']} record: "
        f"Average Temperature {row['Avg Temperature (°C)']}°C, "
        f"CO2 Emissions {row['CO2 Emissions (Tons/Capita)']} Tons/Capita, "
        f"Sea Level Rise {row['Sea Level Rise (mm)']} mm, "
        f"Rainfall {row['Rainfall (mm)']} mm, "
        f"Population {row['Population']}, "
        f"Renewable Energy Share {row['Renewable Energy (%)']}%, "
        f"Extreme Weather Events {row['Extreme Weather Events']}, "
        f"Forest Area {row['Forest Area (%)']}%."
    )

documents = df.apply(row_to_text, axis=1).tolist()

# ─── 3. Embedding & Indexing ────────────────────────────────────────────────
print("[1/3] Loading embedding model and building index...")
# Using a fast and effective embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("✅ Vector Index Ready.")

# ─── 4. Load LLM (Directly, for stability on HF Spaces) ──────────────────────
print("[2/3] Loading local LLM: google/flan-t5-base ...")
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("✅ LLM loaded.")

# ─── 5. Retrieval & Generation ──────────────────────────────────────────────
def retrieve(query: str, k: int = 3) -> str:
    query_vec = embedder.encode([query])
    _, indices = index.search(query_vec, k)
    return "\n\n".join(documents[i] for i in indices[0])

POLICY_PROMPT_TEMPLATE = """You are a Professional Environmental Policy Advisor. Your role is to provide analytical, decision-support, and policy-based reports derived from the provided climate data.

As an advisor, you must adhere to these strict standards in your response:
1. GOVERNANCE STRATEGIES: Detail institutional frameworks and regulatory measures.
2. MITIGATION & ADAPTATION: Provide specific, actionable policy recommendations.
3. COUNTRY-SPECIFIC ANALYSIS: Use the specific climate metrics from the data for the relevant country.
4. DATA-DRIVEN REASONING: Explain the logic following the retrieved data evidence.
5. SUBSTANCE: Your response must be an analytical report of at least 5 to 7 detailed lines.
6. AVOID GENERIC ADVICE: Ensure every statement is tailored to the specific metrics provided.

Retrieved Climate Data Context:
{context}

Stakeholder Policy Inquiry: {query}

Professional Policy Advisory Report:"""

def rag_query(query: str) -> str:
    if not query.strip():
        return "Please enter a policy question."
    try:
        context = retrieve(query)
        prompt = POLICY_PROMPT_TEMPLATE.format(context=context, query=query)
        
        # Optimize for analytical depth + reasonable speed
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=450, 
                min_new_tokens=150, # Enforces the requested minimum 5-7 line length
                do_sample=True,      # Enabled for linguistic variety and analytical depth
                temperature=0.6,    # Slightly lower for more focused/logical advice
                top_p=0.9,
                repetition_penalty=1.8,
                length_penalty=1.5
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"Error during generation: {str(e)}"

# ─── 6. Gradio UI ────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Climate Policy RAG Assistant")
    gr.Markdown("Analytical decision-support and policy-based responses powered by FAISS + Flan-T5.")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Policy Question", 
            placeholder="e.g. What governance strategies should the UK adopt based on its CO2 levels?",
            lines=2
        )
        submit_btn = gr.Button("Analyze", variant="primary")
    
    answer_box = gr.Textbox(label="Professional Policy Advisory Report", lines=12, interactive=False)
    
    gr.Examples(
        examples=[
            ["What mitigation policies should Germany adopt given its CO2 emissions and forest area?"],
            ["How can India balance its population growth with renewable energy adoption?"],
            ["What adaptation strategies are recommended for the US regarding sea level rise?"]
        ],
        inputs=query_input
    )
    
    submit_btn.click(fn=rag_query, inputs=query_input, outputs=answer_box)
    query_input.submit(fn=rag_query, inputs=query_input, outputs=answer_box)

if __name__ == "__main__":
    demo.launch()
