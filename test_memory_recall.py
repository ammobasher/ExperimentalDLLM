import torch
import numpy as np
from src.memory import EpisodicMemory
from src.model import PCModel
from src.config import Config
from src.text_adapter import TextAdapter

def test_memory_recall():
    print(">> Starting Memory Recall Validation...")
    
    # 1. Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PCModel().to(device)
    memory = EpisodicMemory(dim=Config.embed_dim)
    adapter = TextAdapter()
    
    # 2. Add a 'Target Fact' to Memory
    # Fact: "The secret code for the vault is 8472-X"
    fact_text = "The secret code for the vault is 8472-X"
    tokens = adapter.tokenizer.encode(fact_text)
    
    with torch.no_grad():
        token_tensor = torch.tensor(tokens).long().to(device)
        # Mean embedding as key
        embeds = model.embedding(token_tensor)
        key_vec = embeds.mean(dim=0).cpu().numpy()
    
    # Add with high surprise (loss_val)
    memory.add(key_vec, np.array(tokens), loss_val=15.0)
    print(f">> Fact stored in memory: '{fact_text}'")
    
    # 3. Retrieve with a Query
    query_text = "What is the vault code?"
    print(f">> Querying memory with: '{query_text}'")
    
    query_tokens = adapter.tokenizer.encode(query_text)
    with torch.no_grad():
        q_tensor = torch.tensor(query_tokens).long().to(device)
        q_embeds = model.embedding(q_tensor)
        q_vec = q_embeds.mean(dim=0).cpu().numpy()
        
    results = memory.retrieve(q_vec, k=1)
    
    if results:
        ret_tokens, score = results[0]
        decoded = adapter.tokenizer.decode(ret_tokens)
        print(f">> RETRIEVED (Score: {score:.4f}): '{decoded}'")
        
        if "8472-X" in decoded:
            print(">> SUCCESS: Memory retrieval verified.")
        else:
            print(">> FAILURE: Fact not retrieved correctly.")
    else:
        print(">> FAILURE: No results retrieved.")

if __name__ == "__main__":
    test_memory_recall()
