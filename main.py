import os
import json
import difflib
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# ------------------------
# CONFIG
# ------------------------

SESSION_DIR = "./memory/sessions"
HUNTER_INDEX = "./memory/hunter_index"
MAX_ROUNDS = 3

MODELS = {
    "general": "llama3.1:8b",
    "cyber": "deepseek-r1:7b",
    "code": "qwen2.5-coder:7b"
}

supervisor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)
os.makedirs(SESSION_DIR, exist_ok=True)

# ------------------------
# STATE
# ------------------------

class BrainState(TypedDict):
    idea: str
    mode: str
    round: int
    history: list
    retrieved_context: str
    target_file: str


# ------------------------
# SESSION MANAGEMENT
# ------------------------

def save_session(state):
    filename = os.path.join(
        SESSION_DIR,
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(filename, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\nSession saved: {filename}")


def load_last_session():
    files = sorted(os.listdir(SESSION_DIR))
    if not files:
        return None
    last = files[-1]
    with open(os.path.join(SESSION_DIR, last)) as f:
        return json.load(f)


# ------------------------
# RETRIEVAL (AUTO FILE DETECTION)
# ------------------------

def retrieve_hunter_context(query):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=HUNTER_INDEX,
        embedding_function=embeddings
    )

    results = vectorstore.similarity_search(query, k=5)

    if not results:
        return "", None

    context = "\n\n".join([r.page_content for r in results])

    # Auto-detect most frequent source file
    sources = [r.metadata.get("source") for r in results if r.metadata]
    target_file = max(set(sources), key=sources.count) if sources else None

    return context, target_file


# ------------------------
# DIFF GENERATION
# ------------------------

def generate_diff(file_path, new_code):
    if not file_path or not os.path.exists(file_path):
        return "Target file not found."

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        original_code = f.read()

    diff = difflib.unified_diff(
        original_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile="BEFORE",
        tofile="AFTER"
    )

    return "".join(diff)


# ------------------------
# AGENTS
# ------------------------

def agent_general(state: BrainState):
    print("\n[AI1 - GENERAL THINKING]")
    llm = ChatOllama(model=MODELS["general"])
    context = "\n".join([c for _, c in state["history"]])

    response = llm.invoke(
        f"""
    User idea:
    {state['idea']}

    Previous discussion:
    {context}

    Continue the reasoning.
    """
    )
    state["history"].append(("general", response.content))
    return state


def agent_cyber(state: BrainState):
    print("\n[AI2 - CYBER ANALYSIS]")
    llm = ChatOllama(model=MODELS["cyber"])
    prompt = f"Cybersecurity analysis of:\n{state['idea']}"
    response = llm.invoke(prompt)
    state["history"].append(("cyber", response.content))
    return state


def agent_code(state: BrainState):
    print("\n[AI3 - CODE AGENT]")

    context, file_path = retrieve_hunter_context(state["idea"])
    state["retrieved_context"] = context
    state["target_file"] = file_path

    llm = ChatOllama(model=MODELS["code"])

    prompt = f"""
You are modifying an existing codebase.

IMPORTANT:
- Return ONLY the full updated file.
- No explanations.
- Pure code only.

Target file:
{file_path}

Relevant context:
{context}

User request:
{state['idea']}
"""

    response = llm.invoke(prompt)
    new_code = response.content

    diff = generate_diff(file_path, new_code)

    print("\n--- PROPOSED PATCH ---\n")
    print(diff)

    if file_path and "AFTER" in diff:
        apply = input("\nApply patch? (y/n): ")
        if apply.lower() == "y":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            print("Patch applied.")
        else:
            print("Patch discarded.")

    state["history"].append(("code", new_code))
    return state


# ------------------------
# ROUTER
# ------------------------

def router(state: BrainState):
    if state["round"] >= MAX_ROUNDS:
        return END

    state["round"] += 1
    return supervisor_agent(state)
# ------------------------
# SUPERVISOR AGENT
# ------------------------
def supervisor_agent(state):
    print("\n[SUPERVISOR - GPT-4o-mini]")

    history_text = "\n".join([f"{r}: {c}" for r, c in state["history"]])

    prompt = f"""
You supervise a team of AI agents.

User idea:
{state["idea"]}

Conversation so far:
{history_text}

Choose the next agent.

Respond with EXACTLY one of these tokens:

general
cyber
code
finish

Do not explain.
"""

    decision = supervisor_llm.invoke(prompt).content.strip().lower()

    print("Supervisor decision:", decision)

    allowed = ["general", "cyber", "code", "finish"]

    if decision not in allowed:
        print("Invalid supervisor decision, defaulting to general")
        decision = "general"

    if decision == "finish":
        return END

    return decision
# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(BrainState)

builder.add_node("general", agent_general)
builder.add_node("cyber", agent_cyber)
builder.add_node("code", agent_code)

builder.set_entry_point("general")

builder.add_conditional_edges(
    "general",
    router,
    {
        "general": "general",
        "cyber": "cyber",
        "code": "code",
        END: END
    }
)

builder.add_conditional_edges(
    "cyber",
    router,
    {
        "general": "general",
        "cyber": "cyber",
        "code": "code",
        END: END
    }
)

builder.add_conditional_edges(
    "code",
    router,
    {
        "general": "general",
        "cyber": "cyber",
        "code": "code",
        END: END
    }
)

graph = builder.compile()


# ------------------------
# MAIN
# ------------------------

if __name__ == "__main__":

    resume = input("Resume last session? (y/n): ")

    if resume.lower() == "y":
        state = load_last_session()
        if not state:
            print("No previous session found.")
            exit()
    else:
        idea = input("Enter your idea:\n> ")
        mode = input("Mode (general / cyber / code): ")

        state = {
            "idea": idea,
            "mode": mode,
            "round": 0,
            "history": [],
            "retrieved_context": "",
            "target_file": ""
        }

    print("\nRunning Brain (3 rounds max)...")

    final_state = graph.invoke(state)

    print("\n--- FINAL OUTPUT ---\n")
    for role, content in final_state["history"]:
        print(f"\n[{role.upper()}]\n{content}")

    save_session(final_state)
