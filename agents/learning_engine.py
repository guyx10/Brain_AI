import os
import json
import datetime
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# =============================
# CONFIG
# =============================

SUPERVISOR_MODEL = "gpt-4o-mini"
LOCAL_MODEL = "llama3.1:8b"

EXPERIENCE_FILE = "./memory/experience_log.json"


supervisor = ChatOpenAI(
    model=SUPERVISOR_MODEL,
    temperature=0
)

local_llm = ChatOllama(
    model=LOCAL_MODEL
)


# =============================
# VECTOR MEMORY
# =============================

embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    persist_directory="./memory/vector_db",
    embedding_function=embedding
)


# =============================
# EXPERIENCE LOGGER
# =============================

def log_experience(task: str, plan: List[str], result: str):

    entry = {
        "time": str(datetime.datetime.utcnow()),
        "task": task,
        "plan": plan,
        "result": result
    }

    data = []

    if os.path.exists(EXPERIENCE_FILE):
        with open(EXPERIENCE_FILE) as f:
            data = json.load(f)

    data.append(entry)

    with open(EXPERIENCE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    vector_db.add_texts(
        texts=[json.dumps(entry)],
        metadatas=[{"type": "experience"}]
    )


# =============================
# EXPERIENCE RETRIEVAL
# =============================

def retrieve_similar_experiences(task: str):

    docs = vector_db.similarity_search(task, k=5)

    experiences = []

    for d in docs:
        if "experience" in d.metadata.get("type", ""):
            experiences.append(d.page_content)

    return "\n".join(experiences)


# =============================
# STRATEGY IMPROVER
# =============================

def improve_strategy(task: str, plan: List[str], result: str):

    previous = retrieve_similar_experiences(task)

    prompt = f"""
You are an AI research system improving your own strategy.

Task:
{task}

Plan used:
{plan}

Result:
{result}

Previous similar experiences:
{previous}

Suggest improvements to the strategy.

Return JSON:

{{
 "better_plan":[]
}}
"""

    response = supervisor.invoke(prompt)

    try:
        data = json.loads(response.content)
        return data["better_plan"]
    except:
        return plan


# =============================
# PROMPT SELF-OPTIMIZER
# =============================

PROMPT_FILE = "./memory/prompt_versions.json"


def optimize_prompt(agent_name: str, prompt_text: str):

    history = []

    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE) as f:
            history = json.load(f)

    improve_prompt = f"""
You are improving an AI agent prompt.

Agent:
{agent_name}

Current prompt:
{prompt_text}

Make the prompt clearer and more effective.

Return improved prompt.
"""

    response = supervisor.invoke(improve_prompt)

    new_prompt = response.content

    history.append({
        "agent": agent_name,
        "old": prompt_text,
        "new": new_prompt
    })

    with open(PROMPT_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return new_prompt


# =============================
# LEARNING LOOP
# =============================

def learning_cycle(task: str, plan: List[str], result: str):

    print("\n[Learning] Logging experience...")

    log_experience(task, plan, result)

    print("[Learning] Improving strategy...")

    better_plan = improve_strategy(task, plan, result)

    return better_plan


# =============================
# EXPERIENCE SUMMARY
# =============================

def summarize_experience():

    if not os.path.exists(EXPERIENCE_FILE):
        return "No experiences yet."

    with open(EXPERIENCE_FILE) as f:
        data = json.load(f)

    last = data[-10:]

    prompt = f"""
Summarize lessons from these AI experiences:

{json.dumps(last, indent=2)}
"""

    summary = supervisor.invoke(prompt)

    return summary.content


# =============================
# CLI TEST
# =============================

if __name__ == "__main__":

    while True:

        task = input("\nTask > ")

        if task == "exit":
            break

        plan = ["analyze task", "execute solution"]

        result = local_llm.invoke(
            f"Execute task: {task}"
        ).content

        print("\nResult:\n", result)

        improved = learning_cycle(task, plan, result)

        print("\nImproved strategy:")
        print(improved)
