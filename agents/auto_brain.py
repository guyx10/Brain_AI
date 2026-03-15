import os
import json
import subprocess
from typing import TypedDict, List, Dict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from integrations.hunter_adapter import (
    run_hunter_recon,
    run_hunter_scan,
    run_hunter_endpoints
)

# ==============================
# CONFIG
# ==============================

MAX_LOOPS = 6
MODEL_LOCAL = "llama3.1:8b"
MODEL_SUPERVISOR = "llama3.1:8b"


# ==============================
# STATE
# ==============================

class BrainState(TypedDict):
    task: str
    plan: List[str]
    step: int
    observation: str
    result: str
    history: List[str]


# ==============================
# MODELS
# ==============================

local_model = ChatOllama(model=MODEL_LOCAL)

supervisor_model = ChatOllama(
    model=MODEL_SUPERVISOR,
    temperature=0
)


# ==============================
# VECTOR MEMORY
# ==============================

embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    persist_directory="./memory/vector_db",
    embedding_function=embedding
)


# ==============================
# TOOLS
# ==============================

def run_command(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            cmd,
            shell=True,
            stderr=subprocess.STDOUT,
            timeout=60
        )
        return out.decode()
    except Exception as e:
        return str(e)


def search_memory(query: str) -> str:
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])


TOOLS = {

    "hunter_recon": run_hunter_recon,

    "hunter_scan": run_hunter_scan,

    "hunter_endpoints": run_hunter_endpoints,

    "memory_search": search_memory
}


# ==============================
# PLANNER AGENT
# ==============================

def planner(state: BrainState):

    prompt = f"""
You are a planning AI.

Break the task into small executable steps.

TASK:
{state["task"]}

Return JSON list.

Example:
["step1", "step2"]
"""

    response = supervisor_model.invoke(prompt)

    try:
        plan = json.loads(response.content)
    except:
        plan = [state["task"]]

    state["plan"] = plan
    state["step"] = 0

    return state


# ==============================
# EXECUTOR AGENT
# ==============================

def executor(state: BrainState):

    if state["step"] >= len(state["plan"]):
        return state

    current_step = state["plan"][state["step"]]

    memory_context = search_memory(current_step)

    prompt = f"""
You are a security automation agent.

Available tools:

hunter_recon(domain)
hunter_scan(domain)
hunter_endpoints(domain)
memory_search(query)

Step:
{current_step}

Respond ONLY in JSON when using tools.

Example:

{{
 "tool": "hunter_recon",
 "input": "example.com"
}}
"""

    response = local_model.invoke(prompt)
    text = response.content

    if text.startswith("{"):
        try:
            data = json.loads(text)
            tool = data["tool"]
            inp = data["input"]

            if tool in TOOLS:

                if isinstance(inp, str):
                    result = TOOLS[tool](inp)
                else:
                    result = TOOLS[tool](**inp)

        except:
            result = text
    else:
        result = text

    state["observation"] = result
    state["history"].append(result)

    return state


# ==============================
# CRITIC AGENT
# ==============================

def critic(state: BrainState):

    prompt = f"""
You are a critic AI.

Task:
{state["task"]}

Plan:
{state["plan"]}

Last result:
{state["observation"]}

Should the system:

continue
or
finish

Return only one word.
"""

    response = supervisor_model.invoke(prompt)
    decision = response.content.lower()

    if "finish" in decision:
        state["result"] = state["observation"]
        state["step"] = len(state["plan"])
    else:
        state["step"] += 1

    return state


# ==============================
# ROUTER
# ==============================

def router(state: BrainState):

    if state["step"] >= len(state["plan"]):
        return END

    if len(state["history"]) > MAX_LOOPS:
        return END

    return "executor"


# ==============================
# GRAPH
# ==============================

builder = StateGraph(BrainState)

builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("critic", critic)

builder.set_entry_point("planner")

builder.add_edge("planner", "executor")
builder.add_edge("executor", "critic")
builder.add_conditional_edges("critic", router)

graph = builder.compile()


# ==============================
# RUN FUNCTION
# ==============================

from learning_engine import learning_cycle


def run_autonomous_task(task: str):

    state = BrainState(
        task=task,
        plan=[],
        step=0,
        observation="",
        result="",
        history=[]
    )

    result = graph.invoke(state)

    final_result = result["result"]

    improved_plan = learning_cycle(
        task,
        result["plan"],
        final_result
    )

    print("\n[Brain Learning]")
    print("Improved plan suggestion:")
    print(improved_plan)

    return final_result


# ==============================
# CLI
# ==============================

if __name__ == "__main__":

    while True:
        task = input("\nBrain_AI task > ")

        if task.strip() == "exit":
            break

        result = run_autonomous_task(task)

        print("\nRESULT:\n")
        print(result)
