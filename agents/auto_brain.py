#agent/auto_brain
import os
import json
import subprocess
import re
from typing import TypedDict, List, Dict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DEBUG: Verify re is imported
print(f"[DEBUG] re module imported: {re}")

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from integrations.hunter_smart_adapter import HUNTER_TOOLS, SmartHunterAdapter


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

local_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)
supervisor_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
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
    **HUNTER_TOOLS,
    "memory_search": search_memory,
    "shell": run_command,
}

def hunter_results(domain: str) -> str:
    """Get results from a previous Hunter scan"""
    print(f"[Tool] Fetching Hunter results for {domain}")
    try:
        # You could either:
        # 1. Return the results from memory/cache
        # 2. Or just run the scan again
        result = run_hunter_scan(domain)
        return result
    except Exception as e:
        return f"Error fetching Hunter results: {str(e)}"

# ==============================
# PLANNER AGENT
# ==============================

def planner(state: BrainState):
    NEW_PLANNER_PROMPT = """
You are a cybersecurity automation planner.
 
Task:
{task}
 
Available tools:
 
hunter_scan(domain) - Run full Hunter v27 security scan (takes 20-40 min)
hunter_analyze(domain) - AI analysis of scan results with next steps
hunter_findings(domain) - Get concise findings summary
hunter_investigate(domain:index) - Deep-dive into finding #index
hunter_test(domain|url|type) - Run targeted test (sqli, xss, nuclei)
hunter_report(domain:filename) - Read specific report file
memory_search(query) - Search past scan results
shell(command) - Run a shell command
 
WORKFLOW for domain scanning:
1. hunter_scan(domain) — run the full scan
2. hunter_analyze(domain) — get AI analysis of results
3. hunter_investigate(domain:0) — deep-dive top finding
4. hunter_test(domain|url|type) — verify specific findings
 
WORKFLOW for checking previous results:
1. hunter_findings(domain) — get summary
2. hunter_analyze(domain) — get AI recommendations
 
Return ONLY a JSON list of tool calls in order.
Example: ["hunter_scan(demo.testfire.net)", "hunter_analyze(demo.testfire.net)", "hunter_investigate(demo.testfire.net:0)"]
 
IMPORTANT: Use double quotes in JSON.
Return the JSON list now:
"""

    prompt = NEW_PLANNER_PROMPT.format(task=state["task"])
    response = supervisor_model.invoke(prompt)

    try:
        # Clean the response - remove markdown code blocks if present
        content = response.content
        if "```" in content:
            # Extract JSON from between code blocks
            content = content.split("```")[1].split("```")[0]
            if content.startswith("json"):
                content = content[4:]
        
        # Fix: Replace single quotes with double quotes for valid JSON
        content = content.replace("'", '"')
        
        plan = json.loads(content)
    except Exception as e:
        print(f"[DEBUG] JSON parsing error: {e}")
        if any(kw in state["task"].lower() for kw in ["hunter", "attack", "scan", "analyze", "test", "pentest", "recon", "findings", "investigate"]) or re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', state["task"]):
            domain_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', state["task"])
            if domain_match:
                target = domain_match.group(1)
                task_lower = state["task"].lower()
                if any(kw in task_lower for kw in ["analyze", "findings", "results", "report"]):
                    plan = [f"hunter_findings({target})", f"hunter_analyze({target})"]
                elif "investigate" in task_lower:
                    idx_match = re.search(r'finding\s*(\d+)', task_lower)
                    idx = idx_match.group(1) if idx_match else "0"
                    plan = [f"hunter_investigate({target}:{idx})"]
                else:
                    plan = [f"hunter_scan({target})", f"hunter_analyze({target})", f"hunter_investigate({target}:0)"]
            else:
                plan = [state["task"]]
        else:
            plan = [state["task"]]
    state["plan"] = plan
    state["step"] = 0
    return state


# ==============================
# EXECUTOR AGENT
# ==============================

def executor(state: BrainState):
    # Import re locally as a fallback
    import re
    
    if state["step"] >= len(state["plan"]):
        return state

    current_step = state["plan"][state["step"]]
    print(f"[Executor] Running step: {current_step}")
    
    # Debug: Show what we're working with
    print(f"[DEBUG] Current step type: {type(current_step)}")
    print(f"[DEBUG] Current step content: {current_step}")

    # Detect direct tool call
    match = re.search(r'(\w+)\((.*?)\)', current_step)

    if match:
        tool = match.group(1)
        arg = match.group(2).strip("'\"")
        
        print(f"[Executor] Detected tool call: {tool} with arg: {arg}")

        if tool in TOOLS:
            result = TOOLS[tool](arg)
        else:
            result = f"Unknown tool {tool}"

        state["observation"] = result
        state["history"].append(result)
        return state


    # ---------------------------------
    # If it's a raw task like "analyze attack surface of X", try to extract domain
    # ---------------------------------
    if "analyze attack surface" in current_step.lower():
        # Extract domain
        target = current_step.lower().replace("analyze attack surface of", "").strip()
        print(f"[Executor] Detected raw task, extracting target: {target}")
        
        # Try to run hunter directly
        result = TOOLS["hunter_scan"](target)
        state["observation"] = result
        state["history"].append(result)
        return state

    # ---------------------------------
    # ORIGINAL EXECUTION FLOW for other cases
    # ---------------------------------
    memory_context = search_memory(current_step)

    prompt = f"""
You are an execution AI.

Step:
{current_step}

Memory:
{memory_context}

If you need tools respond in JSON:

{{
 "tool":"shell",
 "input":"command"
}}

Otherwise respond with result text.
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
            else:
                result = f"Unknown tool {tool}"
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
    if not final_result:
        # Fallback: use last observation if result is empty
        final_result = result.get("observation", "") or (result["history"][-1] if result["history"] else "No result")

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

# Add this temporarily at the bottom of auto_brain.py to test
if __name__ == "__main__":
    # Test the planner directly
    test_state = BrainState(
        task="analyze attack surface of anduril.com",
        plan=[],
        step=0,
        observation="",
        result="",
        history=[]
    )
    
    print("Testing planner...")
    result = planner(test_state)
    print(f"Planner result: {result['plan']}")
    
    # Then run the normal loop
    while True:
        task = input("\nBrain_AI task > ")
        if task.strip() == "exit":
            break
        result = run_autonomous_task(task)
        print("\nRESULT:\n")
        print(result)
