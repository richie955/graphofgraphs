# -*- coding: utf-8 -*-
"""Split-safe adaptation of the supplied PrimeKG LLM GoG Colab builder.

Run in Colab after preparing primekg_split_package.zip. The original source
script is on the user's Desktop; this workspace copy is the adapted version.
"""

# ==========================================
# CELL 1: Install Dependencies and Imports
# ==========================================
import sys, subprocess, importlib.util

_REQUIRED = {"pandas": "pandas", "numpy": "numpy", "openai": "openai", "tqdm": "tqdm",
             "json_repair": "json-repair"}
_missing = [pip for mod, pip in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *_missing])

import os, re, json, hashlib, random, time, gc
from pathlib import Path
from collections import defaultdict, Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm
from getpass import getpass

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    class _LocalFiles:
        def upload(self): return {}
        def download(self, path): print(f"File saved at: {path}")
    files = _LocalFiles()

print("Setup & imports complete.")

# ==========================================
# CELL 2: Configuration & Budget
# ==========================================
BASE_DIR = "/content"
SPLIT_PACKAGE_ZIP = f"{BASE_DIR}/primekg_split_package.zip"
SPLIT_PACKAGE_DIR = f"{BASE_DIR}/primekg_split_package"
PRIMEKG_CSV = f"{SPLIT_PACKAGE_DIR}/selected_facts.csv"
SELECTED_ENTITIES_CSV = f"{SPLIT_PACKAGE_DIR}/selected_entities.csv"
INITIAL_OUTPUT_JSON = f"{BASE_DIR}/primekg_initial_gog.json"

# Groq Model (Flagship open-weights model on Groq)
OPENAI_MODEL = "openai/gpt-oss-120b"

ALLOWED_TYPES = {
    "anatomy", "biological_process", "cellular_component", "disease",
    "drug", "effect/phenotype", "exposure", "gene/protein",
    "molecular_function", "pathway"
}
OWNER_TYPES = {"drug", "gene/protein", "pathway"}
OWNER_TYPE_ORDER = ["drug", "gene/protein", "pathway"]

# ---- Per-internal-graph limits; ROOT is the full frozen entity subset. ----
MAX_SELECTED_NODES_PER_OWNER = 20
MIN_NODES_PER_GRAPH = 5
MAX_SELECTED_EDGES_PER_OWNER = 40

# ---- Candidate Limits (trimmed to stay safely under daily token caps) ----
MAX_HOPS = 2
MAX_HOP1_CANDIDATES = 15
MAX_HOP2_CANDIDATES = 15
MAX_CONTEXT_EDGES = 35
MAX_CANDIDATE_EDGES_PER_OWNER = 30
MIN_OWNER_DEGREE = 3

MAX_ROOT_RELATIONS = None  # Never silently drop split-package facts.
DEDUPLICATE_REVERSE_EDGES = True
INCLUDE_ANATOMY_FOR_GENES = False
PRUNE_TO_OWNER_COMPONENT = True
LLM_RETRIES = 3
RANDOM_SEED = 42

# Set True for a six-graph smoke test, then False for the 75-graph build.
QUICK_TEST = False
MAX_GRAPHS_PER_TYPE = 2 if QUICK_TEST else 25

random.seed(RANDOM_SEED)

PER_GRAPH_NODE_CAP = MAX_SELECTED_NODES_PER_OWNER

print(f"Configuration loaded | QUICK_TEST: {QUICK_TEST} | Target graphs per type: {MAX_GRAPHS_PER_TYPE}")
print(f"Per-graph node cap: {PER_GRAPH_NODE_CAP}")

# ==========================================
# CELL 3: Groq Multi-Key Manager
# ==========================================
class GroqKeyManager:
    def __init__(self, max_keys=7):
        self.max_keys = max_keys
        self.keys = []
        self.current_idx = 0

        # Check Colab secrets for GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3
        try:
            from google.colab import userdata
            for suffix in ["", "_2", "_3"]:
                k = userdata.get(f"GROQ_API_KEY{suffix}")
                if k and k not in self.keys:
                    self.keys.append(k)
        except Exception:
            pass

        # Check environment variables
        for suffix in ["", "_2", "_3"]:
            k = os.environ.get(f"GROQ_API_KEY{suffix}")
            if k and k not in self.keys:
                self.keys.append(k)

        # If none found, prompt for at least the primary key
        if not self.keys:
            primary_key = getpass("Enter primary Groq API key: ").strip()
            if primary_key:
                self.keys.append(primary_key)

        self.client = None
        self._set_client()

    def _set_client(self):
        if self.keys:
            self.client = OpenAI(
                api_key=self.keys[self.current_idx],
                base_url="https://api.groq.com/openai/v1"
            )

    def handle_rate_limit(self):
        """Switches to the next key or prompts the user for a new one."""
        # 1. If another pre-loaded key is available, switch immediately:
        if self.current_idx + 1 < len(self.keys):
            self.current_idx += 1
            print(f"\n[KeyManager] 429 hit. Switching to pre-loaded key #{self.current_idx + 1}...")
            self._set_client()
            return True

        # 2. If we haven't reached the max of 3 keys, prompt for another:
        if len(self.keys) < self.max_keys:
            prompt_msg = f"\n[KeyManager] Rate limit hit on key #{self.current_idx + 1}. Enter Groq API key #{len(self.keys) + 1} of {self.max_keys}: "
            new_key = getpass(prompt_msg).strip()
            if new_key and new_key not in self.keys:
                self.keys.append(new_key)
                self.current_idx = len(self.keys) - 1
                self._set_client()
                print(f"[KeyManager] Key #{self.current_idx + 1} added! Resuming immediately...")
                return True

        # 3. If all 3 keys are exhausted, wait out the 1-minute bucket:
        print("\n[KeyManager] All 3 keys hit rate limits. Waiting 15s for quotas to clear...")
        time.sleep(15)
        # Cycle back to the first key:
        self.current_idx = 0
        self._set_client()
        return True

groq_mgr = GroqKeyManager(max_keys=7)
client = groq_mgr.client

print(f"KeyManager active with {len(groq_mgr.keys)} key(s) loaded.")
print("Using key #1:", groq_mgr.keys[0][:8] + "..." if groq_mgr.keys else "None")

# ==========================================
# CELL 4: Helpers & Robust JSON LLM Caller
# ==========================================
class LLMSelectionError(RuntimeError):
    pass

def norm_text(value):
    value = "" if value is None else str(value).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()

def safe_str(value):
    if value is None: return ""
    try:
        if pd.isna(value): return ""
    except Exception: pass
    return str(value).strip()

def stable_id(prefix, value):
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"

def canonical_type(value):
    val = norm_text(value).replace(" ", "_")
    aliases = {
        "gene_protein": "gene/protein",
        "gene_and_protein": "gene/protein",
        "effect_phenotype": "effect/phenotype",
        "effect_phenotype_": "effect/phenotype",
    }
    return aliases.get(val, val)

import json_repair

def parse_json_response(text):
    text = (text or "").strip()

    # 1. Extract outermost JSON block ({...} or [...])
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(0)

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        pass

    if parsed is None or isinstance(parsed, str):
        try:
            import json_repair
            parsed = json_repair.loads(text)
        except Exception:
            pass

    if parsed is None or isinstance(parsed, str):
        # Pure Python regex fix
        text = re.sub(r'\}\s*\{', '}, {', text)
        text = re.sub(r'\]\s*\[', '], [', text)
        text = re.sub(r',\s*([\]\}])', r'\1', text)
        try:
            parsed = json.loads(text)
        except Exception:
            pass

    # If it was double-encoded as a JSON string, unwrap it:
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            pass

    # Must be a dict or list; otherwise raise so llm_json catches and retries
    if not isinstance(parsed, (dict, list)):
        raise ValueError(f"Expected JSON object or list, got: {type(parsed)}")

    return parsed

def llm_json(system_prompt, user_payload, temperature=0.1):
    last_error = None
    # Allow up to 6 total attempts (giving room across multiple key rotations)
    for attempt in range(6):
        try:
            response = groq_mgr.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=temperature,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ]
            )
            msg = response.choices[0].message
            raw_text = msg.content or getattr(msg, "reasoning_content", "") or ""
            return parse_json_response(raw_text)

        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()

            # If rate-limited (429), rotate key or ask user:
            if "429" in err_str or "rate_limit" in err_str:
                groq_mgr.handle_rate_limit()
                time.sleep(1) # brief pause before retrying with the new key
                continue
            else:
                time.sleep(2 ** min(attempt, 3))

    raise LLMSelectionError(f"LLM request failed after retries: {last_error}")

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")

def balanced_pick(items, group_of, sort_key, limit):
    groups = defaultdict(list)
    for item in items:
        groups[group_of(item)].append(item)
    for members in groups.values():
        members.sort(key=sort_key)
    order = sorted(groups, key=str)
    pointers = {g: 0 for g in order}
    picked = []
    while len(picked) < limit:
        progressed = False
        for g in order:
            if pointers[g] < len(groups[g]):
                picked.append(groups[g][pointers[g]])
                pointers[g] += 1
                progressed = True
                if len(picked) >= limit: break
        if not progressed: break
    return picked

print("Helper functions ready.")

# ==========================================
# CELL 5: Load PrimeKG Dataset & Build Graph
# ==========================================
DRIVE_PACKAGE_PATH = "/content/drive/MyDrive/primekg_split_package.zip"
if not os.path.exists(SPLIT_PACKAGE_ZIP):
    import shutil
    from google.colab import drive
    drive.mount("/content/drive")
    if os.path.exists(DRIVE_PACKAGE_PATH):
        shutil.copy(DRIVE_PACKAGE_PATH, SPLIT_PACKAGE_ZIP)
    else:
        uploaded = files.upload()
        if "primekg_split_package.zip" not in uploaded:
            raise FileNotFoundError("Upload primekg_split_package.zip")

import zipfile
with zipfile.ZipFile(SPLIT_PACKAGE_ZIP) as archive:
    required = {"selected_facts.csv", "selected_entities.csv", "split_manifest.csv", "audit_report.json"}
    if not required.issubset(archive.namelist()):
        raise ValueError(f"Split package missing: {sorted(required - set(archive.namelist()))}")
    os.makedirs(SPLIT_PACKAGE_DIR, exist_ok=True)
    for name in required:
        archive.extract(name, SPLIT_PACKAGE_DIR)
    split_audit = json.loads(archive.read("audit_report.json"))
if split_audit.get("validation") != "passed":
    raise ValueError("Split package audit did not pass")

all_source_df = pd.read_csv(PRIMEKG_CSV, dtype=str, keep_default_na=False)
if set(all_source_df["split"]) != {"train", "validation", "test"}:
    raise ValueError("Expected train, validation and test source facts")
if all_source_df["fact_id"].duplicated().any():
    raise ValueError("Source fact IDs must be unique")
selected_entity_df = pd.read_csv(SELECTED_ENTITIES_CSV, dtype=str, keep_default_na=False)
selected_root_ids = set(selected_entity_df["entity_id"])
if not (set(all_source_df["x_index"]) | set(all_source_df["y_index"])).issubset(selected_root_ids):
    raise ValueError("Source fact endpoints must have selected ROOT entities")
primekg_df = all_source_df.loc[all_source_df["split"] == "train"].copy()
print("Training-only rows available to LLM builder:", len(primekg_df))

header = primekg_df.columns.tolist()
required_columns = ["x_index", "x_type", "x_name", "y_index", "y_type", "y_name", "relation"]
has_display_relation = "display_relation" in header
usecols = required_columns + (["display_relation"] if has_display_relation else [])

usecols.append("fact_id")
primekg_df = primekg_df[usecols].copy()

# Unique entities
entities = {}
name_to_ids_by_type = defaultdict(list)

for entity_id, entity_type, entity_name in selected_entity_df[["entity_id", "type", "name"]].itertuples(index=False, name=None):
    entity_type = canonical_type(entity_type)
    entities[str(entity_id)] = {"entity_id": str(entity_id), "name": safe_str(entity_name),
                                "type": entity_type, "source": "PrimeKG", "aliases": []}

for entity_id, record in entities.items():
    for name in [record.get("name", "")] + record.get("aliases", []):
        normalized = norm_text(name)
        if normalized:
            name_to_ids_by_type[(normalized, record["type"])].append(entity_id)

ids = list(entities.keys())
idx = {entity_id: i for i, entity_id in enumerate(ids)}
TYPE_OF = [entities[e]["type"] for e in ids]
N_ENT = len(ids)

x_idx = primekg_df["x_index"].map(idx)
y_idx = primekg_df["y_index"].map(idx)
valid_rows = (x_idx.notna() & y_idx.notna() & (x_idx != y_idx)).to_numpy()

src = x_idx[valid_rows].astype(np.int64).to_numpy()
dst = y_idx[valid_rows].astype(np.int64).to_numpy()
rel_codes, REL_NAMES = pd.factorize(primekg_df.loc[valid_rows, "relation"].astype(str))
disp_codes, DISP_NAMES = (pd.factorize(primekg_df.loc[valid_rows, "display_relation"].fillna("").astype(str))
                          if has_display_relation else (rel_codes, REL_NAMES))
REL_NAMES, DISP_NAMES = list(REL_NAMES), list(DISP_NAMES)
E_FACT_ID = primekg_df.loc[valid_rows, "fact_id"].to_numpy(dtype=str)
del primekg_df, x_idx, y_idx, valid_rows
gc.collect()

E_SRC = src.astype(np.int32)
E_DST = dst.astype(np.int32)
E_REL = rel_codes.astype(np.int32)
E_DISP = disp_codes.astype(np.int32)
del src, dst, rel_codes, disp_codes
gc.collect()
N_EDGES = len(E_SRC)

_nodes_all = np.concatenate([E_SRC, E_DST])
_eids_all = np.concatenate([np.arange(N_EDGES, dtype=np.int64)] * 2)
_order = np.argsort(_nodes_all, kind="stable")
ADJ_EID = _eids_all[_order]
DEGREE = np.bincount(_nodes_all, minlength=N_ENT)
ADJ_PTR = np.concatenate([[0], np.cumsum(DEGREE)])
del _nodes_all, _eids_all, _order
gc.collect()

RANK = np.random.RandomState(RANDOM_SEED).permutation(N_ENT)

print(f"PrimeKG loaded: {N_ENT:,} entities | {N_EDGES:,} edges retained.")

# ==========================================
# CELL 6: Retrieval & Graph Building Logic
# ==========================================
def incident_edges(node):
    return ADJ_EID[ADJ_PTR[node]:ADJ_PTR[node + 1]]

def other_endpoint(node, eids):
    s, d = E_SRC[eids], E_DST[eids]
    return np.where(s == node, d, s)

def edge_record(eid):
    eid = int(eid)
    relation = REL_NAMES[E_REL[eid]]
    return {
        "source": ids[E_SRC[eid]], "target": ids[E_DST[eid]],
        "relation": relation, "display_relation": DISP_NAMES[E_DISP[eid]] or relation,
        "source_db": "PrimeKG", "source_fact_id": str(E_FACT_ID[eid]),
        "evidence": "", "paper_id": ""
    }

def induced_edge_ids(node_indices):
    node_arr = np.array(sorted(set(int(n) for n in node_indices)), dtype=np.int64)
    if len(node_arr) == 0: return np.array([], dtype=np.int64)
    pieces = [incident_edges(n) for n in node_arr]
    cand = np.unique(np.concatenate(pieces))
    mask = np.isin(E_SRC[cand], node_arr) & np.isin(E_DST[cand], node_arr)
    return cand[mask]

def root_edge_records_for(root_ids, limit):
    if limit is not None:
        raise ValueError("ROOT relation caps would silently change the frozen split")
    records = []
    for row in all_source_df.itertuples(index=False):
        if row.x_index in root_ids and row.y_index in root_ids:
            records.append({"source": row.x_index, "target": row.y_index,
                            "relation": row.relation, "display_relation": row.display_relation,
                            "source_fact_id": row.fact_id, "split": row.split})
    if len(records) != len(all_source_df):
        raise ValueError("ROOT universe omitted facts from the split package")
    return records, False

def retrieve_two_hop(owner_id):
    owner = idx[owner_id]
    exclude_anatomy = TYPE_OF[owner] == "gene/protein" and not INCLUDE_ANATOMY_FOR_GENES
    eids = incident_edges(owner)
    first = np.unique(other_endpoint(owner, eids))
    first = first[first != owner]
    if exclude_anatomy:
        first = np.array([n for n in first.tolist() if TYPE_OF[n] != "anatomy"], dtype=np.int64)
    first_all = first.tolist()

    hop1 = balanced_pick(first_all, group_of=lambda n: TYPE_OF[n], sort_key=lambda n: RANK[n], limit=MAX_HOP1_CANDIDATES)
    seen = set(first_all) | {owner}
    parts = [np.unique(other_endpoint(h, incident_edges(h))) for h in hop1]
    hop2_items = []
    if parts:
        values, counts = np.unique(np.concatenate(parts), return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            if value in seen or (exclude_anatomy and TYPE_OF[value] == "anatomy"): continue
            hop2_items.append((value, count))
    hop2 = [item[0] for item in balanced_pick(hop2_items, group_of=lambda t: TYPE_OF[t[0]], sort_key=lambda t: (-t[1], RANK[t[0]]), limit=MAX_HOP2_CANDIDATES)]

    hop_of = {owner: 0}
    hop_of.update({n: 1 for n in hop1})
    hop_of.update({n: 2 for n in hop2})
    return [owner] + hop1 + hop2, hop_of

def select_candidate_edges(node_indices, hop_of, limit):
    tiers = defaultdict(list)
    for eid in induced_edge_ids(node_indices).tolist():
        tiers[hop_of[int(E_SRC[eid])] + hop_of[int(E_DST[eid])]].append(eid)
    picked = []
    for tier in sorted(tiers):
        remaining = limit - len(picked)
        if remaining <= 0: break
        picked.extend(balanced_pick(tiers[tier], group_of=lambda e: int(E_REL[e]), sort_key=lambda e: (e * 2654435761) % 4294967296, limit=remaining))
    return picked

def build_node_links(edge_ids, hop_of, max_links=3):
    links = defaultdict(list)
    for eid in edge_ids:
        s, d = int(E_SRC[eid]), int(E_DST[eid])
        hs, hd = hop_of[s], hop_of[d]
        rel = REL_NAMES[E_REL[eid]]
        if hd > hs and len(links[d]) < max_links:
            links[d].append({"relation": rel, "linked_to": entities[ids[s]]["name"], "direction": "incoming"})
        elif hs > hd and len(links[s]) < max_links:
            links[s].append({"relation": rel, "linked_to": entities[ids[d]]["name"], "direction": "outgoing"})
    return links

NODE_SELECTION_SYSTEM = f"""You are a biomedical knowledge-graph curator building the internal graph of ONE owner entity (drug, gene/protein or pathway).
Allowed types: {sorted(ALLOWED_TYPES)}.
TASK: Rank candidate NODES by biological relevance to the owner. Return JSON only:
{{"ranked_nodes":[{{"id":"entity_id","score":8,"reason":"short reason"}}]}}
"""

EDGE_SELECTION_SYSTEM = f"""You are a biomedical knowledge-graph curator. Select the RELATIONSHIPS (edges) that give meaningful biological context.
TASK: Pick the best candidate edges connecting the selected nodes to the owner. Return JSON only:
{{"selected_edges":[{{"index":0,"score":8,"reason":"short reason"}}]}}
"""

def llm_rank_nodes(owner_id, node_payload, max_nodes):
    owner = entities[owner_id]
    result = llm_json(NODE_SELECTION_SYSTEM, {
        "owner": {"id": owner_id, "name": owner["name"], "type": owner["type"]},
        "candidate_nodes": node_payload, "max_nodes_to_select": max_nodes
    })

    # Safely handle dict, list, or string
    if isinstance(result, list):
        items_list = result
    elif isinstance(result, dict):
        items_list = result.get("ranked_nodes", []) or []
    else:
        items_list = []

    valid = {item["id"] for item in node_payload}
    ranked, seen = [], set()
    for position, item in enumerate(items_list):
        node_id = str(item.get("id", "")) if isinstance(item, dict) else str(item)
        score = float(item.get("score", 0)) if isinstance(item, dict) else 0.0
        reason = safe_str(item.get("reason", "")) if isinstance(item, dict) else ""
        if node_id not in valid or node_id == owner_id or node_id in seen:
            continue
        seen.add(node_id)
        ranked.append((-score, position, node_id, reason))
    ranked.sort()
    return [(node_id, reason) for _, _, node_id, reason in ranked][:max_nodes]


def llm_rank_edges(owner_id, node_payload, edge_payload, max_edges):
    owner = entities[owner_id]
    result = llm_json(EDGE_SELECTION_SYSTEM, {
        "owner": {"id": owner_id, "name": owner["name"], "type": owner["type"]},
        "selected_nodes": node_payload, "candidate_edges": edge_payload, "max_edges_to_select": max_edges
    })

    # Safely handle dict, list, or string
    if isinstance(result, list):
        items_list = result
    elif isinstance(result, dict):
        items_list = result.get("selected_edges", []) or []
    else:
        items_list = []

    ranked, seen = [], set()
    for position, item in enumerate(items_list):
        idx_val = item.get("index") if isinstance(item, dict) else item
        score = float(item.get("score", 0)) if isinstance(item, dict) else 0.0
        reason = safe_str(item.get("reason", "")) if isinstance(item, dict) else ""
        if isinstance(idx_val, str) and idx_val.isdigit():
            idx_val = int(idx_val)
        if not isinstance(idx_val, int) or not (0 <= idx_val < len(edge_payload)) or idx_val in seen:
            continue
        seen.add(idx_val)
        ranked.append((-score, position, idx_val, reason))
    ranked.sort()
    return [(index, reason) for _, _, index, reason in ranked][:max_edges]

def prune_to_owner_component(owner_id, node_ids, edges):
    adjacency = defaultdict(set)
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
    reached, queue = {owner_id}, deque([owner_id])
    while queue:
        curr = queue.popleft()
        for neighbour in adjacency[curr]:
            if neighbour not in reached:
                reached.add(neighbour)
                queue.append(neighbour)
    return [n for n in node_ids if n in reached], [e for e in edges if e["source"] in reached and e["target"] in reached]

def membership_source_paths(owner_id, node_ids, edges):
    """Trace each selected occurrence to training source facts in its owner component."""
    adjacency = defaultdict(list)
    for edge in edges:
        for source, target in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
            adjacency[source].append((target, edge["source_fact_id"]))
    support = {owner_id: []}
    queue = deque([owner_id])
    while queue:
        current = queue.popleft()
        for neighbor, fact_id in sorted(adjacency[current]):
            if neighbor not in support:
                support[neighbor] = [*support[current], fact_id]
                queue.append(neighbor)
    if set(support) != set(node_ids):
        raise ValueError("An internal member is disconnected from its owner")
    return support

def build_primekg_internal_graph(entity_type, owner_id):
    owner = idx[owner_id]
    candidates, hop_of = retrieve_two_hop(owner_id)
    if len(candidates) < MIN_NODES_PER_GRAPH: return None

    context_edges = select_candidate_edges(candidates, hop_of, MAX_CONTEXT_EDGES)
    links = build_node_links(context_edges, hop_of)
    node_payload = [{
        "id": ids[n], "name": entities[ids[n]]["name"], "type": TYPE_OF[n], "hop": hop_of[n], "links": links.get(n, [])
    } for n in candidates if n != owner]

    ranked_nodes = llm_rank_nodes(owner_id, node_payload, PER_GRAPH_NODE_CAP - 1)
    if not ranked_nodes: return None
    selected = [owner] + [idx[node_id] for node_id, _ in ranked_nodes]
    node_reasons = {node_id: reason for node_id, reason in ranked_nodes}

    edge_ids = select_candidate_edges(selected, hop_of, MAX_CANDIDATE_EDGES_PER_OWNER)
    if not edge_ids: return None
    candidate_edge_records = [edge_record(e) for e in edge_ids]
    selected_payload = [{"id": ids[n], "name": entities[ids[n]]["name"], "type": TYPE_OF[n], "hop": hop_of[n]} for n in selected]
    edge_payload = [{
        "index": i, "source": rec["source"], "source_name": entities[rec["source"]]["name"],
        "relation": rec["relation"], "display_relation": rec["display_relation"],
        "target": rec["target"], "target_name": entities[rec["target"]]["name"]
    } for i, rec in enumerate(candidate_edge_records)]

    ranked_edges = llm_rank_edges(owner_id, selected_payload, edge_payload, MAX_SELECTED_EDGES_PER_OWNER)
    chosen_edges = []
    for index, reason in ranked_edges:
        rec = dict(candidate_edge_records[index])
        rec["llm_reason"] = reason
        chosen_edges.append(rec)

    node_ids = [ids[n] for n in selected]
    if PRUNE_TO_OWNER_COMPONENT:
        node_ids, chosen_edges = prune_to_owner_component(owner_id, node_ids, chosen_edges)
    if len(node_ids) < MIN_NODES_PER_GRAPH or not chosen_edges: return None
    support_paths = membership_source_paths(owner_id, node_ids, chosen_edges)

    return {
        "graph_id": f"G_{entity_type.replace('/', '_')}_{owner_id}",
        "owner_entity_id": owner_id,
        "owner_type": entity_type,
        "node_ids": node_ids,
        "node_hops": {n: hop_of[idx[n]] for n in node_ids},
        "candidate_count": len(candidates),
        "edges": chosen_edges,
        "membership_source_fact_ids": support_paths,
        "node_reasons": {n: node_reasons.get(n, "owner") for n in node_ids}
    }

print("Graph retrieval & prompt functions compiled.")

# (Tests 1 single drug without running any loops. Confirms that Groq returns valid JSON without errors).
# ==========================================
# CELL 7: Smoke Test on 1 Single Graph
# ==========================================
owner_queues = {}
for entity_type in OWNER_TYPE_ORDER:
    pool = [ids[i] for i in range(N_ENT) if TYPE_OF[i] == entity_type and DEGREE[i] >= MIN_OWNER_DEGREE]
    random.Random(RANDOM_SEED).shuffle(pool)
    owner_queues[entity_type] = pool

print("Eligible owners by type:", {t: len(owner_queues[t]) for t in OWNER_TYPE_ORDER})

# Test on 1 drug:
test_owner_id = owner_queues["drug"][-1]
test_owner_name = entities[test_owner_id]["name"]
print(f"\nTesting 1 graph build on drug '{test_owner_name}' (ID: {test_owner_id}) ...")

test_graph = build_primekg_internal_graph("drug", test_owner_id)

if test_graph:
    print("SUCCESS! Test graph generated:")
    print(f" - Graph ID: {test_graph['graph_id']}")
    print(f" - Selected nodes: {len(test_graph['node_ids'])}")
    print(f" - Selected edges: {len(test_graph['edges'])}")
else:
    print("Test graph returned None (try popping another owner).")

# ==========================================
# CELL 8: Build Balanced GoG & Export JSON
# ==========================================

# 1. Loop Helper: Pop owners until a valid graph is found
def next_graph_for_type(entity_type, max_consecutive_llm_failures=5):
    queue = owner_queues[entity_type]
    failures = 0
    while queue:
        owner_id = queue.pop()
        try:
            graph = build_primekg_internal_graph(entity_type, owner_id)
        except LLMSelectionError as exc:
            failures += 1
            print(f"[{entity_type}] owner {owner_id} skipped: {exc}")
            if failures >= max_consecutive_llm_failures:
                print(f"Hit failure limit for {entity_type}. Stopping queue.")
                return None
            continue
        if graph is not None:
            return graph
    return None

# 2. Serialization Helpers (Defined BEFORE the loop so finally can use them!)
def build_entity_membership(graphs):
    membership = defaultdict(set)
    for g in graphs:
        for eid in g["node_ids"]: membership[eid].add(g["graph_id"])
    return membership

def make_graph_nodes_and_records(graphs, root_ids):
    g_records = [{"graph_id": "ROOT", "type": "root"}]
    gn_records, local_ids = [], {}
    for eid in sorted(root_ids):
        nid = f"{eid}@ROOT"
        local_ids[("ROOT", eid)] = nid
        gn_records.append({"node_id": nid, "entity_id": eid, "graph_id": "ROOT"})
    for g in graphs:
        gid = g["graph_id"]
        g_records.append({"graph_id": gid, "type": "internal", "parent_entity_id": g["owner_entity_id"], "owner_type": g["owner_type"]})
        for eid in g["node_ids"]:
            nid = f"{eid}@{gid}"
            local_ids[(gid, eid)] = nid
            gn_records.append({"node_id": nid, "entity_id": eid, "graph_id": gid,
                               "supporting_source_fact_ids": g["membership_source_fact_ids"][eid]})
    return g_records, gn_records, local_ids

def build_relation_records(graphs, root_ids, root_edges):
    g_recs, gn_recs, local_ids = make_graph_nodes_and_records(graphs, root_ids)
    rel_recs = []
    for idx_num, e in enumerate(root_edges):
        if e["source"] in root_ids and e["target"] in root_ids:
            rel_recs.append({"relation_id": f"R_ROOT_{idx_num}", "source_node_id": local_ids[("ROOT", e["source"])],
                             "type": e["relation"], "target_node_id": local_ids[("ROOT", e["target"])],
                             "display_relation": e.get("display_relation") or e["relation"], "source_db": "PrimeKG",
                             "scope": "root", "source_fact_id": e["source_fact_id"], "split": e["split"]})
    for g in graphs:
        gid = g["graph_id"]
        for idx_num, e in enumerate(g["edges"]):
            if (gid, e["source"]) in local_ids and (gid, e["target"]) in local_ids:
                rel_recs.append({"relation_id": f"R_{gid}_{idx_num}", "source_node_id": local_ids[(gid, e["source"])],
                                 "type": e["relation"], "target_node_id": local_ids[(gid, e["target"])],
                                 "display_relation": e.get("display_relation") or e["relation"], "source_db": "PrimeKG",
                                 "scope": "internal", "llm_reason": e.get("llm_reason", ""),
                                 "source_fact_id": e["source_fact_id"], "split": "train",
                                 "derived_channel": True})
    membership = build_entity_membership(graphs)
    return g_recs, gn_recs, rel_recs, membership, 0


# 3. Main Build Loop with Guaranteed Save
accepted_graphs = []
root_entity_universe = set(selected_root_ids)
stop_reason = f"reached MAX_GRAPHS_PER_TYPE = {MAX_GRAPHS_PER_TYPE}"

try:
    with ThreadPoolExecutor(max_workers=1) as executor:
        for round_index in tqdm(range(MAX_GRAPHS_PER_TYPE), desc="Balanced rounds"):
            futures = {t: executor.submit(next_graph_for_type, t) for t in OWNER_TYPE_ORDER}
            round_graphs = [futures[t].result() for t in OWNER_TYPE_ORDER]
            if any(graph is None for graph in round_graphs):
                stop_reason = "hit quota or owner pool stopped"
                break
            proposed_root = set(root_entity_universe)
            for graph in round_graphs:
                proposed_root.update(graph["node_ids"])
            accepted_graphs.extend(round_graphs)
            root_entity_universe = proposed_root
            time.sleep(2)

finally:
    # Always runs: compiles and downloads whatever was completed
    if accepted_graphs:
        print(f"\n[Save Guard] Saving {len(accepted_graphs)} completed graphs to JSON...")
        root_edges_initial, _ = root_edge_records_for(root_entity_universe, MAX_ROOT_RELATIONS)
        g_recs, gn_recs, rel_recs, membership, cross_count = build_relation_records(
            accepted_graphs, root_entity_universe, root_edges_initial
        )
        entity_records = []
        owner_ids = {g["owner_entity_id"] for g in accepted_graphs}
        for eid in sorted(root_entity_universe):
            rec = dict(entities[eid])
            gids = sorted(membership.get(eid, set()))
            rec["graph_frequency"] = len(gids)
            rec["appears_in"] = gids
            rec["is_common"] = len(gids) >= 2
            rec["has_internal_graph"] = eid in owner_ids
            entity_records.append(rec)

        initial_output = {
            "entities": entity_records,
            "graphs": g_recs,
            "graph_nodes": gn_recs,
            "relations": rel_recs,
            "metadata": {
                "project": "PrimeKG-based LLM-assisted GoG",
                "model": OPENAI_MODEL,
                "internal_graph_count": len(accepted_graphs),
                "root_unique_entities": len(root_entity_universe),
                "total_graph_local_nodes": len(gn_recs),
                "total_relations": len(rel_recs),
                "source_sha256": split_audit["source_sha256"],
                "selected_fact_set_sha256": split_audit["selected_fact_set_sha256"],
                "split_manifest_sha256": split_audit["split_manifest_sha256"],
                "split_counts": split_audit["fact_counts_by_split"],
                "derived_internal_channels": sum(bool(r.get("derived_channel")) for r in rel_recs),
                "cross_context_semantic_facts": 0,
                "stop_reason": stop_reason
            }
        }
        save_json(initial_output, INITIAL_OUTPUT_JSON)
        files.download(INITIAL_OUTPUT_JSON)
    else:
        print("No completed rounds to save.")
