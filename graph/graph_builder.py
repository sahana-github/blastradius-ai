import ast
import pickle
import re
import sys
from pathlib import Path

import networkx as nx

SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", "site-packages", ".git"}
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
WRITE_CALLS = {"add", "add_all", "delete", "commit", "merge", "flush"}


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


_EXCEPT_RE = re.compile(r"^(\s*)except\s+([^():\n]+?,[^():\n]+?)\s*:", re.M)


def parse_source(text: str):
    """Parse Python source. Retries with Python 3.14-style 'except A, B:' rewritten."""
    try:
        return ast.parse(text)
    except SyntaxError:
        return ast.parse(_EXCEPT_RE.sub(r"\1except (\2):", text))


def url_pattern(arg):
    """Turn a string / f-string argument into 'text{}text' (None if not a string)."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        out = ""
        for v in arg.values:
            if isinstance(v, ast.Constant):
                out += str(v.value)
            else:
                out += "{}"
        return out
    return None


class Collector(ast.NodeVisitor):
    """Walks one file and records definitions, imports, calls, tables, endpoints."""

    def __init__(self, mod: str, rel_path: str, is_pkg: bool):
        self.mod = mod
        self.rel_path = rel_path
        self.is_pkg = is_pkg
        self.stack = []
        self.defs = {}         # full name -> (start, end, kind)
        self.imports = {}      # local name -> full dotted name
        self.calls = []        # (caller, called name, kind, base name)
        self.refs = []         # (caller, name used)  -> for table access
        self.attr_calls = {}   # caller -> set of attribute names called
        self.tables = set()    # full names of table classes
        self.routers = {}      # router variable -> prefix
        self.endpoints = []    # (handler full name, METHOD, path, line)
        self.aliases = {}      # module-level X = Annotated[..., Depends(fn)] -> fn name
        self.api_calls = []    # (caller, METHOD, url pattern)

    # ---------- imports ----------
    def _resolve_from(self, node) -> str:
        base = node.module or ""
        if node.level:
            parts = self.mod.split(".")
            if not self.is_pkg:
                parts = parts[:-1]
            parts = parts[: max(0, len(parts) - (node.level - 1))]
            pkg = ".".join(parts)
            base = f"{pkg}.{base}".strip(".") if base else pkg
        return base

    def visit_Import(self, node):
        for a in node.names:
            if a.asname:
                self.imports[a.asname] = a.name
            else:
                self.imports[a.name.split(".")[0]] = a.name.split(".")[0]

    def visit_ImportFrom(self, node):
        base = self._resolve_from(node)
        for a in node.names:
            self.imports[a.asname or a.name] = f"{base}.{a.name}".strip(".")

    # ---------- module-level assignments (routers, Depends aliases) ----------
    def _module_assign(self, target, value):
        if self.stack or not isinstance(target, ast.Name) or value is None:
            return
        if isinstance(value, ast.Call):
            fn = value.func
            fname = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if fname in ("APIRouter", "FastAPI"):
                prefix = ""
                for kw in value.keywords:
                    if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                        prefix = kw.value.value
                self.routers[target.id] = prefix
        for sub in ast.walk(value):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == "Depends" and sub.args
                    and isinstance(sub.args[0], ast.Name)):
                self.aliases[target.id] = sub.args[0].id

    def visit_Assign(self, node):
        if node.targets:
            self._module_assign(node.targets[0], node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._module_assign(node.target, node.value)
        self.generic_visit(node)

    # ---------- definitions ----------
    def _is_table(self, node) -> bool:
        for kw in node.keywords:
            if kw.arg == "table" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
        for stmt in node.body:  # SQLAlchemy style: __tablename__ = "..."
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and t.id == "__tablename__":
                        return True
        return False

    def _decorators(self, node, full):
        for dec in node.decorator_list:
            # endpoint decorator: @router.get("/path")
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr in HTTP_METHODS and isinstance(dec.func.value, ast.Name)
                    and dec.func.value.id in self.routers):
                path = ""
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    path = str(dec.args[0].value)
                prefix = self.routers[dec.func.value.id]
                self.endpoints.append((full, dec.func.attr.upper(), prefix + path, node.lineno))
            # dependencies=[Depends(fn)] inside the decorator
            for sub in ast.walk(dec):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id == "Depends" and sub.args
                        and isinstance(sub.args[0], ast.Name)):
                    self.calls.append((full, sub.args[0].id, "name", None))

    def _def(self, node, kind):
        full = f"{self.mod}." + ".".join(self.stack + [node.name])
        self.defs[full] = (node.lineno, node.end_lineno, kind)
        if kind == "function":
            self._decorators(node, full)
        if kind == "class" and self._is_table(node):
            self.tables.add(full)
        self.stack.append(node.name)
        if kind == "function":
            self.visit(node.args)              # annotations and defaults
            if node.returns:
                self.visit(node.returns)
        for child in node.body:
            self.visit(child)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._def(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._def(node, "class")

    # ---------- usage ----------
    def visit_Name(self, node):
        if self.stack and isinstance(node.ctx, ast.Load):
            self.refs.append((f"{self.mod}." + ".".join(self.stack), node.id))

    def visit_Call(self, node):
        if self.stack:
            caller = f"{self.mod}." + ".".join(self.stack)
            f = node.func
            if isinstance(f, ast.Name):
                if f.id == "Depends" and node.args and isinstance(node.args[0], ast.Name):
                    self.calls.append((caller, node.args[0].id, "name", None))
                else:
                    self.calls.append((caller, f.id, "name", None))
            elif isinstance(f, ast.Attribute):
                base = f.value.id if isinstance(f.value, ast.Name) else None
                self.calls.append((caller, f.attr, "attr", base))
                self.attr_calls.setdefault(caller, set()).add(f.attr)
                # test-style HTTP call: client.get("/users/")
                if f.attr in HTTP_METHODS and node.args:
                    pat = url_pattern(node.args[0])
                    if pat and (pat.startswith("/") or pat.startswith("{}/")):
                        self.api_calls.append((caller, f.attr.upper(), pat))
        self.generic_visit(node)


def endpoint_regex(path: str):
    segs = [("[^/]+" if (s.startswith("{") and s.endswith("}")) else re.escape(s))
            for s in path.strip("/").split("/")]
    return re.compile("^/" + "/".join(segs) + "/?$"), sum(
        1 for s in path.split("/") if s.startswith("{"))


def build(repo: str) -> nx.DiGraph:
    root = Path(repo).resolve()
    G = nx.DiGraph()
    G.graph["skipped"] = []
    collectors = []

    # Pass 1: definitions, tables, endpoints
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            tree = parse_source(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as e:
            G.graph["skipped"].append(f"{p.relative_to(root)} ({type(e).__name__})")
            continue
        c = Collector(module_name(p, root), str(p.relative_to(root)), p.name == "__init__.py")
        c.visit(tree)
        for name, (start, end, kind) in c.defs.items():
            G.add_node(name, file=c.rel_path, start=start, end=end,
                       kind="table" if name in c.tables else kind)
        for handler, method, path, line in c.endpoints:
            ep = f"endpoint:{method} {path}"
            G.add_node(ep, file=c.rel_path, start=line, end=line, kind="endpoint",
                       method=method, path=path)
            G.add_edge(ep, handler, type="EXPOSES", confidence="high")
        collectors.append(c)

    nodes = list(G.nodes)

    # Depends aliases (SessionDep -> get_db), resolved globally
    alias_map = {}
    for c in collectors:
        for alias, dep in c.aliases.items():
            target = c.imports.get(dep, f"{c.mod}.{dep}")
            if target in G:
                alias_map[f"{c.mod}.{alias}"] = target

    # Pass 2: calls
    for c in collectors:
        for caller, called, kind, base in c.calls:
            targets, confidence = [], "high"
            if kind == "name":
                if called in c.imports:
                    full = c.imports[called]
                    if full in G:
                        targets = [full]
                    else:
                        suffix = ".".join(full.split(".")[-2:])
                        targets = [n for n in nodes if n.endswith("." + suffix) or n == suffix]
                        confidence = "low"
                else:
                    local = f"{c.mod}.{called}"
                    targets = [local] if local in G else []
            else:
                resolved = None
                if base and base in c.imports:
                    cand = f"{c.imports[base]}.{called}"
                    if cand in G:
                        resolved = cand
                if resolved:
                    targets = [resolved]
                else:
                    targets = [n for n in nodes
                               if n.endswith("." + called) and G.nodes[n]["kind"] != "endpoint"]
                    confidence = "low"
                    if len(targets) > 3:
                        targets = []
            for t in targets:
                if t == caller:
                    continue
                if G.has_edge(caller, t) and G[caller][t].get("confidence") == "high":
                    continue
                G.add_edge(caller, t, type="CALLS", confidence=confidence)

    # Pass 3: table access (READS / WRITES) and Depends aliases
    for c in collectors:
        accessed = {}
        for caller, name in c.refs:
            if caller not in G or G.nodes[caller]["kind"] != "function":
                continue
            full = c.imports.get(name, f"{c.mod}.{name}")
            if full in alias_map:
                G.add_edge(caller, alias_map[full], type="DEPENDS", confidence="high")
            elif full in G and G.nodes[full]["kind"] == "table":
                accessed.setdefault(caller, set()).add(full)
        for caller, tbls in accessed.items():
            writes = bool(c.attr_calls.get(caller, set()) & WRITE_CALLS)
            for t in tbls:
                G.add_edge(caller, t, type="WRITES" if writes else "READS", confidence="high")

    # Pass 4: tests that call an API URL -> endpoint
    eps = [(n, *endpoint_regex(d["path"]), d["method"])
           for n, d in G.nodes(data=True) if d.get("kind") == "endpoint"]
    for c in collectors:
        for caller, method, pat in c.api_calls:
            if caller not in G:
                continue
            url = pat.split("?")[0]
            if url.startswith("{}/"):
                url = url[2:]                       # leading {settings.API_V1_STR}
            url = re.sub(r"^/api/v\d+", "", url).replace("{}", "X")
            hits = [(n, params) for n, rx, params, m in eps if m == method and rx.match(url)]
            if not hits:
                continue
            best = min(p for _, p in hits)
            best_hits = [n for n, p in hits if p == best]
            for n in best_hits:
                G.add_edge(caller, n, type="CALLS_API",
                           confidence="high" if len(best_hits) == 1 else "low")
    return G


def label(G, n):
    d = G.nodes[n]
    if d["kind"] == "endpoint":
        return "endpoint"
    if d["kind"] == "table":
        return "table"
    f = d.get("file", "").replace("\\", "/")
    if f.startswith("tests/") and n.split(".")[-1].startswith("test_"):
        return "test"
    return "other code"


def impact(G, query, max_depth=4, high_only=False):
    exact = [n for n in G.nodes if n == query or n.endswith("." + query)]
    matches = exact or [n for n in G.nodes if query in n]
    if not matches:
        print("No node matches:", query)
        return

    if high_only:
        keep = [(u, v) for u, v, d in G.edges(data=True) if d.get("confidence") == "high"]
        H = G.edge_subgraph(keep)
    else:
        H = G

    print("Legend: '->' confirmed link, '~>' guessed link (matched by name only)")

    for m in matches:
        if m not in H:
            print(f"\n{m} has no usable edges with these settings")
            continue
        dist = nx.single_source_shortest_path_length(H.reverse(copy=False), m, cutoff=max_depth)
        dist.pop(m)
        print(f"\nChanging {m} ({G.nodes[m]['file']}:{G.nodes[m]['start']}) [{G.nodes[m]['kind']}]")
        counts = {}
        for n in dist:
            counts[label(G, n)] = counts.get(label(G, n), 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing"
        print(f"  {len(dist)} nodes within {max_depth} hops: {summary}")
        for node, d in sorted(dist.items(), key=lambda x: (x[1], label(G, x[0]), x[0])):
            path = nx.shortest_path(H, node, m)
            text = path[0].split(".")[-1]
            for a, b in zip(path, path[1:]):
                arrow = " -> " if H[a][b].get("confidence") == "high" else " ~> "
                text += arrow + b.split(".")[-1]
            print(f"   [{d}] ({label(G, node)}) {node}")
            print(f"        path: {text}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        G = build(sys.argv[2])
        with open("graph.pkl", "wb") as f:
            pickle.dump(G, f)
        kinds, etypes = {}, {}
        for _, d in G.nodes(data=True):
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        for _, _, d in G.edges(data=True):
            etypes[d["type"]] = etypes.get(d["type"], 0) + 1
        high = sum(1 for _, _, d in G.edges(data=True) if d.get("confidence") == "high")
        print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
              f"({high} confirmed, {G.number_of_edges() - high} guessed)")
        print("Nodes:", kinds)
        print("Edges:", etypes)
        if G.graph.get("skipped"):
            print("WARNING - files that could not be parsed (not in the graph):")
            for s in G.graph["skipped"]:
                print("   ", s)
    elif cmd == "impact":
        with open("graph.pkl", "rb") as f:
            G = pickle.load(f)
        depth = 4
        if "--depth" in sys.argv:
            depth = int(sys.argv[sys.argv.index("--depth") + 1])
        impact(G, sys.argv[2], max_depth=depth, high_only="--high" in sys.argv)