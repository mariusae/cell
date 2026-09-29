"""Graphs as Mermaid flowcharts.

Nodes are labeled like the IR listing (Graph.format), so the two views can
be read side by side. Edges:

- data inputs: solid arrows;
- path edges (DESIGN §4.3): dotted, labeled "path";
- effect edges: thick, labeled "effect";
- issue-order (after) edges: dotted, labeled "after".

Nodes of an inlined callee are grouped in a subgraph named for its call.
"""

from __future__ import annotations

from cell.graph import Graph, Node, _format_node

_CLASSES = {
    "param": "fill:#e6f4ea,stroke:#34a853,color:#1e4620",
    "pure": "fill:#e8f0fe,stroke:#4285f4,color:#174ea6",
    "effectful": "fill:#fef3e0,stroke:#f29900,color:#7a4a00",
    "op": "fill:#f1f3f4,stroke:#9aa0a6,color:#202124",
    "pack": "fill:#f1f3f4,stroke:#9aa0a6,color:#202124",
    "source": "fill:#e6f4ea,stroke:#34a853,color:#1e4620",
    "guard": "fill:#fff8c5,stroke:#d4a72c,color:#5a4500",
    "deopt": "fill:#fce8e6,stroke:#d93025,color:#8c1d18",
    "return": "fill:#e6f4ea,stroke:#34a853,color:#1e4620",
    "scope": "fill:#f3e8fd,stroke:#a142f4,color:#4a148c",
}


def _escape(text: str) -> str:
    for a, b in (("&", "&amp;"), ('"', "#quot;"), ("<", "&lt;"), (">", "&gt;"), ("|", "&#124;")):
        text = text.replace(a, b)
    return text


def _class(n: Node) -> str:
    if n.kind in ("call", "enter"):
        return "effectful" if n.attrs["effectful"] else "pure"
    if n.kind == "exit":
        return "scope"
    return n.kind


def to_mermaid(graph: Graph) -> str:
    lines = ["flowchart TD"]
    scoped: dict[tuple[int, ...], list[Node]] = {}
    for n in graph.nodes:
        scoped.setdefault(n.scope, []).append(n)  # an enter node belongs to its caller's scope

    def node_line(n: Node) -> str:
        lhs, tags, _ = _format_node(graph, n)
        label = _escape(lhs if len(lhs) <= 70 else lhs[:69] + "…")
        if tags:
            label += "<br/><small>" + _escape(tags) + "</small>"
        shape = ("{{", "}}") if n.kind == "guard" else ("([", "])") if n.kind in ("param", "return") else ("[", "]")
        return f'  n{n.id}{shape[0]}"{label}"{shape[1]}:::{_class(n)}'

    for scope, nodes in scoped.items():
        if not scope:
            lines += [node_line(n) for n in nodes]
            continue
        callee = next((m.attrs["cell"] for m in nodes if m.kind == "exit"), "callee").rsplit(".", 1)[-1]
        lines.append(f'  subgraph s{"_".join(map(str, scope))}["{callee} at seq {".".join(map(str, scope))}"]')
        lines += ["  " + node_line(n) for n in nodes]
        lines.append("  end")

    for n in graph.nodes:
        for i in n.inputs:
            if isinstance(i, int):
                lines.append(f"  n{i} --> n{n.id}")
        for p in n.path:
            lines.append(f"  n{p} -.->|path| n{n.id}")
        for e in n.effect:
            lines.append(f"  n{e} ==>|effect| n{n.id}")
        if n.after is not None:
            lines.append(f"  n{n.after} -.->|after| n{n.id}")

    for name, style in _CLASSES.items():
        lines.append(f"  classDef {name} {style}")
    return "\n".join(lines)
