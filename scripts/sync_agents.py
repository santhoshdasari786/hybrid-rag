"""Generate Claude Code subagents (.claude/agents) from the Copilot custom agents (.github/agents).

    uv run python scripts/sync_agents.py

The body is shared; only the frontmatter differs (Claude Code: name, description, tools).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".github" / "agents"
TARGET = ROOT / ".claude" / "agents"
READ_ONLY = {"api-reviewer": "Read, Grep, Glob, Bash"}
FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for path in sorted(SOURCE.glob("*.agent.md")):
        text = path.read_text(encoding="utf-8")
        match = FRONTMATTER.match(text)
        if match is None:
            raise SystemExit(f"{path} has no frontmatter")
        meta = dict(re.findall(r"^(name|description):\s*(.+)$", match.group(1), re.MULTILINE))
        name = meta.get("name", path.name.removesuffix(".agent.md"))
        lines = ["---", f"name: {name}", f"description: {meta['description']}"]
        if name in READ_ONLY:
            lines.append(f"tools: {READ_ONLY[name]}")
        lines.append("---")
        body = text[match.end() :].lstrip("\n")
        header = (
            "<!-- Generated from .github/agents by scripts/sync_agents.py — edit the source, not this file. -->\n\n"
        )
        (TARGET / f"{name}.md").write_text("\n".join(lines) + "\n\n" + header + body, encoding="utf-8")
        print(f"wrote .claude/agents/{name}.md")


if __name__ == "__main__":
    main()
