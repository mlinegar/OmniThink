"""Minimal offline manifesto workflow demonstrating the OmniThink extension.

This script assembles a toy information tree, canonicalizes redundant nodes,
exports the contracted summary graph, labels each canonical with the provided
manifesto schema, and finally samples a few nodes for manual inspection.  It
is intentionally self-contained so that we can exercise the extension without
calling external APIs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omnithink_ext.canon import Canonicalizer
from omnithink_ext.graph import GraphExporter
from omnithink_ext.labeler import Labeler
from omnithink_ext.state import Evidence, InfoTree, Node


# ---------------------------------------------------------------------------
# Toy manifesto content
# ---------------------------------------------------------------------------


@dataclass
class Plank:
    id: str
    title: str
    query: str
    claims: Sequence[str]
    sentence: str
    evidence: Dict[str, str | int]


@dataclass
class Section:
    id: str
    title: str
    query: str
    summary: str
    claims: Sequence[str]
    planks: Sequence[Plank]


MANIFESTO_SECTIONS: Sequence[Section] = (
    Section(
        id="economy",
        title="Rebuilding the Economy",
        query="prosperity party jobs and growth plan",
        summary="Invest in infrastructure and small businesses to create good jobs nationwide.",
        claims=(
            "Invest in modern infrastructure and support small businesses to create good jobs.",
            "Prioritise apprenticeships and resilient supply chains so no region is left behind.",
        ),
        planks=(
            Plank(
                id="infrastructure_bank",
                title="National Infrastructure Bank",
                query="national infrastructure bank funding plan",
                claims=(
                    "Launch a National Infrastructure Bank to finance rail upgrades and broadband expansion in every region.",
                ),
                sentence="We will launch a National Infrastructure Bank to finance clean rail upgrades and broadband expansion that create one million jobs.",
                evidence={
                    "url": "https://example.org/infrastructure-bank",
                    "title": "Prosperity Party infrastructure briefing",
                    "snippet": "The manifesto pledges a National Infrastructure Bank focused on clean transport and broadband.",
                    "span": (0, 122),
                    "passage_hash": "ev_infra_bank",
                },
            ),
            Plank(
                id="small_business_relief",
                title="Small Business Tax Relief",
                query="small business tax relief apprenticeships",
                claims=(
                    "Offer targeted tax credits for manufacturers that hire apprentices and raise wages.",
                ),
                sentence="Targeted tax relief will reward manufacturers that hire apprentices and raise wages in local supply chains.",
                evidence={
                    "url": "https://example.org/smb-tax",
                    "title": "Tax Relief Factsheet",
                    "snippet": "Credits will favour firms that hire apprentices and make goods domestically.",
                    "span": (0, 118),
                    "passage_hash": "ev_smb_relief",
                },
            ),
        ),
    ),
    Section(
        id="climate",
        title="Clean Energy Future",
        query="clean energy jobs manifesto",
        summary="Cut carbon pollution by half by 2030 through large-scale renewable projects.",
        claims=(
            "Cut national emissions in half by 2030 with investments in renewable power and efficiency.",
        ),
        planks=(
            Plank(
                id="offshore_wind",
                title="Offshore Wind Build-out",
                query="offshore wind grid connections",
                claims=(
                    "Deploy fifteen gigawatts of offshore wind and modernise grid connections to bring renewable jobs to coastal towns.",
                ),
                sentence="Deploying new offshore wind farms will deliver renewable jobs while driving down carbon emissions.",
                evidence={
                    "url": "https://example.org/offshore-wind",
                    "title": "Clean Energy Programme",
                    "snippet": "Investments connect offshore wind to the national grid and train local technicians.",
                    "span": (0, 128),
                    "passage_hash": "ev_wind",
                },
            ),
            Plank(
                id="home_retrofit",
                title="Home Retrofit Programme",
                query="national home retrofit heat pump",
                claims=(
                    "Fund zero-interest loans for home insulation and heat pumps to lower household emissions.",
                ),
                sentence="Zero-interest retrofit loans will help families cut carbon emissions while reducing energy bills.",
                evidence={
                    "url": "https://example.org/home-retrofit",
                    "title": "Warm Homes Initiative",
                    "snippet": "Households receive support to install insulation, heat pumps, and smart meters.",
                    "span": (0, 116),
                    "passage_hash": "ev_retrofit",
                },
            ),
        ),
    ),
    Section(
        id="healthcare",
        title="Health and Care for Everyone",
        query="universal healthcare prevention manifesto",
        summary="Guarantee affordable healthcare and expand community mental health support.",
        claims=(
            "Guarantee universal affordable healthcare by expanding community clinics and preventive services.",
        ),
        planks=(
            Plank(
                id="community_clinics",
                title="Expanded Community Clinics",
                query="community health clinic expansion",
                claims=(
                    "Open two hundred and fifty community clinics with bilingual staff and extended hours.",
                ),
                sentence="New community clinics will provide affordable healthcare close to home with bilingual nurses.",
                evidence={
                    "url": "https://example.org/community-clinics",
                    "title": "Community Health Expansion",
                    "snippet": "Clinics will offer primary care, vaccinations, and preventative screenings.",
                    "span": (0, 124),
                    "passage_hash": "ev_clinics",
                },
            ),
            Plank(
                id="mental_health",
                title="Mental Health Support",
                query="mental health telehealth counsellors",
                claims=(
                    "Place counsellors in every secondary school and expand telehealth therapy services.",
                ),
                sentence="Every secondary school will host a mental health counsellor and expanded telehealth support.",
                evidence={
                    "url": "https://example.org/mental-health",
                    "title": "Mental Health Guarantee",
                    "snippet": "Students and families gain rapid access to counselling and telehealth appointments.",
                    "span": (0, 121),
                    "passage_hash": "ev_mental",
                },
            ),
        ),
    ),
    Section(
        id="education",
        title="Opportunity through Education",
        query="education apprenticeships lifelong learning manifesto",
        summary="Invest in world-class schools, apprenticeships, and lifelong learning so everyone can thrive.",
        claims=(
            "Invest in modern classrooms, apprenticeships, and digital skills so every learner can thrive.",
        ),
        planks=(
            Plank(
                id="modern_classrooms",
                title="Modern Classrooms Fund",
                query="school modernisation digital classrooms",
                claims=(
                    "Upgrade every secondary school with modern science labs and digital learning devices.",
                ),
                sentence="A Modern Classrooms Fund will deliver new science labs and digital devices to students.",
                evidence={
                    "url": "https://example.org/modern-classrooms",
                    "title": "Education Infrastructure Plan",
                    "snippet": "Capital grants renew labs, libraries, and digital learning spaces across the country.",
                    "span": (0, 123),
                    "passage_hash": "ev_classrooms",
                },
            ),
            Plank(
                id="apprenticeships",
                title="Apprenticeship Guarantee",
                query="apprenticeship guarantee training allowance",
                claims=(
                    "Guarantee a paid apprenticeship or training place for every young person under twenty-five.",
                ),
                sentence="Every young person will access a paid apprenticeship or training place with industry mentors.",
                evidence={
                    "url": "https://example.org/apprenticeship-guarantee",
                    "title": "Skills for the Future",
                    "snippet": "Partnerships with employers provide paid training pathways into growing sectors.",
                    "span": (0, 118),
                    "passage_hash": "ev_apprenticeships",
                },
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Helper utilities for the offline demo
# ---------------------------------------------------------------------------


class BagOfWordsEmbedder:
    """A tiny deterministic embedder used by the canonicalizer."""

    def __call__(self, text: str) -> Dict[str, float]:
        tokens = [tok.lower() for tok in text.split() if tok.strip()]
        counts: Dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        return counts


class DummyInsight:
    def __init__(self, support_node_ids: Sequence[str], status: str = "fact") -> None:
        self.support_node_ids = list(support_node_ids)
        self.status = status


class DummyPool:
    def __init__(self, insights: Sequence[DummyInsight]) -> None:
        self.insights = list(insights)


def build_tree() -> tuple[InfoTree, str, Dict[str, List[str]]]:
    """Populate an InfoTree with the toy manifesto data."""

    tree = InfoTree()
    root = Node(
        id="root",
        parent=None,
        title="Prosperity Party Manifesto 2025",
        query="prosperity party manifesto",
        claims=["A manifesto for inclusive prosperity, clean energy, and care."],
        node_type="topic",
        depth=0,
    )
    tree.nodes[root.id] = root

    sentence_bucket: Dict[str, List[str]] = {root.id: [
        "The Prosperity Party manifesto charts a course for inclusive prosperity across every community.",
    ]}

    evidence_counter = 1

    for section in MANIFESTO_SECTIONS:
        section_id = f"sec_{section.id}"
        section_node = Node(
            id=section_id,
            parent=root.id,
            title=section.title,
            query=section.query,
            claims=list(section.claims),
            node_type="concept",
            depth=1,
        )
        tree.add_child(root.id, section_node, kind="expands")
        sentence_bucket[section_id] = [section.summary]

        for plank in section.planks:
            node_id = f"{section_id}_{plank.id}"
            node = Node(
                id=node_id,
                parent=section_id,
                title=plank.title,
                query=plank.query,
                claims=list(plank.claims),
                node_type="concept",
                depth=2,
            )
            tree.add_child(section_id, node, kind="expands")
            sentence_bucket[node_id] = [plank.sentence]

            ev = Evidence(
                id=evidence_counter,
                url=str(plank.evidence["url"]),
                title=str(plank.evidence["title"]),
                snippet=str(plank.evidence["snippet"]),
                span=tuple(plank.evidence["span"]),
                passage_hash=str(plank.evidence["passage_hash"]),
            )
            tree.attach_evidence(node_id, ev)
            evidence_counter += 1

    return tree, root.id, sentence_bucket


def canonicalise_tree(tree: InfoTree, root_id: str) -> Canonicalizer:
    embedder = BagOfWordsEmbedder()
    canon = Canonicalizer(embedder=embedder, lambda_=0.55, theta=0.4, ngram_n=2)
    for node_id in list(tree.nodes.keys()):
        canon.find(node_id)
        if node_id != root_id:
            canon.update_with_new_node(tree, node_id)
    return canon


def export_graph(tree: InfoTree, canon: Canonicalizer) -> tuple[Dict[str, Dict], List[tuple[str, str, str]]]:
    insights = [
        DummyInsight(["sec_economy_infrastructure_bank", "sec_climate_offshore_wind"], status="fact"),
        DummyInsight(["sec_economy_small_business_relief", "sec_education_apprenticeships"], status="fact"),
        DummyInsight(["sec_healthcare_mental_health", "sec_healthcare_community_clinics"], status="fact"),
    ]
    pool = DummyPool(insights)
    exporter = GraphExporter(canon, pool)
    return exporter.export(tree)


def load_manifesto_schema() -> Labeler:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "label_schemas" / "manifesto.example.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return Labeler(
        schema=cfg["labels"],
        lexicon=cfg.get("lexicon", {}),
        rel_weights=cfg.get("relation_weights", {}),
        alpha=cfg.get("scoring", {}).get("alpha", 1.0),
        beta=cfg.get("scoring", {}).get("beta", 0.6),
        gamma=cfg.get("scoring", {}).get("gamma", 0.3),
    )


def make_sentence_index(canon: Canonicalizer, bucket: Dict[str, List[str]]) -> Dict[str, List[str] | str]:
    counter = 1
    index: Dict[str, List[str] | str] = {}
    for node_id, sentences in bucket.items():
        c_id = canon.find(node_id)
        slot = index.setdefault(c_id, [])
        assert isinstance(slot, list)
        for sentence in sentences:
            sid = f"s{counter}"
            counter += 1
            slot.append(sid)
            index[sid] = sentence
    return index


def build_canonical_tree(
    tree: InfoTree, canon: Canonicalizer, nodes: Dict[str, Dict], edges: List[tuple[str, str, str]]
) -> Dict[str, dict]:
    tree_edges = [e for e in edges if e[2] not in {"supports", "contradicts"}]
    children: Dict[str, List[tuple[str, str]]] = {}
    for parent, child, kind in tree_edges:
        children.setdefault(parent, []).append((child, kind))

    root_candidates = [
        cid
        for cid, data in nodes.items()
        if any(tree.nodes[mid].depth == 0 for mid in data["members"])
    ]
    root_id = root_candidates[0] if root_candidates else next(iter(nodes))

    def build_node(c_id: str) -> dict:
        label = tree.nodes[nodes[c_id]["members"][0]].title
        payload = {
            "id": c_id,
            "label": label,
            "claims": nodes[c_id]["claims"],
            "members": nodes[c_id]["members"],
            "children": [],
        }
        for child_id, kind in sorted(children.get(c_id, []), key=lambda x: tree.nodes[nodes[x[0]]["members"][0]].title):
            child_payload = build_node(child_id)
            child_payload["relation"] = kind
            payload["children"].append(child_payload)
        return payload

    return build_node(root_id)


def sample_nodes(
    tree: InfoTree,
    nodes: Dict[str, Dict],
    labels: Dict[str, Dict[str, float]],
    sample_size: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    available = list(nodes.keys())
    if sample_size > len(available):
        sample_size = len(available)
    chosen = rng.sample(available, sample_size)
    samples = []
    for c_id in chosen:
        members = nodes[c_id]["members"]
        title = tree.nodes[members[0]].title
        dist = labels.get(c_id, {})
        top_label = max(dist, key=dist.get) if dist else None
        samples.append(
            {
                "id": c_id,
                "title": title,
                "top_label": top_label,
                "label_distribution": dist,
                "claims": nodes[c_id]["claims"],
                "members": members,
            }
        )
    return samples


def render_canonical_tree(node: dict, indent: int = 0) -> List[str]:
    pad = "  " * indent
    relation = node.get("relation")
    rel_text = f" ({relation})" if relation else ""
    lines = [f"{pad}- [{node['id']}] {node['label']}{rel_text}"]
    for claim in node.get("claims", []):
        lines.append(f"{pad}    • {claim}")
    for child in node.get("children", []):
        lines.extend(render_canonical_tree(child, indent + 1))
    return lines


def render_samples(samples: Sequence[dict]) -> List[str]:
    lines: List[str] = []
    for sample in samples:
        lines.append(f"[{sample['id']}] {sample['title']}")
        if sample.get("top_label"):
            lines.append(f"  top label: {sample['top_label']}")
        dist = sample.get("label_distribution", {})
        if dist:
            lines.append("  distribution:")
            for label, score in sorted(dist.items(), key=lambda x: -x[1]):
                lines.append(f"    - {label}: {score:.3f}")
        claims = sample.get("claims", [])
        if claims:
            lines.append("  claims:")
            for claim in claims:
                lines.append(f"    • {claim}")
        lines.append("")
    return lines


def format_report(canonical_tree: dict, samples: Sequence[dict]) -> str:
    sections: List[str] = ["=== Canonical summary tree ==="]
    sections.extend(render_canonical_tree(canonical_tree))
    sections.append("")
    sections.append("=== Random samples ===")
    sections.extend(render_samples(samples))
    report = "\n".join(sections)
    if not report.endswith("\n"):
        report += "\n"
    return report


def write_log(report: str, log_dir: Optional[Path]) -> Optional[Path]:
    if log_dir is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"manifesto_example_{timestamp}.log"
    log_path.write_text(report, encoding="utf-8")
    return log_path


def main(args: argparse.Namespace) -> None:
    tree, root_id, sentence_bucket = build_tree()
    canon = canonicalise_tree(tree, root_id)
    canonical_nodes, canonical_edges = export_graph(tree, canon)
    canonical_tree = build_canonical_tree(tree, canon, canonical_nodes, canonical_edges)

    labeler = load_manifesto_schema()
    sentence_index = make_sentence_index(canon, sentence_bucket)
    labels = {
        c_id: labeler.score_node(canonical_nodes, canonical_edges, c_id, sentence_index, {})
        for c_id in canonical_nodes
    }

    samples = sample_nodes(tree, canonical_nodes, labels, args.sample_size, args.seed)

    if args.json:
        payload = {
            "summary_tree": canonical_tree,
            "labels": labels,
            "samples": samples,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = format_report(canonical_tree, samples)
    print(report, end="")

    log_dir: Optional[Path]
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser()
    else:
        log_dir = None
    log_path = write_log(report, log_dir)
    if log_path is not None:
        print(f"Wrote run log to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the offline manifesto summary example.")
    parser.add_argument("--sample-size", type=int, default=3, help="Number of canonical nodes to sample.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    parser.add_argument("--json", type=str, help="Optional path to write the summary tree as JSON.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(REPO_ROOT / "logs"),
        help=(
            "Directory to write a timestamped run log. "
            "Set to an empty string to skip writing logs."
        ),
    )
    main(parser.parse_args())

