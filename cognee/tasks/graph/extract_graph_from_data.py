import asyncio
from typing import Type, List, Optional

from pydantic import BaseModel

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.storage import add_data_points
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge

logger = get_logger("cognify")

# The user-provided prompt for the LLM
CONFLICT_DETECTION_PROMPT = """
You are a conflict detection engine. Identify all pairs of rule nodes that have **explicit, direct conflicts** in the given graph nodes and create a relationship labeled **_only_ as**:  
`conflicted_by`  

---

## Conflict Criteria — A `conflicted_by` relationship should be created **only if all** the following are met:

1. **Identical Contexts:**  
   The rules operate in the *same activity, domain, and timeframe*. Context includes any overlap or equivalence in subjects, locations, or conditions relevant to the rules.  
   - Prefer explicit edges or metadata that establish subject or contextual equivalences or overlaps.  
   - If explicit links are missing, use strong semantic similarity or equivalence between node descriptions or properties as supporting evidence, but require multiple corroborating signals.

2. **Mutually Exclusive Actions:**  
   The rules mandate *incompatible requirements* (e.g., `must do X` vs `must not do X`, or conflicting quantitative/temporal demands).

---

## Do **NOT** create a conflict if any of the following apply:

- One rule is a **contextual override** (e.g., emergency protocol superseding standard rule).  
- There is only **partial scope overlap** without direct equivalence (e.g., "all employees" vs. "night shift employees").  
- The rules apply to **different timeframes** or non-overlapping conditions.  
- The rules are **identical** — a rule cannot conflict with itself.

---

## Important Implementation Notes:

- Do **not** require identical subject node IDs; conflicts should be detected if the subject concepts meaningfully overlap or are equivalent in context.  
- Prefer conflicts grounded in **explicit evidence** from graph edges, node properties, or metadata.  
- In cases of sparse graph structure, use semantic similarity of node descriptions cautiously and only if strongly supported.  
- Each conflict edge must be named exactly `conflicted_by`—no synonyms or alternative labels allowed.  
- Avoid false positives by requiring strict mutual exclusivity and context matching.

---

## Output:

- List all detected `conflicted_by` relationships as edges between Rule nodes, specifying the direction from the contradicted rule to the contradicting rule.  
- Provide concise conflict type labels (e.g., `temporal_conflict`, `direct_contradiction`, `quantitative_conflict`) and optional human-readable conflict notes explaining the conflict.

"""

def extract_nodes(document_chunks):
    nodes = []
    for chunk in document_chunks:
        node_set_names = [ns.name for ns in chunk.belongs_to_set] if hasattr(chunk, "belongs_to_set") else []
        for entity in getattr(chunk, "contains", []):
            node = {
                "id": str(entity.id),
                "name": entity.name,
                "type": entity.is_a.name if hasattr(entity, "is_a") else None,
                "description": getattr(entity, "description", ""),
                "node_set": node_set_names[0] if node_set_names else None,
            }
            nodes.append(node)
    return nodes

async def _handle_rule_conflicts(
    graph_nodes: List[BaseModel], llm_prompt_template: str, graph_model: Type[BaseModel]
):
    """Identifies conflicts between Rule nodes using an LLM."""
    llm_client = get_llm_client()
    conflicted_edges = []
    # prompt = BULK_RULE_CONFLICT_PROMPT.format(
    #     rules_json=json.dumps(rules, indent=2)
    # )

    response = await llm_client.acreate_structured_output(graph_nodes,llm_prompt_template, graph_model)
    logger.info("response ::: %s", response)
    logger.info("response is edges ::: %s", response.edges)
    for edge in response.edges:
        conflicted_edges.append(
            (
                edge.source_node_id,
                edge.target_node_id,
                edge.relationship_name,
                dict(
                    relationship_name=edge.relationship_name,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    ontology_valid=False
                )
            )
        )
    return conflicted_edges

async def integrate_chunk_graphs(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list,
    graph_model: Type[BaseModel],
    ontology_adapter: OntologyResolver,
) -> List[DocumentChunk]:
    """Updates DocumentChunk objects, integrates data points and edges into databases."""
    graph_engine = await get_graph_engine()
    rules_for_llm = []

    logger.info("chunk_graphs ::: %s", chunk_graphs)

    if graph_model is not KnowledgeGraph:
        for chunk_index, chunk_graph in enumerate(chunk_graphs):
            data_chunks[chunk_index].contains = chunk_graph

        await add_data_points(chunk_graphs)
        return data_chunks

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
        graph_engine,
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks, chunk_graphs, ontology_adapter, existing_edges_map
    )

    # for data_chunk, graph in zip(data_chunks, chunk_graphs):
    #     if not graph:
    #         continue

    #     for node in graph.nodes:
    #         rules_for_llm.append(node)

    nodes_extracted = extract_nodes(graph_nodes)

    logger.info("nodes_extracted ::: %s", nodes_extracted)
    logger.info("graph_nodes ::: %s", graph_nodes)
    logger.info("graph_edges ::: %s", graph_edges)

    # --- NEW LOGIC FOR CONFLICT DETECTION ---
    # This logic is applied only when processing complex knowledge graphs.
    conflict_edges = await _handle_rule_conflicts(nodes_extracted, CONFLICT_DETECTION_PROMPT, graph_model)
    logger.info("conflict_edges ::: %s", conflict_edges)
    if conflict_edges:
        graph_edges.extend(conflict_edges)
    # --- END OF NEW LOGIC ---

    if len(graph_nodes) > 0:
        await add_data_points(graph_nodes)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return data_chunks


async def extract_graph_from_data(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    ontology_adapter: OntologyResolver = None,
) -> List[DocumentChunk]:
    """
    Extracts and integrates a knowledge graph from the text content of document chunks using a specified graph model.
    """
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    # Note: Filter edges with missing source or target nodes
    if graph_model == KnowledgeGraph:
        for graph in chunk_graphs:
            valid_node_ids = {node.id for node in graph.nodes}
            graph.edges = [
                edge
                for edge in graph.edges
                if edge.source_node_id in valid_node_ids and edge.target_node_id in valid_node_ids
            ]

    return await integrate_chunk_graphs(
        data_chunks, chunk_graphs, graph_model, ontology_adapter or OntologyResolver()
    )
