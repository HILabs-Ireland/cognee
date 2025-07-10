import asyncio
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.data_models import KnowledgeGraph, Edge as KGEdge
from cognee.shared.logging_utils import get_logger

logger = get_logger("detect_and_update_conflicts")


class ConflictEdge(BaseModel):
    """Represents a detected conflict between two nodes."""
    source_node_id: str
    target_node_id: str
    relationship_name: str = "conflicted_by"
    conflict_type: str
    confidence_score: float
    conflict_description: str

class ConflictDetectionResult(BaseModel):
    """Result of conflict detection analysis."""
    conflicts: List[ConflictEdge]
    summary: str

CONFLICT_DETECTION_PROMPT_PATH = "conflict_detection_for_graph.txt"


def _is_rule_entity(node_data: Dict) -> bool:
    """
    Check if a node is an entity with entity type 'rule'.

    This function checks various fields to determine if an entity represents a rule:
    - entity_type field (direct or nested)
    - type field as fallback
    - rule-specific properties and content indicators

    Args:
        node_data (Dict): The node data dictionary to check

    Returns:
        bool: True if the entity is a rule entity, False otherwise
    """
    # Check for explicit entity_type field
    entity_type = node_data.get('entity_type', '').lower()
    if entity_type == 'rule':
        return True

    # Check for entity_type in nested structures
    if isinstance(node_data.get('entity_type'), dict):
        nested_type = node_data.get('entity_type', {}).get('value', '').lower()
        if nested_type == 'rule':
            return True

    # Check node type field as fallback
    node_type = node_data.get('type', '').lower()
    if node_type == 'rule':
        return True

    # Check for rule-specific properties that indicate this is a rule entity
    rule_indicators = [
        'rule',           # Direct rule field
        'rule_text',      # Rule text field
        'policy',         # Policy rules
        'constraint',     # Constraint rules
        'requirement'     # Requirement rules
    ]

    # Must have entity-like structure AND rule indicators
    has_name = 'name' in node_data
    has_rule_content = any(indicator in node_data for indicator in rule_indicators)

    # Check if it has rule-like content in description
    description = node_data.get('description', '').lower()
    content = node_data.get('content', '').lower()

    has_rule_description = any(rule_word in description or rule_word in content
                             for rule_word in ['rule', 'policy', 'must', 'shall', 'required', 'mandatory'])

    # Return True only if it's clearly a rule entity
    if has_name and (has_rule_content or has_rule_description):
        return True

    return False


def filter_unified_graph_for_rules(unified_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter the unified graph to only include rule entities and edges between rule entities.

    Args:
        unified_graph (Dict[str, Any]): The original unified graph

    Returns:
        Dict[str, Any]: Filtered unified graph containing only rule entities
    """
    logger.info("Filtering unified graph to include only rule entities")

    # Filter nodes to only include rule entities
    rule_nodes = []
    rule_node_ids = set()

    for node in unified_graph.get("nodes", []):
        if _is_rule_entity(node):
            rule_nodes.append(node)
            rule_node_ids.add(node.get("id", ""))
        else:
            node_name = node.get("name", "Unknown")
            node_type = node.get("type", "Unknown")
            logger.debug(f"Filtering out non-rule entity: {node_name} (Type: {node_type})")

    # Filter edges to only include edges between rule entities
    rule_edges = []
    for edge in unified_graph.get("edges", []):
        source_id = edge.get("source_node_id", "")
        target_id = edge.get("target_node_id", "")

        if source_id in rule_node_ids and target_id in rule_node_ids:
            rule_edges.append(edge)
            logger.debug(f"Including edge between rule entities: {source_id} -> {target_id}")
        else:
            logger.debug(f"Filtering out edge not between rule entities: {source_id} -> {target_id}")

    # Update chunk metadata to only include rule entities
    filtered_chunk_metadata = []
    for chunk_info in unified_graph.get("chunk_metadata", []):
        filtered_entities = []
        for entity in chunk_info.get("entities", []):
            if _is_rule_entity(entity):
                filtered_entities.append(entity)

        chunk_info_copy = chunk_info.copy()
        chunk_info_copy["entities"] = filtered_entities
        filtered_chunk_metadata.append(chunk_info_copy)

    filtered_graph = {
        "nodes": rule_nodes,
        "edges": rule_edges,
        "chunk_metadata": filtered_chunk_metadata,
        "total_nodes": len(rule_nodes),
        "total_edges": len(rule_edges),
        "total_chunks": unified_graph.get("total_chunks", 0)
    }

    logger.info(f"Filtered unified graph: {len(rule_nodes)} rule entities, {len(rule_edges)} edges between rule entities")
    return filtered_graph


async def extract_complete_graph_data(data_chunks: List[DocumentChunk]) -> Dict[str, Any]:
    all_nodes = {}
    all_edges = []
    chunk_metadata = []

    for chunk in data_chunks:
        chunk_info = {
            "chunk_id": str(chunk.id),
            "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            "entities": []
        }

        if hasattr(chunk, 'contains') and chunk.contains:
            entities = getattr(chunk, 'contains', [])
            for entity in entities:
                try:
                    node_id = str(getattr(entity, 'id', ''))
                    entity_name = getattr(entity, 'name', '')

                    entity_type = ''
                    if hasattr(entity, 'is_a'):
                        is_a_attr = getattr(entity, 'is_a', None)
                        if is_a_attr:
                            if hasattr(is_a_attr, 'name'):
                                entity_type = getattr(is_a_attr, 'name', '')
                            else:
                                entity_type = str(is_a_attr)
                    elif hasattr(entity, 'type'):
                        entity_type = getattr(entity, 'type', '')

                    entity_description = getattr(entity, 'description', '')

                    node_data = {
                        "id": node_id,
                        "name": entity_name,
                        "type": entity_type,
                        "entity_type": entity_type,  # Add entity_type field for consistency
                        "description": entity_description,
                        "source_chunk": str(chunk.id)
                    }

                    # Only process rule entities for conflict detection
                    if not _is_rule_entity(node_data):
                        logger.debug(f"Skipping non-rule entity: {entity_name} (Type: {entity_type})")
                        continue

                    logger.debug(f"Including rule entity: {entity_name} (Type: {entity_type})")

                    # Avoid duplicate nodes, but track all sources
                    if node_id in all_nodes:
                        if "source_chunks" not in all_nodes[node_id]:
                            all_nodes[node_id]["source_chunks"] = [all_nodes[node_id]["source_chunk"]]
                        all_nodes[node_id]["source_chunks"].append(str(chunk.id))
                    else:
                        all_nodes[node_id] = node_data

                    chunk_info["entities"].append(node_data)

                except Exception as e:
                    logger.warning(f"Error processing entity {entity}: {str(e)}")
                    continue

        chunk_metadata.append(chunk_info)

    # Extract relationships/edges from graph structure if available
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
        graph_engine = await get_graph_engine()

        # Get all edges in the graph
        graph_nodes, graph_edges = await graph_engine.get_graph_data()

        for edge in graph_edges:
            try:
                if isinstance(edge, tuple):
                    edge_data = {
                        "source_node_id": str(edge[0]) if len(edge) > 0 else '',
                        "target_node_id": str(edge[1]) if len(edge) > 1 else '',
                        "relationship_name": str(edge[2]) if len(edge) > 2 else '',
                        "properties": edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
                    }
                else:
                    edge_data = {
                        "source_node_id": str(getattr(edge, 'source_node_id', '')),
                        "target_node_id": str(getattr(edge, 'target_node_id', '')),
                        "relationship_name": str(getattr(edge, 'relationship_name', '')),
                        "properties": getattr(edge, 'properties', {}) if hasattr(edge, 'properties') else {}
                    }

                # Only include edges where both source and target are rule entities
                source_id = edge_data["source_node_id"]
                target_id = edge_data["target_node_id"]

                if source_id and target_id and source_id in all_nodes and target_id in all_nodes:
                    # Both nodes are rule entities (since all_nodes only contains rule entities now)
                    all_edges.append(edge_data)
                    logger.debug(f"Including edge between rule entities: {source_id} -> {target_id}")
                else:
                    logger.debug(f"Skipping edge - not between rule entities: {source_id} -> {target_id}")

            except Exception as edge_error:
                logger.warning(f"Error processing individual edge: {str(edge_error)}")
                continue

    except Exception as e:
        logger.warning(f"Could not extract edges from graph engine: {str(e)}")

    unified_graph = {
        "nodes": list(all_nodes.values()),
        "edges": all_edges,
        "chunk_metadata": chunk_metadata,
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "total_chunks": len(data_chunks)
    }

    logger.info(f"Extracted unified graph with rule entities only: {len(all_nodes)} rule nodes, {len(all_edges)} edges from {len(data_chunks)} chunks")
    return unified_graph


async def analyze_conflicts_in_complete_graph(
    unified_graph: Dict[str, Any],
    llm_client,
    confidence_threshold: float = 0.5
) -> ConflictDetectionResult:
    try:
        if not unified_graph["nodes"]:
            logger.debug("No rule entities found in unified graph")
            return ConflictDetectionResult(
                conflicts=[],
                summary="No rule entities available for conflict analysis"
            )

        # Double-check that all nodes in the unified graph are rule entities
        rule_nodes = []
        non_rule_nodes = []

        for node in unified_graph["nodes"]:
            if _is_rule_entity(node):
                rule_nodes.append(node)
            else:
                non_rule_nodes.append(node)
                logger.warning(f"Non-rule entity found in unified graph: {node.get('name', 'Unknown')} (Type: {node.get('type', 'Unknown')})")

        if non_rule_nodes:
            logger.warning(f"Found {len(non_rule_nodes)} non-rule entities in unified graph - filtering them out")
            # Update the unified graph to only contain rule entities
            unified_graph["nodes"] = rule_nodes
            unified_graph["total_nodes"] = len(rule_nodes)

        if not rule_nodes:
            logger.debug("No rule entities found after filtering")
            return ConflictDetectionResult(
                conflicts=[],
                summary="No rule entities found for conflict analysis after filtering"
            )

        # Prepare comprehensive context for LLM analysis with rule entities only
        analysis_context = {
            "total_rule_nodes": unified_graph["total_nodes"],
            "total_edges": unified_graph["total_edges"],
            "total_chunks": unified_graph["total_chunks"],
            "rule_nodes": unified_graph["nodes"],
            "edges": unified_graph["edges"],
            "confidence_threshold": confidence_threshold,
            "analysis_scope": "rule_entities_only"
        }

        # Load conflict detection prompt from external file
        conflict_detection_prompt = read_query_prompt(CONFLICT_DETECTION_PROMPT_PATH)

        # Enhanced prompt for rule entity conflict analysis
        enhanced_prompt = f"""
{conflict_detection_prompt}

## Rule Entity Conflict Analysis Context:
- Total rule entities: {unified_graph['total_nodes']}
- Total edges between rule entities: {unified_graph['total_edges']}
- Source chunks: {unified_graph['total_chunks']}
- Analysis scope: Rule entities only (non-rule entities have been filtered out)

IMPORTANT: Only analyze conflicts between rule entities. All provided nodes are rule entities.

"""

        # Use LLM to detect conflicts across rule entities only
        conflict_result = await llm_client.acreate_structured_output(
            text_input=f"Analyze rule entities for conflicts:\n{analysis_context}",
            system_prompt=enhanced_prompt,
            response_model=ConflictDetectionResult
        )

        # Filter conflicts by confidence threshold
        high_confidence_conflicts = [
            conflict for conflict in conflict_result.conflicts
            if conflict.confidence_score >= confidence_threshold
        ]

        return ConflictDetectionResult(
            conflicts=high_confidence_conflicts,
            summary=f"Rule entity conflict analysis: {conflict_result.summary}"
        )

    except Exception as error:
        logger.error(f"Error analyzing conflicts in rule entities: {str(error)}", exc_info=True)
        return ConflictDetectionResult(
            conflicts=[],
            summary=f"Error during rule entity conflict analysis: {str(error)}"
        )


async def update_graph_with_conflicts(
    conflicts: List[ConflictEdge],
    graph_engine,
    update_mode: str = "annotate"
) -> Dict[str, Any]:
    update_stats = {
        "conflicts_processed": 0,
        "edges_added": 0,
        "edges_annotated": 0,
        "errors": []
    }
    
    try:
        if update_mode == "add_edges":
            # Add new conflict edges to the graph
            conflict_edges = []
            for conflict in conflicts:
                edge_properties = {
                    "conflict_type": conflict.conflict_type,
                    "confidence_score": conflict.confidence_score,
                    "conflict_description": conflict.conflict_description,
                    "detected_at": asyncio.get_event_loop().time(),
                    "conflict_flag": True
                }
                
                conflict_edges.append((
                    conflict.source_node_id,
                    conflict.target_node_id,
                    conflict.relationship_name,
                    edge_properties
                ))
            
            if conflict_edges:
                await graph_engine.add_edges(conflict_edges)
                update_stats["edges_added"] = len(conflict_edges)
                logger.info(f"Added {len(conflict_edges)} conflict edges to graph")
        
        elif update_mode == "annotate":
            # Annotate existing edges with conflict information
            # This would require additional graph engine methods to update edge properties
            logger.info("Conflict annotation mode - conflicts logged for review")
            update_stats["edges_annotated"] = len(conflicts)
        
        update_stats["conflicts_processed"] = len(conflicts)
        
    except Exception as error:
        error_msg = f"Error updating graph with conflicts: {str(error)}"
        logger.error(error_msg, exc_info=True)
        update_stats["errors"].append(error_msg)
    
    return update_stats


async def detect_and_update_conflicts(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel] = KnowledgeGraph,
    confidence_threshold: float = 0.7,
    update_mode: str = "add_edges"
) -> List[DocumentChunk]:
    try:
        # Initialize LLM client and graph engine
        llm_client = get_llm_client()
        graph_engine = await get_graph_engine()

        # Extract complete unified graph data
        raw_unified_graph = await extract_complete_graph_data(data_chunks)

        # Apply additional filtering to ensure only rule entities are included
        unified_graph = filter_unified_graph_for_rules(raw_unified_graph)

        if unified_graph["total_nodes"] == 0:
            logger.info("No rule entities found in graph - skipping conflict detection")
            return data_chunks

        logger.info(f"Analyzing unified graph with rule entities only: {unified_graph['total_nodes']} rule nodes, "
                   f"{unified_graph['total_edges']} edges between rule entities")

        # perform LLM-based rule entity conflict analysis
        conflict_result = await analyze_conflicts_in_complete_graph(
            unified_graph, llm_client, confidence_threshold
        )

        # Remove duplicates based on source/target node pairs
        unique_conflicts = []
        seen_pairs = set()
        for conflict in conflict_result.conflicts:
            pair_key = (conflict.source_node_id, conflict.target_node_id, conflict.conflict_type)
            if pair_key not in seen_pairs:
                unique_conflicts.append(conflict)
                seen_pairs.add(pair_key)

        all_conflicts = unique_conflicts
        logger.info(f"Rule entity conflict analysis detected {len(all_conflicts)} conflicts")

        # Update graph with detected conflicts
        if all_conflicts:
            update_stats = await update_graph_with_conflicts(
                all_conflicts, graph_engine, update_mode
            )

            logger.info(f"Graph update completed: {update_stats}")

        else:
            logger.info("No conflicts detected between rule entities")

        return data_chunks

    except Exception as error:
        logger.error(f"Error in rule entity conflict detection: {str(error)}", exc_info=True)
        raise error