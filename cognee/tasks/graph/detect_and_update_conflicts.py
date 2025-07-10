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


def _is_rule_entity(entity_data: Dict[str, Any]) -> bool:
    """Check if an entity is of type 'rule'."""
    # Check for explicit entity_type field
    entity_type = entity_data.get('entity_type', '').lower()
    if entity_type == 'rule':
        return True

    # Exclude non-rule entity types explicitly
    excluded_types = ['person', 'location', 'concept', 'document', 'organization', 'event', 'time']
    if entity_type in excluded_types:
        return False

    # Check type field as fallback
    node_type = entity_data.get('type', '').lower()
    if node_type == 'rule':
        return True
    if node_type in excluded_types:
        return False

    # Check for rule-specific properties
    rule_indicators = [
        'rule',           # Direct rule field
        'rule_text',      # Rule text field
        'policy',         # Policy rules
        'constraint',     # Constraint rules
        'requirement'     # Requirement rules
    ]

    # Must have name and rule indicators
    has_name = 'name' in entity_data
    has_rule_content = any(indicator in entity_data for indicator in rule_indicators)

    # Check if it has rule-like content in description (but be more specific)
    description = entity_data.get('description', '').lower()
    content = entity_data.get('content', '').lower()

    # Look for rule-like language patterns that indicate actual rules/policies
    rule_patterns = [
        'must ', 'shall ', 'required to', 'mandatory', 'prohibited', 'forbidden',
        'policy states', 'regulation requires', 'rule specifies', 'constraint is',
        'requirement is', 'users must', 'employees must', 'all users shall',
        'it is required', 'compliance requires'
    ]

    has_rule_description = any(pattern in description or pattern in content
                             for pattern in rule_patterns)

    # Return True only if it's clearly a rule entity and not excluded
    if has_name and (has_rule_content or has_rule_description):
        return True

    return False


async def extract_complete_graph_data(data_chunks: List[DocumentChunk]) -> Dict[str, Any]:
    all_nodes = {}
    all_edges = []
    chunk_metadata = []
    total_entities_processed = 0
    rule_entities_included = 0

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
                    total_entities_processed += 1
                    node_id = str(getattr(entity, 'id', ''))
                    entity_name = getattr(entity, 'name', '')

                    # Extract entity type information
                    entity_type = ''
                    entity_type_field = ''

                    # Check for entity_type field first (most specific)
                    if hasattr(entity, 'entity_type'):
                        entity_type_field = str(getattr(entity, 'entity_type', ''))
                        entity_type = entity_type_field
                    elif hasattr(entity, 'is_a'):
                        is_a_attr = getattr(entity, 'is_a', None)
                        if is_a_attr:
                            if hasattr(is_a_attr, 'name'):
                                entity_type = getattr(is_a_attr, 'name', '')
                            else:
                                entity_type = str(is_a_attr)
                    elif hasattr(entity, 'type'):
                        entity_type = getattr(entity, 'type', '')

                    entity_description = getattr(entity, 'description', '')

                    # Extract additional rule-related fields
                    rule_content = ''
                    if hasattr(entity, 'rule'):
                        rule_content = str(getattr(entity, 'rule', ''))
                    elif hasattr(entity, 'policy'):
                        rule_content = str(getattr(entity, 'policy', ''))
                    elif hasattr(entity, 'constraint'):
                        rule_content = str(getattr(entity, 'constraint', ''))
                    elif hasattr(entity, 'requirement'):
                        rule_content = str(getattr(entity, 'requirement', ''))

                    node_data = {
                        "id": node_id,
                        "name": entity_name,
                        "type": entity_type,
                        "entity_type": entity_type_field,
                        "description": entity_description,
                        "rule": rule_content,
                        "source_chunk": str(chunk.id)
                    }

                    # Only include rule entities
                    if _is_rule_entity(node_data):
                        rule_entities_included += 1

                        # Avoid duplicate nodes, but track all sources
                        if node_id in all_nodes:
                            if "source_chunks" not in all_nodes[node_id]:
                                all_nodes[node_id]["source_chunks"] = [all_nodes[node_id]["source_chunk"]]
                            all_nodes[node_id]["source_chunks"].append(str(chunk.id))
                        else:
                            all_nodes[node_id] = node_data

                        chunk_info["entities"].append(node_data)
                        logger.debug(f"Included rule entity: {entity_name} (ID: {node_id}, Type: {entity_type})")
                    else:
                        logger.debug(f"Filtered out non-rule entity: {entity_name} (ID: {node_id}, Type: {entity_type})")

                except Exception as e:
                    logger.warning(f"Error processing entity {entity}: {str(e)}")
                    continue

        chunk_metadata.append(chunk_info)

    # Extract relationships/edges from graph structure if available
    # Only include edges between rule entities
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
        graph_engine = await get_graph_engine()

        # Get all edges in the graph
        graph_nodes, graph_edges = await graph_engine.get_graph_data()

        # Create a set of rule entity IDs for fast lookup
        rule_entity_ids = set(all_nodes.keys())
        total_edges_processed = 0
        rule_edges_included = 0

        for edge in graph_edges:
            try:
                total_edges_processed += 1

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

                if (source_id and target_id and
                    source_id in rule_entity_ids and
                    target_id in rule_entity_ids):
                    all_edges.append(edge_data)
                    rule_edges_included += 1
                    logger.debug(f"Included rule-to-rule edge: {source_id} → {target_id} ({edge_data['relationship_name']})")
                else:
                    logger.debug(f"Filtered out non-rule edge: {source_id} → {target_id}")

            except Exception as edge_error:
                logger.warning(f"Error processing individual edge: {str(edge_error)}")
                continue

        logger.info(f"Edge filtering completed:")
        logger.info(f"  - Total edges processed: {total_edges_processed}")
        logger.info(f"  - Rule-to-rule edges included: {rule_edges_included}")
        logger.info(f"  - Non-rule edges filtered: {total_edges_processed - rule_edges_included}")

    except Exception as e:
        logger.warning(f"Could not extract edges from graph engine: {str(e)}")

    unified_graph = {
        "nodes": list(all_nodes.values()),
        "edges": all_edges,
        "chunk_metadata": chunk_metadata,
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "total_chunks": len(data_chunks),
        "filtering_stats": {
            "total_entities_processed": total_entities_processed,
            "rule_entities_included": rule_entities_included,
            "non_rule_entities_filtered": total_entities_processed - rule_entities_included
        }
    }

    logger.info(f"Extracted unified graph with rule entity filtering:")
    logger.info(f"  - Total entities processed: {total_entities_processed}")
    logger.info(f"  - Rule entities included: {rule_entities_included}")
    logger.info(f"  - Non-rule entities filtered: {total_entities_processed - rule_entities_included}")
    logger.info(f"  - Total edges: {len(all_edges)}")
    logger.info(f"  - Source chunks: {len(data_chunks)}")

    return unified_graph


async def analyze_conflicts_in_complete_graph(
    unified_graph: Dict[str, Any],
    llm_client,
    confidence_threshold: float = 0.5
) -> ConflictDetectionResult:
    try:
        if not unified_graph["nodes"]:
            logger.debug("No nodes found in unified graph")
            return ConflictDetectionResult(
                conflicts=[],
                summary="No graph data available for conflict analysis"
            )

        # Get filtering statistics
        filtering_stats = unified_graph.get("filtering_stats", {})

        # Prepare comprehensive context for LLM analysis
        analysis_context = {
            "total_nodes": unified_graph["total_nodes"],
            "total_edges": unified_graph["total_edges"],
            "total_chunks": unified_graph["total_chunks"],
            "nodes": unified_graph["nodes"],
            "edges": unified_graph["edges"],
            "confidence_threshold": confidence_threshold,
            "analysis_scope": "rule_entities_only",
            "filtering_stats": filtering_stats
        }

        # Load conflict detection prompt from external file
        conflict_detection_prompt = read_query_prompt(CONFLICT_DETECTION_PROMPT_PATH)

        # Enhanced prompt for rule entity conflict analysis
        enhanced_prompt = f"""
{conflict_detection_prompt}

## Rule Entity Conflict Analysis Context:
- Rule entities (nodes): {unified_graph['total_nodes']}
- Rule-to-rule edges: {unified_graph['total_edges']}
- Source chunks: {unified_graph['total_chunks']}
- Entity filtering applied: Only entities with entity_type='rule' are included
- Total entities processed: {filtering_stats.get('total_entities_processed', 'N/A')}
- Non-rule entities filtered: {filtering_stats.get('non_rule_entities_filtered', 'N/A')}

IMPORTANT: All nodes in this analysis are rule entities (policies, constraints, requirements, regulations).
Focus on detecting conflicts between rules that contradict each other, have incompatible requirements,
or create logical inconsistencies in policy enforcement.

"""

        # Use LLM to detect conflicts across the complete graph
        conflict_result = await llm_client.acreate_structured_output(
            text_input=f"Analyze the complete graph for conflicts:\n{analysis_context}",
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
            summary=f"Complete graph analysis: {conflict_result.summary}"
        )

    except Exception as error:
        logger.error(f"Error analyzing conflicts in complete graph: {str(error)}", exc_info=True)
        return ConflictDetectionResult(
            conflicts=[],
            summary=f"Error during complete graph conflict analysis: {str(error)}"
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
        unified_graph = await extract_complete_graph_data(data_chunks)

        if unified_graph["total_nodes"] == 0:
            logger.info("No nodes found in graph - skipping conflict detection")
            return data_chunks

        logger.info(f"Analyzing unified graph: {unified_graph['total_nodes']} nodes, "
                   f"{unified_graph['total_edges']} edges")

        # perform LLM-based complete graph conflict analysis
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
        logger.info(f"Complete graph analysis detected {len(all_conflicts)} conflicts")

        # Update graph with detected conflicts
        if all_conflicts:
            update_stats = await update_graph_with_conflicts(
                all_conflicts, graph_engine, update_mode
            )

            logger.info(f"Graph update completed: {update_stats}")

        else:
            logger.info("No conflicts detected in complete graph analysis")

        return data_chunks

    except Exception as error:
        logger.error(f"Error in complete graph conflict detection: {str(error)}", exc_info=True)
        raise error