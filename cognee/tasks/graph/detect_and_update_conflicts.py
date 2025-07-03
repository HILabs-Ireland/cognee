import asyncio
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.data_models import KnowledgeGraph, Edge as KGEdge
from cognee.shared.logging_utils import get_logger

logger = get_logger("detect_and_update_conflicts")


class ConflictEdge(BaseModel):
    """Represents a detected conflict between two nodes."""
    source_node_id: str
    target_node_id: str
    relationship_name: str = "conflicted_by"
    conflict_type: str = Field(..., description="Type of conflict: direct_contradiction, temporal_conflict, quantitative_conflict, semantic_conflict")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the conflict detection")
    conflict_description: str = Field(..., description="Human-readable description of the conflict")

class ConflictDetectionResult(BaseModel):
    """Result of conflict detection analysis."""
    conflicts: List[ConflictEdge] = Field(default_factory=list)
    summary: str = Field(..., description="Summary of conflicts detected")

# LLM prompt for conflict detection
CONFLICT_DETECTION_PROMPT = """
You are an advanced conflict detection engine for knowledge graphs. Your task is to identify all pairs of rule nodes that have **explicit, direct conflicts** within the complete graph structure as a unified system and create a relationship labeled **_only_ as**:  `conflicted_by`.

## **Conflict Detection Criteria:**
    - A conflict exists **only if all** the following conditions are satisfied: 
        - **identical subjects**
             - The rules must apply to the **same canonical entity** (exact ID match).
        - **identical contexts**
            - The rules must operate in the **same activity, domain, and timeframe**.
        - **mutually exclusive actions**
            - The rules mandate **incompatible requirements** (e.g., `must do X` vs `must not do X`, or `do A in 4 hours` vs `do A in 8 hours`).

    - **Implementation Rules**
        - When all three criteria are met:
            - Add a `conflicted_by` edge:
                - **Direction**: from the **rule being contradicted** to the **rule causing the conflict**
            - Add a `conflict_type` property:
                - `direct_contradiction`: Opposing prescriptions (e.g., must vs must_not)
                - `temporal_conflict`: Same rule, different times (e.g., 4 hrs vs 8 hrs)
                - `quantitative_conflict`: Same rule, different values (e.g., 50kg vs 60kg)
            - Optionally include a `conflict_note`:
                - A concise human-readable description of the contradiction

    - **Exclusion Criteria:**
       - Do **not** create a conflict edge if **any** of the following apply:
            - **Contextual Override**  
                - E.g., emergency protocols overriding standard rules
            - **Partial Scope Overlap**  
                - E.g., one rule applies to "all employees", the other to "employees on night shift"
            - **Temporal Exception**  
                - E.g., conflicting rules apply during **different timeframes**

## Complete Graph Analysis Guidelines:

- Analyze the **ENTIRE graph structure** as a unified system
- Consider **cross-chunk relationships** and global patterns
- Look for **transitive conflicts** (A conflicts with B, B conflicts with C)
- Identify **systemic inconsistencies** that span multiple nodes/edges
- Consider **domain-specific rules** and constraints
- Examine **node clustering** for potential duplicates or conflicts

## Confidence Scoring:

- **0.9-1.0**: Very high confidence (explicit, undeniable contradiction)
- **0.8-0.89**: High confidence (strong semantic conflict with clear evidence)
- **0.7-0.79**: Good confidence (likely conflict with supporting evidence)
- **0.6-0.69**: Medium confidence (potential conflict requiring review)
- **0.5-0.59**: Low confidence (flag for human review)
- **Below 0.5**: Very low confidence (exclude from results)

## Analysis Requirements:

- **Global Perspective**: Consider the entire graph, not just local relationships
- **Evidence-Based**: Provide specific evidence for each detected conflict
- **Context-Aware**: Consider domain context and semantic meaning
- **Systematic**: Look for patterns and systemic issues
- **Precise**: Avoid false positives by requiring substantial evidence

## Output Requirements:

Return a comprehensive analysis with:
- List of detected conflicts with detailed evidence
- Conflict type classification and confidence scores
- Clear descriptions explaining the conflict reasoning
- Summary of overall graph health and conflict patterns

Focus on **quality over quantity** - it's better to miss some conflicts than to create false positives.
"""


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

                    # Handle entity type - could be an object with 'name' attribute or a string
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
                        "description": entity_description,
                        "source_chunk": str(chunk.id)
                    }

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
                    # Handle edge objects
                    edge_data = {
                        "source_node_id": str(getattr(edge, 'source_node_id', '')),
                        "target_node_id": str(getattr(edge, 'target_node_id', '')),
                        "relationship_name": str(getattr(edge, 'relationship_name', '')),
                        "properties": getattr(edge, 'properties', {}) if hasattr(edge, 'properties') else {}
                    }

                if edge_data["source_node_id"] and edge_data["target_node_id"]:
                    all_edges.append(edge_data)

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

    logger.info(f"Extracted unified graph: {len(all_nodes)} nodes, {len(all_edges)} edges from {len(data_chunks)} chunks")
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

        # Prepare comprehensive context for LLM analysis
        analysis_context = {
            "total_nodes": unified_graph["total_nodes"],
            "total_edges": unified_graph["total_edges"],
            "total_chunks": unified_graph["total_chunks"],
            "nodes": unified_graph["nodes"],
            "edges": unified_graph["edges"],
            "confidence_threshold": confidence_threshold,
            "analysis_scope": "complete_graph"
        }

        # Enhanced prompt for complete graph analysis
        enhanced_prompt = f"""
{CONFLICT_DETECTION_PROMPT}

## Complete Graph Analysis Context:
- Total nodes: {unified_graph['total_nodes']}
- Total edges: {unified_graph['total_edges']}
- Source chunks: {unified_graph['total_chunks']}

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

        logger.info(f"Complete graph analysis: {len(conflict_result.conflicts)} total conflicts detected, "
                   f"{len(high_confidence_conflicts)} above confidence threshold {confidence_threshold}")

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
    """
    Update the graph with detected conflicts.
    
    Args:
        conflicts: List of detected conflicts
        graph_engine: Graph database engine
        update_mode: How to handle conflicts ("annotate" or "add_edges")
        
    Returns:
        Dictionary with update statistics
    """
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
    """
    LLM-based conflict detection and graph update task for complete graph analysis.

    This task analyzes the entire graph structure as a unified unit for conflicts,
    such as contradictory relationships, duplicate nodes, or inconsistent edge types.
    It then updates the graph by marking or annotating conflicted edges.

    Args:
        data_chunks: List of document chunks containing graph data
        graph_model: Graph model class (default: KnowledgeGraph)
        confidence_threshold: Minimum confidence score for conflict detection (default: 0.7)
        update_mode: How to handle conflicts - "add_edges" or "annotate" (default: "add_edges")

    Returns:
        List of document chunks (unchanged, as this is a graph analysis task)
    """
    logger.info(f"Starting complete graph conflict detection on {len(data_chunks)} document chunks")

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