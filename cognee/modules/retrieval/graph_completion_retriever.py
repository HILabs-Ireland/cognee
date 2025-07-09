from typing import Any, Optional, Type, List, Dict
from collections import Counter
import string
import logging

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS
from cognee.infrastructure.databases.graph import get_graph_engine



class GraphCompletionRetriever(BaseRetriever):
    """
    Retriever for handling graph-based completion searches.

    This class provides methods to retrieve graph nodes and edges, resolve them into a
    human-readable format, and generate completions based on graph context. Public methods
    include:
    - resolve_edges_to_text
    - get_triplets
    - get_context
    - get_completion
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 5
        self.node_type = node_type
        self.node_name = node_name

    def _get_nodes(self, retrieved_edges: list) -> dict:
        """Creates a dictionary of nodes with their names and content."""
        nodes = {}
        for edge in retrieved_edges:
            for node in (edge.node1, edge.node2):
                if node.id not in nodes:
                    text = node.attributes.get("text")
                    if text:
                        name = self._get_title(text)
                        content = text
                    else:
                        name = node.attributes.get("name", "Unnamed Node")
                        content = name
                    nodes[node.id] = {"node": node, "name": name, "content": content}
        return nodes

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """
        Converts retrieved graph edges into a human-readable string format.

        Parameters:
        -----------

            - retrieved_edges (list): A list of edges retrieved from the graph.

        Returns:
        --------

            - str: A formatted string representation of the nodes and their connections.
        """
        nodes = self._get_nodes(retrieved_edges)
        node_section = "\n".join(
            f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
            for info in nodes.values()
        )
        connection_section = "\n".join(
            f"{nodes[edge.node1.id]['name']} --[{edge.attributes['relationship_type']}]--> {nodes[edge.node2.id]['name']}"
            for edge in retrieved_edges
        )
        return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"

    async def get_triplets(self, query: str) -> list:
        """
        Retrieves relevant graph triplets based on a query string.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant triplets in the graph.

        Returns:
        --------

            - list: A list of found triplets that match the query.
        """
        subclasses = get_all_subclasses(DataPoint)
        vector_index_collections = []

        for subclass in subclasses:
            if "metadata" in subclass.model_fields:
                metadata_field = subclass.model_fields["metadata"]
                if hasattr(metadata_field, "default") and metadata_field.default is not None:
                    if isinstance(metadata_field.default, dict):
                        index_fields = metadata_field.default.get("index_fields", [])
                        for field_name in index_fields:
                            vector_index_collections.append(f"{subclass.__name__}_{field_name}")

        found_triplets = await brute_force_triplet_search(
            query,
            top_k=self.top_k,
            collections=vector_index_collections or None,
            node_type=self.node_type,
            node_name=self.node_name,
        )

        return found_triplets

    async def get_context(self, query: str) -> str:
        """
        Retrieves and resolves graph triplets into context based on a query.

        Parameters:
        -----------

            - query (str): The query string used to retrieve context from the graph triplets.

        Returns:
        --------

            - str: A string representing the resolved context from the retrieved triplets, or an
              empty string if no triplets are found.
        """
        triplets = await self.get_triplets(query)

        if len(triplets) == 0:
            return ""

        # Check if this is a conflict-related query and we have specific node_name
        if self._is_conflict_query(query) and self.node_name:
            return await self._get_cross_nodeset_conflict_context(triplets, query)

        return await self.resolve_edges_to_text(triplets)

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Generates a completion using graph connections context based on a query.

        Parameters:
        -----------

            - query (str): The query string for which a completion is generated.
            - context (Optional[Any]): Optional context to use for generating the completion; if
              not provided, context is retrieved based on the query. (default None)

        Returns:
        --------

            - Any: A generated completion based on the query and context provided.
        """
        if context is None:
            context = await self.get_context(query)

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]

    def _top_n_words(self, text, stop_words=None, top_n=3, separator=", "):
        """Concatenates the top N frequent words in text."""
        if stop_words is None:
            stop_words = DEFAULT_STOP_WORDS

        words = [word.lower().strip(string.punctuation) for word in text.split()]

        if stop_words:
            words = [word for word in words if word and word not in stop_words]

        top_words = [word for word, freq in Counter(words).most_common(top_n)]

        return separator.join(top_words)

    def _get_title(self, text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
        """Creates a title, by combining first words with most frequent words from the text."""
        first_n_words = text.split()[:first_n_words]
        top_n_words = self._top_n_words(text, top_n=top_n_words)
        return f"{' '.join(first_n_words)}... [{top_n_words}]"

    def _is_conflict_query(self, query: str) -> bool:
        """Check if the query is asking for conflict information."""
        conflict_keywords = ["conflict", "conflicted_by", "rule", "contradiction", "inconsistent"]
        return any(keyword in query.lower() for keyword in conflict_keywords)

    async def _get_cross_nodeset_conflict_context(self, triplets: list, query: str) -> str:
        """
        Get enhanced context with cross-nodeset conflict detection.

        Steps:
        1. Fetch all rules for the given nodeset - use get_nodeset_subgraph to fetch all entities
        2. Fetch any conflicting edges to or from fetched rules
        3. Extract those conflicting rules from other nodeset and mark the direction
        4. Summarize all findings into JSON of given format
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting cross-nodeset conflict detection for nodeset: {self.node_name}")

        try:
            graph_engine = await get_graph_engine()
            logger.debug("Graph engine obtained successfully")

            # Step 1: Fetch all rule entities for the given nodeset using get_nodeset_subgraph
            logger.info(f"Step 1: Fetching rule entities for nodeset: {self.node_name}")
            target_rule_entities = await self._fetch_rules_for_nodeset_using_subgraph(graph_engine, self.node_name, self.node_type)
            logger.info(f"Step 1 completed: Found {len(target_rule_entities)} target rule entities")

            if not target_rule_entities:
                logger.warning("No target rule entities found, falling back to basic context")
                return await self.resolve_edges_to_text(triplets)

            # Step 2: Fetch any conflicting edges to or from fetched rule entities
            logger.info("Step 2: Fetching conflict edges")
            conflict_edges = await self._fetch_conflict_edges(graph_engine, target_rule_entities)
            logger.info(f"Step 2 completed: Found {len(conflict_edges)} conflict edges")

            # Step 3: Extract conflicting rule entities from other nodesets and mark direction
            logger.info("Step 3: Extracting cross-nodeset conflicts")
            cross_nodeset_conflicts = await self._extract_cross_nodeset_conflicts(
                graph_engine, target_rule_entities, conflict_edges
            )

            outgoing_count = len(cross_nodeset_conflicts['outgoing'])
            incoming_count = len(cross_nodeset_conflicts['incoming'])
            other_rule_entities_count = len(cross_nodeset_conflicts['other_nodeset_rules'])

            logger.info(f"Step 3 completed: {outgoing_count} outgoing, {incoming_count} incoming conflicts")
            logger.info(f"Found {other_rule_entities_count} rule entities from other nodesets")

            # Step 4: Summarize all findings into enhanced context
            logger.info("Step 4: Formatting enhanced context")
            enhanced_context = await self._format_cross_nodeset_context(
                target_rule_entities, cross_nodeset_conflicts, triplets, query
            )

            context_length = len(enhanced_context)
            logger.info(f"Step 4 completed: Generated context with {context_length} characters")
            logger.info("Cross-nodeset conflict detection completed successfully")

            return enhanced_context

        except Exception as e:
            logger.error(f"Error in cross-nodeset conflict detection: {str(e)}")
            logger.exception("Full exception details:")
            logger.info("Falling back to basic context")
            # Fallback to basic context on any error
            return await self.resolve_edges_to_text(triplets)

    async def _fetch_rules_for_nodeset_using_subgraph(self, graph_engine, node_names: List[str], node_type: Optional[Type] = None) -> Dict[str, Dict]:
        """
        Step 1: Fetch all entities for the given nodeset using get_nodeset_subgraph.
        Returns only entities that have entity type "rule".

        Returns:
            Dict mapping node_id to node_data for all rule entities in the specified nodesets
        """
        logger = logging.getLogger(__name__)
        rule_entities = {}

        try:
            logger.debug(f"Fetching rule entities for nodesets: {node_names}, node_type: {node_type}")

            # Use get_nodeset_subgraph to fetch all entities for the given nodeset
            if hasattr(graph_engine, 'get_nodeset_subgraph') and node_names:
                # Force node_type to Entity to return only entity nodes
                if node_type is None:
                    # Import Entity type for filtering
                    try:
                        from cognee.modules.engine.models import Entity
                        node_type = Entity
                        logger.debug("Using Entity as node_type to fetch only entity nodes")
                    except ImportError:
                        logger.warning("Could not import Entity type, using fallback query method")
                        return await self._fetch_rule_entities_fallback_query(graph_engine, node_names)

                logger.debug(f"Using get_nodeset_subgraph with node_type: {node_type.__name__} (rule entities only)")

                # Get subgraph for the specified nodeset, filtering for entity type
                nodes_data, edges_data = await graph_engine.get_nodeset_subgraph(
                    node_type=node_type,
                    node_name=node_names
                )

                logger.debug(f"get_nodeset_subgraph returned {len(nodes_data)} nodes and {len(edges_data)} edges")

                # Process nodes to extract rule entities only
                for node_id, node_data in nodes_data:
                    if node_data:
                        # Convert node_id to string for consistency
                        str_node_id = str(node_id)

                        # Only include entities with entity type "rule"
                        if self._is_rule_entity(node_data):
                            rule_entities[str_node_id] = node_data
                            node_name = node_data.get('name', str_node_id)
                            entity_type = node_data.get('entity_type', node_data.get('type', 'Unknown'))
                            logger.debug(f"Added rule entity: {node_name} (ID: {str_node_id}, EntityType: {entity_type})")
                        else:
                            node_name = node_data.get('name', str_node_id)
                            entity_type = node_data.get('entity_type', node_data.get('type', 'Unknown'))
                            logger.debug(f"Skipping non-rule entity: {node_name} (EntityType: {entity_type})")

                logger.info(f"Successfully fetched {len(rule_entities)} rule entities using get_nodeset_subgraph for nodesets: {node_names}")

            else:
                logger.warning("get_nodeset_subgraph not available, using fallback query method")
                return await self._fetch_rule_entities_fallback_query(graph_engine, node_names)

        except Exception as e:
            logger.error(f"Error fetching rule entities using get_nodeset_subgraph for nodesets {node_names}: {str(e)}")
            logger.exception("Full exception details:")
            logger.info("Attempting fallback query method")
            return await self._fetch_rule_entities_fallback_query(graph_engine, node_names)

        return rule_entities

    def _is_rule_entity(self, node_data: Dict) -> bool:
        """Check if a node is an entity with entity type 'rule'."""
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

    async def _fetch_rule_entities_fallback_query(self, graph_engine, node_names: List[str]) -> Dict[str, Dict]:
        """Fallback method to fetch rule entities using direct graph queries."""
        logger = logging.getLogger(__name__)
        rule_entities = {}

        try:
            if hasattr(graph_engine, 'query') and node_names:
                # Query for entity type nodes with entity_type = 'rule'
                query = """
                MATCH (n:Entity)
                WHERE n.name IN $node_names
                AND (n.entity_type = 'rule' OR n.type = 'rule' OR EXISTS(n.rule) OR EXISTS(n.policy))
                RETURN n
                """

                logger.debug(f"Executing fallback rule entity query: {query}")
                results = await graph_engine.query(query, {'node_names': node_names})
                logger.debug(f"Fallback rule entity query returned {len(results)} results")

                for result in results:
                    # Handle different result formats
                    if isinstance(result, tuple) and len(result) > 0:
                        node = result[0]
                    elif isinstance(result, dict):
                        node = result.get('n', {})
                    else:
                        continue

                    if node:
                        node_dict = self._normalize_node_data(node)
                        node_id = str(self._get_node_property(node, 'id', ''))
                        node_name = self._get_node_property(node, 'name', 'N/A')

                        if node_id and self._is_rule_entity(node_dict):
                            rule_entities[node_id] = node_dict
                            entity_type = node_dict.get('entity_type', node_dict.get('type', 'Unknown'))
                            logger.debug(f"Added rule entity via fallback: {node_name} (ID: {node_id}, EntityType: {entity_type})")

            logger.info(f"Fallback method fetched {len(rule_entities)} rule entities for nodesets: {node_names}")

        except Exception as e:
            logger.error(f"Error in fallback rule entity fetching for nodesets {node_names}: {str(e)}")
            logger.exception("Full exception details:")

        return rule_entities

    async def _fetch_conflict_edges(self, graph_engine, target_rules: Dict[str, Dict]) -> List[Dict]:
        """
        Step 2: Fetch any conflicting edges to or from fetched rules.

        Returns:
            List of conflict edges involving the target rules
        """
        logger = logging.getLogger(__name__)
        conflict_edges = []
        target_rule_ids = set(target_rules.keys())

        logger.debug(f"Fetching conflict edges for {len(target_rule_ids)} target rules")
        logger.debug(f"Target rule IDs: {list(target_rule_ids)}")

        try:
            if hasattr(graph_engine, 'query'):
                # Get all conflict edges in the graph
                conflict_query = """
                MATCH (source)-[r:conflicted_by]->(target)
                RETURN source, r, target
                """
                logger.debug(f"Executing conflict query: {conflict_query}")
                results = await graph_engine.query(conflict_query, {})
                logger.debug(f"Conflict query returned {len(results)} total conflict edges")

                relevant_edges = 0
                for result in results:
                    # Handle both tuple and dict result formats
                    if isinstance(result, tuple):
                        # Result is a tuple: (source_node, relationship, target_node)
                        if len(result) >= 3:
                            source_node = result[0]
                            relationship = result[1]
                            target_node = result[2]
                        else:
                            logger.warning(f"Unexpected tuple format in result: {result}")
                            continue
                    elif isinstance(result, dict):
                        # Result is a dict: {'source': source_node, 'r': relationship, 'target': target_node}
                        source_node = result.get('source', {})
                        target_node = result.get('target', {})
                        relationship = result.get('r', {})
                    else:
                        logger.warning(f"Unexpected result format: {type(result)} - {result}")
                        continue

                    if source_node and target_node:
                        # Handle different node data formats
                        source_id = str(self._get_node_property(source_node, 'id', ''))
                        target_id = str(self._get_node_property(target_node, 'id', ''))
                        source_name = self._get_node_property(source_node, 'name', source_id)
                        target_name = self._get_node_property(target_node, 'name', target_id)

                        # Include edge if it involves any of our target rules
                        if source_id in target_rule_ids or target_id in target_rule_ids:
                            relevant_edges += 1
                            conflict_type = self._get_node_property(relationship, 'conflict_type', 'N/A') if relationship else 'N/A'

                            conflict_edges.append({
                                'source_id': source_id,
                                'target_id': target_id,
                                'source_node': self._normalize_node_data(source_node),
                                'target_node': self._normalize_node_data(target_node),
                                'relationship_properties': self._normalize_node_data(relationship) if relationship else {}
                            })

                            logger.debug(f"Added relevant conflict edge: {source_name} → {target_name} ({conflict_type})")

                logger.info(f"Found {relevant_edges} relevant conflict edges out of {len(results)} total")

        except Exception as e:
            logger.error(f"Error fetching conflict edges: {str(e)}")
            logger.exception("Full exception details:")

        return conflict_edges

    def _get_node_property(self, node, property_name: str, default_value=""):
        """Safely get a property from a node, handling different data formats."""
        if not node:
            return default_value

        # Handle dict format
        if isinstance(node, dict):
            return node.get(property_name, default_value)

        # Handle object with attributes
        if hasattr(node, property_name):
            return getattr(node, property_name, default_value)

        # Handle object with get method
        if hasattr(node, 'get'):
            return node.get(property_name, default_value)

        return default_value

    def _normalize_node_data(self, node) -> Dict:
        """Convert node data to a consistent dictionary format."""
        if not node:
            return {}

        # Already a dict
        if isinstance(node, dict):
            return dict(node)

        # Convert object to dict
        if hasattr(node, '__dict__'):
            return dict(node.__dict__)

        # Try to extract common properties
        result = {}
        common_properties = ['id', 'name', 'type', 'rule', 'description', 'conflict_type', 'confidence_score', 'conflict_description']

        for prop in common_properties:
            value = self._get_node_property(node, prop)
            if value:
                result[prop] = value

        return result

    async def _extract_cross_nodeset_conflicts(self, graph_engine, target_rules: Dict[str, Dict], conflict_edges: List[Dict]) -> Dict:
        """
        Step 3: Extract conflicting rules from other nodeset and mark the direction.
        Only includes conflicts where both source and target entities are of entity type "rule".

        Returns:
            Dict with organized cross-nodeset conflicts
        """
        logger = logging.getLogger(__name__)

        cross_nodeset_conflicts = {
            'outgoing': {},  # target_rule_id -> [conflicts_with_other_nodesets]
            'incoming': {},  # target_rule_id -> [conflicted_by_other_nodesets]
            'other_nodeset_rules': {}  # rule_id -> rule_data for rules from other nodesets
        }

        target_rule_ids = set(target_rules.keys())
        target_nodesets = set(rule_data.get('name', '') for rule_data in target_rules.values())

        logger.debug(f"Extracting cross-nodeset conflicts from {len(conflict_edges)} edges")
        logger.debug(f"Target nodesets: {list(target_nodesets)}")

        outgoing_count = 0
        incoming_count = 0
        internal_conflicts = 0
        non_rule_conflicts = 0

        for edge in conflict_edges:
            source_id = edge['source_id']
            target_id = edge['target_id']
            source_node = edge['source_node']
            target_node = edge['target_node']

            source_nodeset = source_node.get('name', '')
            target_nodeset = target_node.get('name', '')

            # Check if both source and target are rule entities
            source_is_rule = self._is_rule_entity(source_node)
            target_is_rule = self._is_rule_entity(target_node)

            logger.debug(f"Analyzing edge: {source_nodeset} → {target_nodeset}")
            logger.debug(f"Source is rule entity: {source_is_rule}, Target is rule entity: {target_is_rule}")

            # Only process conflicts where both entities are rule entities
            if not (source_is_rule and target_is_rule):
                non_rule_conflicts += 1
                source_type = source_node.get('entity_type', source_node.get('type', 'Unknown'))
                target_type = target_node.get('entity_type', target_node.get('type', 'Unknown'))
                logger.debug(f"Skipping non-rule conflict: {source_nodeset} ({source_type}) → {target_nodeset} ({target_type})")
                continue

            # Check if this is a cross-nodeset conflict
            source_in_target_nodesets = source_nodeset in target_nodesets
            target_in_target_nodesets = target_nodeset in target_nodesets

            logger.debug(f"Source in target nodesets: {source_in_target_nodesets}, Target in target nodesets: {target_in_target_nodesets}")

            if source_in_target_nodesets and not target_in_target_nodesets:
                # Outgoing conflict: our rule conflicts with rule from other nodeset
                outgoing_count += 1
                logger.debug(f"Found outgoing rule-to-rule conflict: {source_nodeset} → {target_nodeset}")

                if source_id not in cross_nodeset_conflicts['outgoing']:
                    cross_nodeset_conflicts['outgoing'][source_id] = []

                conflict_info = {
                    'conflicted_rule_id': target_id,
                    'conflicted_rule_name': target_nodeset,
                    'conflicted_rule_text': target_node.get('rule', target_node.get('description', target_nodeset)),
                    'conflict_description': edge['relationship_properties'].get('conflict_description', 'Rules are in conflict'),
                    'conflict_type': edge['relationship_properties'].get('conflict_type', 'rule_contradiction'),
                    'confidence_score': edge['relationship_properties'].get('confidence_score', 'N/A'),
                    'conflict_direction': 'outgoing'
                }

                cross_nodeset_conflicts['outgoing'][source_id].append(conflict_info)
                cross_nodeset_conflicts['other_nodeset_rules'][target_id] = target_node
                logger.debug(f"Added outgoing rule conflict for {source_nodeset}: {conflict_info['conflict_type']}")

            elif not source_in_target_nodesets and target_in_target_nodesets:
                # Incoming conflict: rule from other nodeset conflicts with our rule
                incoming_count += 1
                logger.debug(f"Found incoming rule-to-rule conflict: {source_nodeset} → {target_nodeset}")

                if target_id not in cross_nodeset_conflicts['incoming']:
                    cross_nodeset_conflicts['incoming'][target_id] = []

                conflict_info = {
                    'conflicted_rule_id': source_id,
                    'conflicted_rule_name': source_nodeset,
                    'conflicted_rule_text': source_node.get('rule', source_node.get('description', source_nodeset)),
                    'conflict_description': edge['relationship_properties'].get('conflict_description', 'Rules are in conflict'),
                    'conflict_type': edge['relationship_properties'].get('conflict_type', 'rule_contradiction'),
                    'confidence_score': edge['relationship_properties'].get('confidence_score', 'N/A'),
                    'conflict_direction': 'incoming'
                }

                cross_nodeset_conflicts['incoming'][target_id].append(conflict_info)
                cross_nodeset_conflicts['other_nodeset_rules'][source_id] = source_node
                logger.debug(f"Added incoming rule conflict for {target_nodeset}: {conflict_info['conflict_type']}")

            else:
                # Internal conflict within same nodeset - not cross-nodeset
                internal_conflicts += 1
                logger.debug(f"Skipping internal rule conflict: {source_nodeset} → {target_nodeset} (both in target nodesets)")

        logger.info(f"Cross-nodeset rule conflict extraction completed:")
        logger.info(f"  - Outgoing rule conflicts: {outgoing_count}")
        logger.info(f"  - Incoming rule conflicts: {incoming_count}")
        logger.info(f"  - Internal rule conflicts (skipped): {internal_conflicts}")
        logger.info(f"  - Non-rule conflicts (skipped): {non_rule_conflicts}")
        logger.info(f"  - Other nodeset rule entities involved: {len(cross_nodeset_conflicts['other_nodeset_rules'])}")

        return cross_nodeset_conflicts

    async def _format_cross_nodeset_context(self, target_rules: Dict[str, Dict], cross_nodeset_conflicts: Dict, triplets: list, query: str) -> str:
        """
        Step 4: Summarize all findings into enhanced context for JSON generation.

        Returns:
            Formatted context string with cross-nodeset conflict information
        """
        logger = logging.getLogger(__name__)
        logger.debug("Starting context formatting for cross-nodeset conflicts")

        # Extract JSON format from query if present
        json_format_instruction = ""
        if "JSON format" in query:
            start_idx = query.find("{")
            end_idx = query.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                json_format_instruction = query[start_idx:end_idx]
                logger.debug(f"Extracted JSON format instruction: {len(json_format_instruction)} characters")
        else:
            logger.debug("No JSON format instruction found in query")

        context = f"""CROSS-NODESET CONFLICT ANALYSIS CONTEXT
==========================================

INSTRUCTIONS: Generate JSON output for each entity in the target nodeset.
Include conflicts with entities from OTHER nodesets only.

REQUIRED JSON FORMAT:
{json_format_instruction}

ANALYSIS SUMMARY:
- Target Entities Found: {len(target_rules)}
- Cross-Nodeset Outgoing Conflicts: {len(cross_nodeset_conflicts['outgoing'])}
- Cross-Nodeset Incoming Conflicts: {len(cross_nodeset_conflicts['incoming'])}
- Other Nodeset Entities Involved: {len(cross_nodeset_conflicts['other_nodeset_rules'])}

TARGET ENTITIES WITH CROSS-NODESET CONFLICTS:
============================================
"""

        # Format each target entity with its cross-nodeset conflicts
        for entity_id, entity_data in target_rules.items():
            entity_name = entity_data.get('name', entity_id)
            entity_text = entity_data.get('rule', entity_data.get('description', entity_data.get('content', entity_name)))
            entity_description = entity_data.get('rule_description', entity_data.get('description', 'N/A'))
            additional_labels = entity_data.get('additional_labels', entity_data.get('labels', []))
            entity_type = entity_data.get('type', 'Entity')

            # Ensure additional_labels is a list
            if isinstance(additional_labels, str):
                additional_labels = [additional_labels] if additional_labels else []

            context += f"""
ENTITY: {entity_name}
- Entity Type: {entity_type}
- Entity Content: {entity_text}
- Entity Description: {entity_description}
- Additional Labels: {additional_labels}
- Node ID: {entity_id}

CROSS-NODESET CONFLICTS FOR THIS ENTITY:
"""

            # Add outgoing conflicts (this entity conflicts with entities from other nodesets)
            outgoing_conflicts = cross_nodeset_conflicts['outgoing'].get(entity_id, [])
            if outgoing_conflicts:
                context += "OUTGOING CONFLICTS (conflicts with entities from other nodesets):\n"
                for conflict in outgoing_conflicts:
                    context += f"""  - Conflicted Entity: {conflict['conflicted_rule_text']}
    Entity Name: {conflict['conflicted_rule_name']}
    Conflict Description: {conflict['conflict_description']}
    Conflict Type: {conflict['conflict_type']}
    Confidence Score: {conflict['confidence_score']}
    Conflicting Conditions: identical subject, identical context (domain + timeframe), mutually exclusive actions
    Conflict Direction: {conflict['conflict_direction']}
"""

            # Add incoming conflicts (entities from other nodesets conflict with this entity)
            incoming_conflicts = cross_nodeset_conflicts['incoming'].get(entity_id, [])
            if incoming_conflicts:
                context += "INCOMING CONFLICTS (entities from other nodesets that conflict with this entity):\n"
                for conflict in incoming_conflicts:
                    context += f"""  - Conflicted Entity: {conflict['conflicted_rule_text']}
    Entity Name: {conflict['conflicted_rule_name']}
    Conflict Description: {conflict['conflict_description']}
    Conflict Type: {conflict['conflict_type']}
    Confidence Score: {conflict['confidence_score']}
    Conflicting Conditions: identical subject, identical context (domain + timeframe), mutually exclusive actions
    Conflict Direction: {conflict['conflict_direction']}
"""

            if not outgoing_conflicts and not incoming_conflicts:
                context += "No cross-nodeset conflicts detected for this entity.\n"

            context += "=" * 60 + "\n"

        # Add basic triplet context for additional graph connections
        basic_context = await self.resolve_edges_to_text(triplets)
        if basic_context:
            context += f"\nADDITIONAL GRAPH CONNECTIONS:\n{basic_context}\n"
            logger.debug(f"Added basic triplet context: {len(basic_context)} characters")

        context += """
IMPORTANT NOTES:
- Only include conflicts with entities from OTHER nodesets (cross-nodeset conflicts)
- Use "outgoing" for conflicts where the current entity conflicts with others
- Use "incoming" for conflicts where other entities conflict with the current entity
- Generate valid JSON array with each entity as a separate object
- Include ALL cross-nodeset conflicts found in the data above
"""

        total_context_length = len(context)
        logger.info(f"Context formatting completed: {total_context_length} total characters")
        logger.debug(f"Context includes {len(target_rules)} target entities with cross-nodeset conflicts")

        return context

    def _top_n_words(self, text, stop_words=None, top_n=3, separator=", "):
        """Concatenates the top N frequent words in text."""
        if stop_words is None:
            stop_words = DEFAULT_STOP_WORDS

        words = [word.lower().strip(string.punctuation) for word in text.split()]

        if stop_words:
            words = [word for word in words if word and word not in stop_words]

        top_words = [word for word, freq in Counter(words).most_common(top_n)]

        return separator.join(top_words)

    def _get_title(self, text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
        """Creates a title, by combining first words with most frequent words from the text."""
        first_n_words = text.split()[:first_n_words]
        top_n_words = self._top_n_words(text, top_n=top_n_words)
        return f"{' '.join(first_n_words)}... [{top_n_words}]"