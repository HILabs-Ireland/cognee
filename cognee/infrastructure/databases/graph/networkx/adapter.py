"""Adapter for NetworkX graph database."""

from datetime import datetime, timezone
import os
import json
import asyncio
from cognee.shared.logging_utils import get_logger
from typing import Dict, Any, List, Union
from uuid import UUID
import aiofiles
import aiofiles.os as aiofiles_os
import networkx as nx
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.modules.storage.utils import JSONEncoder
import numpy as np

logger = get_logger()


class NetworkXAdapter(GraphDBInterface):
    _instance = None
    graph = None  # Class variable to store the singleton instance

    def __new__(cls, filename):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.filename = filename
        return cls._instance

    def __init__(self, filename="cognee_graph.pkl"):
        self.filename = filename

    async def get_graph_data(self):
        await self.load_graph_from_file()
        return (list(self.graph.nodes(data=True)), list(self.graph.edges(data=True, keys=True)))

    async def query(self, query: str, params: dict):
        pass

    async def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    async def add_node(
        self,
        node: DataPoint,
    ) -> None:
        self.graph.add_node(node.id, **node.model_dump())

        await self.save_graph_to_file(self.filename)

    async def add_nodes(
        self,
        nodes: list[DataPoint],
    ) -> None:
        nodes = [(node.id, node.model_dump()) for node in nodes]

        self.graph.add_nodes_from(nodes)
        await self.save_graph_to_file(self.filename)

    async def get_graph(self):
        return self.graph

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        return self.graph.has_edge(from_node, to_node, key=edge_label)

    async def has_edges(self, edges):
        result = []

        for from_node, to_node, edge_label in edges:
            if self.graph.has_edge(from_node, to_node, edge_label):
                result.append((from_node, to_node, edge_label))

        return result

    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = {},
    ) -> None:
        edge_properties["updated_at"] = datetime.now(timezone.utc)
        self.graph.add_edge(
            from_node,
            to_node,
            key=relationship_name,
            **(edge_properties if edge_properties else {}),
        )

        await self.save_graph_to_file(self.filename)

    async def add_edges(
        self,
        edges: tuple[str, str, str, dict],
    ) -> None:
        edges = [
            (
                edge[0],
                edge[1],
                edge[2],
                {
                    **(edge[3] if len(edge) == 4 else {}),
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            for edge in edges
        ]

        self.graph.add_edges_from(edges)
        await self.save_graph_to_file(self.filename)

    async def get_edges(self, node_id: str):
        return list(self.graph.in_edges(node_id, data=True)) + list(
            self.graph.out_edges(node_id, data=True)
        )

    async def delete_node(self, node_id: str) -> None:
        """Asynchronously delete a node from the graph if it exists."""
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            await self.save_graph_to_file(self.filename)

    async def delete_nodes(self, node_ids: List[str]) -> None:
        self.graph.remove_nodes_from(node_ids)
        await self.save_graph_to_file(self.filename)

    async def get_disconnected_nodes(self) -> List[str]:
        connected_components = list(nx.weakly_connected_components(self.graph))

        disconnected_nodes = []
        biggest_subgraph = max(connected_components, key=len)

        for component in connected_components:
            if component != biggest_subgraph:
                disconnected_nodes.extend(list(component))

        return disconnected_nodes

    async def extract_node(self, node_id: str) -> dict:
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]

        return None

    async def extract_nodes(self, node_ids: List[str]) -> List[dict]:
        return [self.graph.nodes[node_id] for node_id in node_ids if self.graph.has_node(node_id)]

    async def get_predecessors(self, node_id: UUID, edge_label: str = None) -> list:
        if self.graph.has_node(node_id):
            if edge_label is None:
                return [
                    self.graph.nodes[predecessor]
                    for predecessor in list(self.graph.predecessors(node_id))
                ]

            nodes = []

            for predecessor_id in list(self.graph.predecessors(node_id)):
                if self.graph.has_edge(predecessor_id, node_id, edge_label):
                    nodes.append(self.graph.nodes[predecessor_id])

            return nodes

    async def get_successors(self, node_id: UUID, edge_label: str = None) -> list:
        if self.graph.has_node(node_id):
            if edge_label is None:
                return [
                    self.graph.nodes[successor]
                    for successor in list(self.graph.successors(node_id))
                ]

            nodes = []

            for successor_id in list(self.graph.successors(node_id)):
                if self.graph.has_edge(node_id, successor_id, edge_label):
                    nodes.append(self.graph.nodes[successor_id])

            return nodes

    async def get_neighbours(self, node_id: str) -> list:
        if not self.graph.has_node(node_id):
            return []

        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id),
            self.get_successors(node_id),
        )

        neighbours = predecessors + successors

        return neighbours

    async def get_connections(self, node_id: UUID) -> list:
        if not self.graph.has_node(node_id):
            return []

        node = self.graph.nodes[node_id]

        if "id" not in node:
            return []

        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id),
            self.get_successors(node_id),
        )

        connections = []

        for neighbor in predecessors:
            if "id" in neighbor:
                edge_data = self.graph.get_edge_data(neighbor["id"], node["id"])
                for edge_properties in edge_data.values():
                    connections.append((neighbor, edge_properties, node))

        for neighbor in successors:
            if "id" in neighbor:
                edge_data = self.graph.get_edge_data(node["id"], neighbor["id"])
                for edge_properties in edge_data.values():
                    connections.append((node, edge_properties, neighbor))

        return connections

    async def remove_connection_to_predecessors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                for predecessor_id in list(self.graph.predecessors(node_id)):
                    if self.graph.has_edge(predecessor_id, node_id, edge_label):
                        self.graph.remove_edge(predecessor_id, node_id, edge_label)

        await self.save_graph_to_file(self.filename)

    async def remove_connection_to_successors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                for successor_id in list(self.graph.successors(node_id)):
                    if self.graph.has_edge(node_id, successor_id, edge_label):
                        self.graph.remove_edge(node_id, successor_id, edge_label)

        await self.save_graph_to_file(self.filename)

    async def create_empty_graph(self, file_path: str) -> None:
        self.graph = nx.MultiDiGraph()

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        await self.save_graph_to_file(file_path)

    async def save_graph_to_file(self, file_path: str = None) -> None:
        """Asynchronously save the graph to a file in JSON format."""
        if not file_path:
            file_path = self.filename

        graph_data = nx.readwrite.json_graph.node_link_data(self.graph, edges="links")

        async with aiofiles.open(file_path, "w") as file:
            json_data = json.dumps(graph_data, cls=JSONEncoder)
            await file.write(json_data)

    async def load_graph_from_file(self, file_path: str = None):
        """Asynchronously load the graph from a file in JSON format."""
        if file_path == self.filename:
            return

        if not file_path:
            file_path = self.filename
        try:
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, "r") as file:
                    graph_data = json.loads(await file.read())
                    for node in graph_data["nodes"]:
                        try:
                            if not isinstance(node["id"], UUID):
                                try:
                                    node["id"] = UUID(node["id"])
                                except Exception:
                                    # If conversion fails, keep the original id
                                    pass
                        except Exception as e:
                            logger.error(e)
                            raise e

                        if isinstance(node.get("updated_at"), int):
                            node["updated_at"] = datetime.fromtimestamp(
                                node["updated_at"] / 1000, tz=timezone.utc
                            )
                        elif isinstance(node.get("updated_at"), str):
                            node["updated_at"] = datetime.strptime(
                                node["updated_at"], "%Y-%m-%dT%H:%M:%S.%f%z"
                            )

                    for edge in graph_data["links"]:
                        try:
                            if not isinstance(edge["source"], UUID):
                                source_id = parse_id(edge["source"])
                            else:
                                source_id = edge["source"]

                            if not isinstance(edge["target"], UUID):
                                target_id = parse_id(edge["target"])
                            else:
                                target_id = edge["target"]

                            edge["source"] = source_id
                            edge["target"] = target_id
                            edge["source_node_id"] = source_id
                            edge["target_node_id"] = target_id
                        except Exception as e:
                            logger.error(e)
                            raise e

                        if isinstance(
                            edge.get("updated_at"), int
                        ):  # Handle timestamp in milliseconds
                            edge["updated_at"] = datetime.fromtimestamp(
                                edge["updated_at"] / 1000, tz=timezone.utc
                            )
                        elif isinstance(edge.get("updated_at"), str):
                            edge["updated_at"] = datetime.strptime(
                                edge["updated_at"], "%Y-%m-%dT%H:%M:%S.%f%z"
                            )

                    self.graph = nx.readwrite.json_graph.node_link_graph(graph_data, edges="links")

                    for node_id, node_data in self.graph.nodes(data=True):
                        node_data["id"] = node_id
            else:
                # Log that the file does not exist and an empty graph is initialized
                logger.warning("File %s not found. Initializing an empty graph.", file_path)
                await self.create_empty_graph(file_path)

        except Exception:
            logger.error("Failed to load graph from file: %s", file_path)

            await self.create_empty_graph(file_path)

    async def delete_graph(self, file_path: str = None):
        """Asynchronously delete the graph file from the filesystem."""
        if file_path is None:
            file_path = (
                self.filename
            )  # Assuming self.filename is defined elsewhere and holds the default graph file path
        try:
            if os.path.exists(file_path):
                await aiofiles_os.remove(file_path)

            self.graph = None
            logger.info("Graph deleted successfully.")
        except Exception as error:
            logger.error("Failed to delete graph: %s", error)
            raise error

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ):
        """
        Fetches nodes and relationships filtered by specified attribute values.

        Args:
            attribute_filters (list of dict): A list of dictionaries where keys are attributes and values are lists of values to filter on.
                                              Example: [{"community": ["1", "2"]}]

        Returns:
            tuple: A tuple containing two lists:
                - Nodes: List of tuples (node_id, node_properties).
                - Edges: List of tuples (source_id, target_id, relationship_type, edge_properties).
        """
        # Create filters for nodes based on the attribute filters
        where_clauses = []
        for attribute, values in attribute_filters[0].items():
            where_clauses.append((attribute, values))

        # Filter nodes
        filtered_nodes = [
            (node, data)
            for node, data in self.graph.nodes(data=True)
            if all(data.get(attr) in values for attr, values in where_clauses)
        ]

        # Filter edges where both source and target nodes satisfy the filters
        filtered_edges = [
            (source, target, data.get("relationship_type", "UNKNOWN"), data)
            for source, target, data in self.graph.edges(data=True)
            if (
                all(self.graph.nodes[source].get(attr) in values for attr, values in where_clauses)
                and all(
                    self.graph.nodes[target].get(attr) in values for attr, values in where_clauses
                )
            )
        ]

        return filtered_nodes, filtered_edges

    async def get_graph_metrics(self, include_optional=False):
        graph = self.graph

        def _get_mean_degree(graph):
            degrees = [d for _, d in graph.degree()]
            return np.mean(degrees) if degrees else 0

        def _get_edge_density(graph):
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            num_possible_edges = num_nodes * (num_nodes - 1)
            edge_density = num_edges / num_possible_edges if num_possible_edges > 0 else 0
            return edge_density

        def _get_diameter(graph):
            try:
                return nx.diameter(nx.DiGraph(graph.to_undirected()))
            except Exception as e:
                logger.warning("Failed to calculate diameter: %s", e)
                return None

        def _get_avg_shortest_path_length(graph):
            try:
                return nx.average_shortest_path_length(nx.DiGraph(graph.to_undirected()))
            except Exception as e:
                logger.warning("Failed to calculate average shortest path length: %s", e)
                return None

        def _get_avg_clustering(graph):
            try:
                return nx.average_clustering(nx.DiGraph(graph.to_undirected()))
            except Exception as e:
                logger.warning("Failed to calculate clustering coefficient: %s", e)
                return None

        mandatory_metrics = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "mean_degree": _get_mean_degree(graph),
            "edge_density": _get_edge_density(graph),
            "num_connected_components": nx.number_weakly_connected_components(graph),
            "sizes_of_connected_components": [
                len(c) for c in nx.weakly_connected_components(graph)
            ],
        }

        if include_optional:
            optional_metrics = {
                "num_selfloops": sum(1 for u, v in graph.edges() if u == v),
                "diameter": _get_diameter(graph),
                "avg_shortest_path_length": _get_avg_shortest_path_length(graph),
                "avg_clustering": _get_avg_clustering(graph),
            }
        else:
            optional_metrics = {
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }

        return mandatory_metrics | optional_metrics
