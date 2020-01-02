# coding: utf8

import networkx as nx
import random
import types

from typing import List

from ._base import Grammar, Sampler


class Graph(nx.DiGraph):
    def __init__(self, **attrs):
        super(Graph, self).__init__(**attrs)

    def build_order(self):
        """
        Returns a iterable of `(node, in_nodes)` in topological order.

        In detail, returns every `node` and the list of nodes that are incoming edges to `node`.
        This way, you can call a method on each node whose args are the
        previous nodes.
        """
        for node in nx.topological_sort(self):
            in_nodes = [u for u, v in self.in_edges(node)]
            yield (node, in_nodes)

    def apply(self, function):
        """
        Applies a function to all nodes in `build_order`'s order.

        The function receives the current three arguments:
            - The current `node` instance
            - A list of incoming node instances
            - A list of the result of the application of the
              function in the previous node instances.

        Returns the last of the values computed.
        """
        previous_values = {}

        for node, in_nodes in self.build_order():
            in_values = [previous_values[n] for n in in_nodes]
            value = function(node, in_nodes, in_values)
            previous_values[node] = value

        return value

    def contains_any(self, *items):
        return any((node in items) for node in self)


def uniform_selection(items):
    return random.choice(items)


def first_selection(items):
    return items[0]


def default_initializer(cls):
    return cls()


# Holds a string to class map for automatically generated classes
_GENERATED_CLASSES = {}


def _get_generated_class(name):
    if name in _GENERATED_CLASSES:
        return _GENERATED_CLASSES[name]

    clss = types.new_class(name)
    _GENERATED_CLASSES[name] = clss
    return clss


class Production:
    def __init__(self, pattern, replacement, *, initializer=default_initializer):
        if not isinstance(pattern, Graph):
            obj = pattern
            pattern = Graph()
            pattern.add_node(obj)

        if not isinstance(replacement, GraphPattern):
            replacement = Node(replacement)

        self.pattern = pattern
        self.replacement = replacement
        self.initializer = initializer

    def _matches(self, graph: Graph):
        # TODO: Generalizar a permitir cualquier tipo de grafo como patrón, no solo un nodo
        pattern_node = list(self.pattern.nodes)[0]

        for node in graph.nodes:
            if node.__class__ == pattern_node:
                yield node

    def match(self, graph: Graph):
        """
        Returns True if it finds a subgraph in `graph` that matches the pattern.
        NOTE: Right now only works if pattern is a single node.
        """
        for _ in self._matches(graph):
            return True

        return False

    def apply(self, graph: Graph, pattern_selection=uniform_selection) -> Graph:
        """
        Applies a production in a graph and returns the modified graph.
        """
        matches = list(self._matches(graph))
        node = pattern_selection(matches)

        in_edges = graph.in_edges(node)
        out_edges = graph.out_edges(node)

        in_nodes = [u for u, v in in_edges]
        out_nodes = [v for u, v in out_edges]

        graph.remove_node(node)
        self.replacement.build(
            graph, in_nodes=in_nodes, out_nodes=out_nodes, initializer=self.initializer
        )

        return graph


class GraphPattern:
    def build(
        self,
        graph: Graph,
        *,
        in_nodes=[],
        out_nodes=[],
        initializer=default_initializer,
    ):
        raise NotImplementedError()

    def _add_in_nodes(self, graph, in_nodes, node):
        for in_node in in_nodes:
            graph.add_edge(in_node, node)

    def _add_out_nodes(self, graph, out_nodes, node):
        for out_node in out_nodes:
            graph.add_edge(node, out_node)

    def make(self, *, initializer=default_initializer) -> Graph:
        graph = Graph()
        self.build(graph, in_nodes=[], out_nodes=[], initializer=initializer)
        return graph


class Node(GraphPattern):
    def __init__(self, cls):
        self.cls = _get_generated_class(cls) if isinstance(cls, str) else cls

    def build(
        self,
        graph: Graph,
        *,
        in_nodes=[],
        out_nodes=[],
        initializer=default_initializer,
    ):
        obj = initializer(self.cls)
        graph.add_node(obj)
        self._add_in_nodes(graph, in_nodes, obj)
        self._add_out_nodes(graph, out_nodes, obj)


class Path(GraphPattern):
    def __init__(self, *items):
        self.items = [
            (_get_generated_class(i) if isinstance(i, str) else i) for i in items
        ]

    def build(
        self,
        graph: Graph,
        *,
        in_nodes=[],
        out_nodes=[],
        initializer=default_initializer,
    ):
        items = [initializer(cls) for cls in self.items]
        graph.add_nodes_from(items)

        for i, j in zip(items, items[1:]):
            graph.add_edge(i, j)

        self._add_in_nodes(graph, in_nodes, items[0])
        self._add_out_nodes(graph, out_nodes, items[-1])


class Block(GraphPattern):
    def __init__(self, *items):
        self.items = [
            (_get_generated_class(i) if isinstance(i, str) else i) for i in items
        ]

    def build(
        self,
        graph: Graph,
        *,
        in_nodes=[],
        out_nodes=[],
        initializer=default_initializer,
    ):
        items = [initializer(cls) for cls in self.items]
        graph.add_nodes_from(items)

        for item in items:
            self._add_in_nodes(graph, in_nodes, item)
            self._add_out_nodes(graph, out_nodes, item)


class GraphGrammar(Grammar):
    def __init__(self, start, *, non_terminals=None):
        if isinstance(start, str):
            start = Node(start)

        if isinstance(start, GraphPattern):
            start = start.make()

        super(GraphGrammar, self).__init__(start)
        self._productions: List[Production] = []
        self._non_terminals = set(non_terminals or [])

    def add(
        self, pattern, replacement: GraphPattern, *, initializer=default_initializer
    ):
        if isinstance(pattern, str):
            pattern = _get_generated_class(pattern)
            self._non_terminals.add(pattern)

        self._productions.append(
            Production(pattern, replacement, initializer=initializer)
        )

    # def expand(
    #     self, graph: Graph, *, max_iters=100, production_selector=uniform_selection,
    # ) -> Graph:
    #     if graph is None:
    #         raise ValueError("`graph` cannot be `None`")

    #     if not isinstance(graph, Graph):
    #         obj = graph
    #         graph = Graph()
    #         graph.add_node(obj)
    #     else:
    #         graph = graph.copy()

    #     return self._expand(graph, max_iters, production_selector)

    def _sample(self, symbol, max_iterations, sampler):
        if symbol is None:
            raise ValueError("`symbol` cannot be `None`")

        if max_iterations == 0:
            return symbol

        valid_productions = [p for p in self._productions if p.match(symbol)]

        if self._non_terminals:
            non_terminal_productions = [
                p
                for p in valid_productions
                if p.pattern.contains_any(*self._non_terminals)
            ]

            if non_terminal_productions:
                valid_productions = non_terminal_productions

        if not valid_productions:
            return symbol

        production = sampler.choice(valid_productions)
        symbol = production.apply(symbol)

        return self._sample(symbol, max_iterations - 1, sampler)