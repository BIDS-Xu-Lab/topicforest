from collections import defaultdict

class TopicTree:
    '''
    A tree data structure to represent the hierarchical relationships between topics.
    '''
    child_to_parent = {}
    parent_to_children = {}

    def __init__(self):
        self.child_to_parent = {}
        self.parent_to_children = defaultdict(list)

    def _add_edge(self, child: str, parent: str):
        self.child_to_parent[child] = parent
        self.parent_to_children[parent].append(child)

    def add_edges(self, edges: list[tuple[str, str]]):
        for child, parent in edges:
            self._add_edge(child, parent)

    def get_ancestors(self, child: str) -> list[str]:
        ancestors = []
        while child in self.child_to_parent:
            child = self.child_to_parent[child]
            ancestors.append(child)
        return ancestors

    def get_all_subordinates(self, top_node: str) -> list[str]:
        subordinates = [top_node]
        
        def dfs(node):
            if node in self.parent_to_children:
                for child in self.parent_to_children[node]:
                    subordinates.append(child)
                    dfs(child)

        dfs(top_node)

        # Sort by the numerical value after "L"
        sorted_subordinates = sorted(subordinates, key=lambda x: int(x.split('_')[0][1:]))
        return sorted_subordinates

    