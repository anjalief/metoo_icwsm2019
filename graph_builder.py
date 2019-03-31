# Helper classes for building power graphs
# Used in metoo_analysis.py
from collections import Counter, defaultdict

class EntityScoreTracker:
    def __init__(self):
        # parallel maps track verb scores and number of verbs
        self.score = 0.0
        self.count = 0
        self.is_missing = False

    def update(self, verb_score, verb_count, is_missing):
        self.score += verb_score
        self.count += verb_count
        self.is_missing = self.is_missing or is_missing

class Edge:
    def __init__(self):
        self.score = 0.0
        self.count = 0
        self.missing = False

    def update(self, n_score, n_count, is_missing):
        self.score += n_score
        self.count += n_count
        self.missing = self.missing or is_missing

    def get_weight(self):
        return self.score / self.count

class EdgeTracker:
    def __init__(self):
        self.ents_to_edge = defaultdict(Edge)
        self.vertices = set()

    def update(self, e1, e2, e1_score, e2_score, is_missing):
        if e1 == e2:
            assert(e1_score == e2_score)
            return
        self.vertices.add(e1)
        self.vertices.add(e2)

        # be consist about key ordering
        if e1 < e2:
            key = (e1, e2)
            value = e1_score - e2_score
        else:
            key = (e2, e1)
            value = e2_score - e1_score

        self.ents_to_edge[key].update(value, 1, is_missing)

    def get_edge_list(self, to_delete):
        # This is the final vertex list. It should not change
        vertices = [x for x in self.vertices if not x in to_delete]
        vertex_to_idx = {}
        for i,v in enumerate(vertices):
            vertex_to_idx[v] = i

        edges = []
        for e in self.ents_to_edge:
            if not e[0] in to_delete and not e[1] in to_delete:
                edges.append(e)

        missing = [self.ents_to_edge[e].missing for e in edges]
        weights = [self.ents_to_edge[e].get_weight() for e in edges]

        # Now we need to make all edge weights positive
        for i,e in enumerate(edges):
            if weights[i] < 0:
                weights[i] = abs(weights[i])
                edges[i] = (e[1], e[0])
        # Now the edge list has been finalized

        # Then we need to compute vertex weights
        vertex_to_score = defaultdict(float)
        for i,e in enumerate(edges):

            # sum outgoing edges and subtract incoming edges
            vertex_to_score[e[0]] += weights[i]
            vertex_to_score[e[1]] -= weights[i]

        vertex_weights = [vertex_to_score[v] for v in vertices]

        # the graph packages wants numbered vertices
        # do this after we flip edges
        enumerated_edges = []
        for e in edges:
            enumerated_edges.append((vertex_to_idx[e[0]], vertex_to_idx[e[1]]))


        return enumerated_edges, weights, vertices, vertex_weights, missing
