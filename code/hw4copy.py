#!/usr/bin/env python

import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random
import math

class Graph():
    """
    A class for creating graph objects. A graph will be
    defined by (i) a set of vertices and (ii) a dictionary
    mapping each vertex Vi to its set of parents pa_G(Vi).
    """

    def __init__(self, vertices, edges=set()):
        """
        Constructor for the Graph class. A Graph is created
        by accepting a set of vertices and edges as inputs

        Inputs:
        vertices: a set/list of vertex names
        edges: a set/list of tuples for example [(Vi, Vj), (Vj, Vk)] where the tuple (Vi, Vj) indicates Vi->Vj exists in G
        """

        self.vertices = [vertex for vertex in vertices]

        # dictionary mapping each vertex to its parents
        self.parents = defaultdict(list)
        for parent, child in edges:
            self.parents[child].append(parent)

    def vertices(self):
        return self.vertices

    def parentss(self):
        return self.parents

    def add_edge(self, parent, child):
        """
        Function to add an edge to the graph from parent -> child
        """

        self.parents[child].append(parent)

    def delete_edge(self, parent, child):
        """
        Function to delete an edge to the graph from parent -> child
        """

        if parent in self.parents[child]:
            self.parents[child].remove(parent)

    def edges(self):
        """
        Returns a list of tuples [(Vi, Vj), (Vx, Vy),...] corresponding to edges
        present in the graph
        """

        edges = []
        for v in self.vertices:
            edges.extend([(p, v) for p in self.parents[v]])



        return edges

    def produce_visualization_code(self, filename):
        """
        Function that outputs a text file with the necessary graphviz
        code that can be pasted into https://dreampuf.github.io/GraphvizOnline/
        to visualize the graph.
        """

        # set up a Digraph object in graphviz
        gviz_file = open(filename, "w")
        gviz_file.write("Digraph G { \n")

        # iterate over all vertices and add them to the graph
        for v in self.vertices:
            gviz_file.write('  {} [shape="plaintext"];\n'.format(v))

        # add edges between the vertices
        for v in self.vertices:
            for p in self.parents[v]:
                gviz_file.write('  {} -> {} [color="blue"];\n'.format(p, v))

        # close the object definition and close the file
        gviz_file.write("}\n")
        gviz_file.close()



def acyclic(G):
    for v in G.vertices:
        if dfs(G,v) == True:
            return False
    return True


"true is start was found in dfs"
def dfs(G, start):
    visited = {}
    stack = []
    for vert in G.vertices:
        visited[vert] = False
    stack.append(start)
    while len(stack) > 0:
        s = stack.pop()
        if (not visited[s]):
            #print(s)
            visited[s] = True
            #print(visited)
        for e in G.edges():
            if e[0] == s:
                if visited[e[1]]:
                    if e[1] == start:
                        return True
                else:
                    stack.append(e[1])
    return False

#    """
#     A function that uses depth first traversal to determine whether the
#     graph G is acyclic.
#     """
#     visited = []
#     stack = []
#     stack.append(G.vertices[0])
#
#     while(len(stack) != 0):
#         s = stack.pop()
#         if s not in visited:
#             visited.append(s)
#         else:
#             return False
#         for e in G.edges():
#             if e[0] == s:
#                 stack.append(e[1])
#     return True

def search_helper(G, start, visited):
    pass


def bic_score(G, data):
    """
    Compute the BIC score for a given graph G and a dataset provided as a pandas data frame.

    Inputs:
    G: a Graph object as defined by the Graph class above
    data: a pandas data frame
    """


    #find the parents of each variable
    #run a linear regression predicting the given variable using the parents as explanatory variables
    #ask for liklihood
    sum = 0
    sum1 = 0
    for col in data.columns:
        #print(col)
        p = G.parentss()
        p = p[col]
        X = data[p]
        X = sm.add_constant(X)

        model = sm.OLS(data[col], X).fit()
        loggliklisum = model.llf
        res = (-2*loggliklisum) + len(data.columns) * math.log(len(data.index))
        sum1 += res
        score = model.bic
        sum += score

    return sum1



def causal_discovery(data, num_steps=50):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    """

    # initalize an empty graph as the optimal one and gets its BIC score
    G_star = Graph(vertices=data.columns)
    bic_star = bic_score(G_star, data)
    # print(bic_star, "initial")
    # quit()
    var = None
    # forward phase of causal discovery:

    last_edge = ("","")
    i = 0
    for i in range(num_steps):
        print("i:", i)
        while True:
            one = random.randint(0, len(G_star.vertices) - 1)
            two = random.randint(0, len(G_star.vertices) - 1)
            verts = G_star.vertices
            if (one != two) and ((verts[one],  verts[two]) != last_edge):
                break
        verts = G_star.vertices
        vertone = verts[one]
        verttwo = verts[two]
        if not((vertone, verttwo) in G_star.edges()):
            G_clone = Graph(G_star.vertices, G_star.edges())
            G_clone.add_edge(vertone, verttwo)
            if acyclic(G_clone):
                assert acyclic(G_clone)
                #good = True
                G_star.add_edge(vertone, verttwo)
                assert acyclic(G_star)
                #i += 1 #now we know we're completing a step and not just restarting the loop
                bic_starNew  = bic_score(G_star, data)
                if bic_starNew < bic_star:
                    # print("happy")
                    bic_star = bic_starNew
                else:
                    G_star.delete_edge(vertone, verttwo) #undo change
            else:
                last_edge = (vertone, verttwo)
                continue
        else:
            last_edge = (vertone, verttwo)
            continue
    g = 0
    while g < num_steps:
        print("g:", g)
        #print(i)
        edgess = G_star.edges()
        doo = len(G_star.edges()) - 1
        one = random.randint(0, doo)
        # print(one)
        # print(edgess)
        eOne = edgess[one]
        #eTwo = edgess[two]
        delete = Graph(G_star.vertices, G_star.edges()) #copy
        rev = Graph(G_star.vertices, G_star.edges())
        delete.delete_edge(eOne[0], eOne[1])
        rev.delete_edge(eOne[0], eOne[1])
        rev.add_edge(eOne[1], eOne[0])
        d_score = bic_score(delete, data)
        r_score = bic_score(rev, data)
        #update to smallest bic value
        if not(acyclic(rev)):
            r_score = math.inf
        if d_score < bic_star:
            bic_star = d_score
            G_star = delete
        if r_score < bic_star:
            bic_star = r_score
            G_star = rev
        g += 1
    return G_star



def main():
    ################################################
    # Tests for your acyclic function
    ################################################

    # G = X->Y<-Z, Z->X
    G1 = Graph(vertices=["X", "Y", "Z"], edges=[("X", "Y"), ("Z", "Y"), ("Z", "X")])

    # X->Y->Z, Z->X
    G2 = Graph(vertices=["X", "Y", "Z"], edges=[("X", "Y"), ("Y", "Z"), ("Z", "X")])
    # print(dfs(G2, 'X'))
    # quit()

    # X->Y->Z, Y->Y
    G3 = Graph(vertices=["X", "Y", "Z"], edges=[("X", "Y"), ("Y", "Z"), ("Y", "Y")])

    print(acyclic(G1), "TRUE")
    print(acyclic(G2), "FALSE")
    print(acyclic(G3), "FALSE")


    ################################################
    # Tests for your bic_score function
    ################################################
    data = pd.read_csv("bic_test_data.txt")

    # fit model for G1: A->B->C->D, B->D and get BIC
    G1 = Graph(vertices=["A", "B", "C", "D"], edges=[("A", "B"), ("B", "C"), ("C", "D"), ("B", "D")])

    print("G1", bic_score(G1, data), acyclic(G1))
    G1.produce_visualization_code("G1_viz.txt")

    # fit model for G2: A<-B->C->D, B->D and get BIC
    G2 = Graph(vertices=["A", "B", "C", "D"], edges=[("B", "A"), ("B", "C"), ("C", "D"), ("B", "D")])
    print("G2", bic_score(G2, data), acyclic(G2))

    # fit model for G3: A->B<-C->D, B->D and get BIC
    G3 = Graph(vertices=["A", "B", "C", "D"], edges=[("A", "B"), ("C", "B"), ("C", "D"), ("B", "D")])
    print("G3", bic_score(G3, data), acyclic(G3))

    # fit model for G4: A<-B->C<-D, B->D and get BIC
    G4 = Graph(vertices=["A", "B", "C", "D"], edges=[("B", "A"), ("B", "C"), ("D", "C"), ("B", "D")])
    print("G4", bic_score(G4, data), acyclic(G4))



    ################################################
    # Tests for your causal_discovery function
    ################################################
    np.random.seed(42)
    random.seed(42)
    data = pd.read_csv("data.txt")
    G_opt = causal_discovery(data, 1000)
    print(acyclic(G_opt))

    print("output")
    print("V", G_opt.vertices)
    print("E", G_opt.edges())
    print(bic_score(G_opt, data))
    #you can paste the code in protein_viz.txt into the online interface of Graphviz
    G_opt.produce_visualization_code("protein_viz.txt")


if __name__ == "__main__":
    main()
