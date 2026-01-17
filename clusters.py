# Clustering training examples, gathering medoids
from collections import defaultdict
from dataclasses import dataclass
import sqlglot
from sqlglot import exp
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from langchain_ollama import OllamaEmbeddings
import json

#This clustering script was generated using ChatGPT with the following series of prompts:
########################################################################################
# I want to explore selecting examples based on representativeness and informativeness across the training set. A function which selects examples based on this criteria would ensure that a wide coverage on types of natural queries and also on types of SQL commands would be present in the examples. This would be similar to an active learning setup. My intuition is to use some measure of surprise to decide which samples to add to the example set. How might I go about designing such an example selection method?

# ...

# Let's focus on representativeness and a python implementation, clustering on the NL query embeddings using scikit learn clustering on each domain / database, then clustering on the SQL. When making SQL templates I need to keep the original SQL along with the data point so I can retrieve the original example for few shot examples.

# ...

# Demonstrate an example of this pipeline, starting from a json file containing the examples, through the clustering, and finally returning the representative samples as NL-SQL pairs.

# ...

# How would I instead cluster on the SQL templates, and then select the medoid and the top-k nearest examples? Also, the SQL templates are too unique for clustering, how can we simplify their representation?

# ...

# in the sql_signature function, I am getting an error for len(tree.find_all(exp.Select)), TypeError: object of type 'generator' has no len()

#######################################################################################################

# ---- From https://github.com/ollama/ollama/issues/4128, normalizing OllamaEmbeddings
class OllamaEmbeddingsNormalized(OllamaEmbeddings):
    def _process_emb_response(self, input: str) -> list[float]:
        emb = super()._process_emb_response(input)
        return (np.array(emb) / np.linalg.norm(emb)).tolist()

@dataclass
class QueryExample:
    idx: int
    database: str
    nl: str
    sql: str
    nl_embedding: np.ndarray
    sql_embedding: np.ndarray
    sql_features: np.ndarray


def cluster_nl(examples, n_clusters):
    print("clustering NL queries.... \n")
    X = np.vstack([e.nl_embedding for e in examples])
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(X)
    return labels


def cluster_sql_features(examples, n_clusters):
    X = np.vstack([e.sql_features for e in examples])
    k = min(n_clusters, len(examples))
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X)
    return labels

def cluster_sql_embeddings(examples, n_clusters):
    X = np.vstack([e.sql_embedding for e in examples])
    k = min(n_clusters, len(examples))
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X)
    return labels

def load_examples(path, embedding="nomic-embed-text"):
    with open(path) as f:
        raw = json.load(f)

    model = OllamaEmbeddingsNormalized(model=embedding)

    nls = [ex["question"] for ex in raw]
    nl_embeddings = [model.embed_query(n) for n in nls]
    sqls = [ex["query"] for ex in raw]
    sql_embeddings = [model.embed_query(q) for q in sqls]

    examples = []
    for i, (ex, nlemb, sqlemb) in enumerate(zip(raw, nl_embeddings, sql_embeddings)):
        sql_features = sql_signature(ex["query"])
        examples.append(
            QueryExample(
                idx=i,
                database=ex["db_id"],
                nl=ex["question"],
                sql=ex["query"],
                nl_embedding=nlemb,
                sql_embedding=sqlemb,
                sql_features=sql_features,
            )
        )

    return examples

AGG_FUNCS = {
    exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max
}

def sql_signature(sql: str) -> np.ndarray:
    tree = sqlglot.parse_one(sql)

    return np.array([
        len(list(tree.find_all(exp.Select))),                  # selects
        len(list(tree.find_all(exp.Join))),                    # joins
        len(list(tree.find_all(exp.Where))),                   # where clauses
        len(list(tree.find_all(exp.Group))),                   # group by
        len(list(tree.find_all(exp.Having))),                  # having
        len(list(tree.find_all(exp.Subquery))),                # subqueries
        sum(isinstance(n, tuple(AGG_FUNCS)) for n in tree.walk()),
        len(list(tree.find_all(exp.Order))),                   # order by
        len(list(tree.find_all(exp.Limit))),                   # limit
    ], dtype=float)


def select_medoid_and_neighbors(examples, k_neighbors=3, by_feature=True):
    if by_feature:
        X = np.vstack([e.sql_features for e in examples])
    else:
        X = np.vstack([e.sql_embedding for e in examples])
    D = pairwise_distances(X)

    medoid_idx = np.argmin(D.sum(axis=0))
    distances = D[medoid_idx]

    nearest_idxs = np.argsort(distances)[1:k_neighbors+1]

    return examples[medoid_idx], [examples[i] for i in nearest_idxs]


def select_sql_representatives(
    examples,
    sql_clusters_per_db=5,
    k_neighbors=2,
    by_feature=True,
):
    selected = []

    by_db = defaultdict(list)
    for e in examples:
        by_db[e.database].append(e)

    for db, db_examples in by_db.items():
        if by_feature:
            labels = cluster_sql_features(db_examples, sql_clusters_per_db)
        else:
            labels = cluster_sql_embeddings(db_examples, sql_clusters_per_db)

        for cluster_id in set(labels):
            sql_cluster = [
                e for e, l in zip(db_examples, labels) if l == cluster_id
            ]

            medoid, neighbors = select_medoid_and_neighbors(
                sql_cluster, k_neighbors, by_feature=by_feature,
            )

            selected.append(medoid)
            selected.extend(neighbors)

    return selected