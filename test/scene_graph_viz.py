# vis_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class Node(BaseModel):
    id: str
    label: str

class Edge(BaseModel):
    source: str
    target: str
    label: str

class SceneGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

current_graph = SceneGraph(nodes=[], edges=[])

@app.post("/update")
def update_graph(graph: SceneGraph):
    global current_graph
    current_graph = graph
    return {"status": "updated"}

@app.get("/graph")
def get_graph():
    return current_graph


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
