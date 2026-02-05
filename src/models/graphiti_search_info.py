from pydantic import BaseModel, Field
from typing import Optional


class GraphitiEdgeInfo(BaseModel):
    uuid: str = Field(description="The unique identifier for this fact")
    fact: str = Field(
        description="The factual statement retrieved from the knowledge graph"
    )
    valid_at: Optional[str] = Field(
        None, description="When this fact became valid (if known)"
    )
    invalid_at: Optional[str] = Field(
        None, description="When this fact became invalid (if known)"
    )
    group_id: Optional[str] = Field(
        None, description="The group identifier for the edge"
    )


class GraphitiNodeInfo(BaseModel):
    uuid: str = Field(description="The unique identifier for this node")
    summary: Optional[str] = Field(None, description="The summary of the node")
    group_id: Optional[str] = Field(
        None, description="The group identifier for the node"
    )


class GraphitiEpisodeInfo(BaseModel):
    uuid: str = Field(description="The unique identifier for this episode")
    content: str = Field(description="The content of the episode")
    group_id: Optional[str] = Field(
        None, description="The group identifier for the episode"
    )
