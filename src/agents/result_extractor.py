"""Utility to extract tool calls, retrieval info, and citations from pydantic_ai results."""

import time
from typing import List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

from pydantic_ai import RunResult
from src.models import (
    ToolCallInfo,
    TurnMetadata,
    RetrievalInfo,
    GraphitiNodeInfo,
    GraphitiEdgeInfo,
    GraphitiEpisodeInfo,
)


def extract_tool_calls(result: RunResult) -> List[ToolCallInfo]:
    """Extract tool calls from pydantic_ai result.
    
    Args:
        result: The RunResult from pydantic_ai agent run
        
    Returns:
        List of ToolCallInfo objects
    """
    tool_calls = []
    messages = result.all_messages()
    
    # Track tool requests and responses
    tool_requests = {}  # tool_call_id -> tool_name, arguments
    
    for msg in messages:
        # Check for tool request messages
        if hasattr(msg, 'kind'):
            if msg.kind == 'request' and hasattr(msg, 'tool_calls'):
                # This is a tool request
                for tool_call in msg.tool_calls:
                    tool_id = getattr(tool_call, 'id', None)
                    tool_name = getattr(tool_call, 'name', None)
                    arguments = getattr(tool_call, 'arguments', {})
                    
                    if tool_id and tool_name:
                        tool_requests[tool_id] = {
                            'tool_name': tool_name,
                            'arguments': arguments,
                            'start_time': time.time(),
                        }
            
            elif msg.kind == 'response' and hasattr(msg, 'content'):
                # This is a tool response
                content = msg.content
                if isinstance(content, list):
                    for part in content:
                        # Check for tool result
                        tool_name = getattr(part, 'tool_name', None)
                        tool_result = getattr(part, 'content', None)
                        tool_call_id = getattr(part, 'tool_call_id', None)
                        
                        if tool_name and tool_call_id:
                            # Find matching request
                            request_info = tool_requests.get(tool_call_id, {})
                            tool_name = request_info.get('tool_name', tool_name)
                            arguments = request_info.get('arguments', {})
                            start_time = request_info.get('start_time', time.time())
                            execution_time = time.time() - start_time
                            
                            # Determine status
                            status = "success"
                            if tool_result is None:
                                status = "error"
                            elif isinstance(tool_result, Exception):
                                status = "error"
                            
                            tool_call = ToolCallInfo(
                                tool_name=tool_name,
                                arguments=arguments,
                                result=tool_result,
                                execution_time=execution_time,
                                status=status,
                            )
                            tool_calls.append(tool_call)
                
                # Also check if content is directly a tool result
                elif hasattr(content, 'tool_name'):
                    tool_name = getattr(content, 'tool_name', None)
                    tool_result = getattr(content, 'content', None)
                    
                    if tool_name:
                        tool_call = ToolCallInfo(
                            tool_name=tool_name,
                            arguments={},
                            result=tool_result,
                            status="success" if tool_result is not None else "error",
                        )
                        tool_calls.append(tool_call)
    
    return tool_calls


def extract_retrieval_info(
    result: RunResult, 
    agent_type: str = "basic_rag"
) -> List[RetrievalInfo]:
    """Extract retrieval info from tool results.
    
    Args:
        result: The RunResult from pydantic_ai agent run
        agent_type: Type of agent ("basic_rag" or "graph_rag")
        
    Returns:
        List of RetrievalInfo objects
    """
    retrieval_infos = []
    messages = result.all_messages()
    
    for msg in messages:
        if hasattr(msg, 'kind') and msg.kind == 'response':
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                for part in msg.content:
                    tool_name = getattr(part, 'tool_name', None)
                    tool_result = getattr(part, 'content', None)
                    
                    if agent_type == "basic_rag" and tool_name == "search_basic_rag":
                        # Tool returns (retrieval_infos, answer)
                        if isinstance(tool_result, tuple) and len(tool_result) >= 1:
                            retrieval_infos = tool_result[0]
                            if isinstance(retrieval_infos, list):
                                return retrieval_infos
                    
                    elif agent_type == "graph_rag" and tool_name == "search_graphiti":
                        # Tool returns (node_infos, edge_infos, episode_infos)
                        # Convert graph results to RetrievalInfo format
                        if isinstance(tool_result, tuple) and len(tool_result) >= 3:
                            node_infos, edge_infos, episode_infos = tool_result[0], tool_result[1], tool_result[2]
                            
                            # Convert episodes to RetrievalInfo (they have content)
                            for episode in episode_infos:
                                if isinstance(episode, GraphitiEpisodeInfo):
                                    retrieval_info = RetrievalInfo(
                                        content=episode.content,
                                        source=f"episode_{episode.uuid}",
                                        score=1.0,  # Default score for graph results
                                    )
                                    retrieval_infos.append(retrieval_info)
                            
                            # Convert edges to RetrievalInfo
                            for edge in edge_infos:
                                if isinstance(edge, GraphitiEdgeInfo):
                                    retrieval_info = RetrievalInfo(
                                        content=edge.fact,
                                        source=f"edge_{edge.uuid}",
                                        score=1.0,
                                    )
                                    retrieval_infos.append(retrieval_info)
                            
                            # Convert nodes to RetrievalInfo
                            for node in node_infos:
                                if isinstance(node, GraphitiNodeInfo) and node.summary:
                                    retrieval_info = RetrievalInfo(
                                        content=node.summary,
                                        source=f"node_{node.uuid}",
                                        score=1.0,
                                    )
                                    retrieval_infos.append(retrieval_info)
                            
                            return retrieval_infos
    
    return retrieval_infos


def build_turn_metadata(
    result: RunResult,
    rag_mode: str,
    collection: str,
    retrieval_infos: Optional[List[RetrievalInfo]] = None
) -> TurnMetadata:
    """Build turn metadata from result.
    
    Args:
        result: The RunResult from pydantic_ai agent run
        rag_mode: RAG mode used ("basic", "agentic", "graph")
        collection: Collection name used
        retrieval_infos: Optional pre-extracted retrieval infos
        
    Returns:
        TurnMetadata object
    """
    import uuid
    
    # Extract tool calls
    tool_calls = extract_tool_calls(result)
    
    # Extract retrieval info if not provided
    if retrieval_infos is None:
        agent_type = "graph_rag" if rag_mode == "graph" else "basic_rag"
        retrieval_infos = extract_retrieval_info(result, agent_type)
    
    # Convert retrieval infos to citations (will be done by CitationFormatter in UI)
    # For now, we'll store retrieval_infos in metadata
    
    turn_id = str(uuid.uuid4())
    
    metadata = TurnMetadata(
        turn_id=turn_id,
        tool_calls=tool_calls,
        citations=[],  # Citations will be created by CitationFormatter
        rag_mode=rag_mode,
        collection=collection,
        timestamp=datetime.now(),
    )
    
    return metadata
