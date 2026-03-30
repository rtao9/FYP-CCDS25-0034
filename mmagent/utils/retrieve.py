import json
import re
import logging
import random
from mmagent.utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
    get_embedding_with_retry,
)
from mmagent.utils.general import validate_and_fix_python_list
from .prompts import *
from mmagent.memory_processing import parse_video_caption
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

def translate(video_graph, memories):
    new_memories = []
    for memory in memories:
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        entities = list(set(entities))
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.reverse_character_mappings.keys():
                new_memory = new_memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories

def back_translate(video_graph, queries):
    translated_queries = []
    for query in queries:
        entities = parse_video_caption(video_graph, query)
        entities = list(set(entities))
        to_be_translated = [query]
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.character_mappings.keys():
                mappings = video_graph.character_mappings[entity_str]
                
                # Create new queries for each mapping
                new_queries = []
                for mapping in mappings:
                    for partially_translated in to_be_translated:
                        new_query = partially_translated.replace(entity_str, mapping)
                        new_queries.append(new_query)
                
                # Update translated_query with all variants
                to_be_translated = new_queries
                
        # Add all variants of the translated query
        translated_queries.extend(to_be_translated)
    return translated_queries

# retrieve by clip
def retrieve_from_videograph(video_graph, query, topk=5, mode='max', threshold=0, before_clip=None):
    top_clips = []
    # find all CLIP_x in query
    pattern = r"CLIP_(\d+)"
    matches = re.finditer(pattern, query)
    top_clips = []
    for match in matches:
        try:
            clip_id = int(match.group(1))
            top_clips.append(clip_id)
        except ValueError:
            continue
    
    queries = back_translate(video_graph, [query])
    if len(queries) > 100:
        logger.error(f"Anomaly detected from query: {query}, randomly sample 100 translated queries")
        queries = random.sample(queries, 100)
    
    related_nodes = get_related_nodes(video_graph, query)

    query_embeddings = parallel_get_embedding(queries)[0]

    full_clip_scores = {}
    clip_scores = {}

    if mode not in ['sum', 'max', 'mean']:
        raise ValueError(f"Unknown mode: {mode}")

    # calculate scores for each node
    nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode='max')
    print(f"DEBUG: Found {len(nodes)} raw nodes. Best score: {nodes[0][1] if nodes else 'N/A'}")
    
    # collect node scores for each clip
    for node_id, node_score in nodes:
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        if clip_id not in full_clip_scores:
            full_clip_scores[clip_id] = []
        full_clip_scores[clip_id].append(node_score)

    # calculate scores for each clip
    for clip_id, scores in full_clip_scores.items():
        if mode == 'sum':
            clip_score = sum(scores)
        elif mode == 'max':
            clip_score = max(scores)
        elif mode == 'mean':
            clip_score = np.mean(scores)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        clip_scores[clip_id] = clip_score

    # sort clips by score
    sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)
    # filter out clips that have 0 score and get top k clips
    if before_clip is not None:
        top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold and clip_id <= before_clip][:topk]
    else:
        top_clips = [clip_id for clip_id, score in sorted_clips if score >= threshold][:topk]
    print(f"DEBUG: Found {len(top_clips)} good nodes.")
    return top_clips, clip_scores, nodes

def get_related_nodes(video_graph, query):
    related_nodes = []
    entities = parse_video_caption(video_graph, query)
    for entity in entities:
        type = entity[0]
        node_id = entity[1]
        if not (f"{type}_{node_id}" in video_graph.character_mappings.keys() or f"{type}_{node_id}" in video_graph.reverse_character_mappings.keys()):
            continue
        if type == "character":
            related_nodes.extend([int(node.split("_")[1]) for node in video_graph.character_mappings[f"{type}_{node_id}"]])
        else:
            related_nodes.append(node_id)
    return list(set(related_nodes))

def select_queries(action_content, responses):
    if not action_content:
        return None
    
    history_queries = [response["action_content"] for response in responses]
    history_embeddings = parallel_get_embedding(history_queries)[0]
    
    queries = action_content
    embeddings = parallel_get_embedding(queries)[0]
    
    # If there are no history queries, return the first query
    if not history_queries:
        return queries[0]
    
    # Calculate cosine similarity between each query and all history queries
    avg_similarities = []
    for query_embedding in embeddings:
        similarities = []
        for history_embedding in history_embeddings:
            # Compute cosine similarity
            dot_product = sum(a*b for a,b in zip(query_embedding, history_embedding))
            query_norm = sum(a*a for a in query_embedding) ** 0.5
            history_norm = sum(b*b for b in history_embedding) ** 0.5
            cos_sim = dot_product / (query_norm * history_norm)
            similarities.append(cos_sim)
        # Calculate average similarity for this query
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarities.append(avg_similarity)
    
    # Return query with lowest average similarity
    min_similarity_idx = avg_similarities.index(min(avg_similarities))
    return queries[min_similarity_idx]

def search(video_graph, query, current_clips, topk=5, mode='max', threshold=0, mem_wise=False, before_clip=None, episodic_only=False):
    top_clips, clip_scores, nodes = retrieve_from_videograph(video_graph, query, topk, mode, threshold, before_clip)
    
    if mem_wise:
        new_memories = {}
        top_nodes_num = 0
        # fetch top nodes
        for top_node, _ in nodes:
            clip_id = video_graph.nodes[top_node].metadata['timestamp']
            if before_clip is not None and clip_id > before_clip:
                continue
            if clip_id not in new_memories:
                new_memories[clip_id] = []
            new_ = translate(video_graph, video_graph.nodes[top_node].metadata['contents'])
            new_memories[clip_id].extend(new_)
            top_nodes_num += len(new_)
            if top_nodes_num >= topk:
                break
        # sort related_memories by timestamp
        new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
        new_memories = {f"CLIP_{k}": v for k, v in new_memories.items() if len(v) > 0}
        return new_memories, current_clips, clip_scores
    
    new_clips = [top_clip for top_clip in top_clips if top_clip not in current_clips]
    new_memories = {}
    current_clips.extend(new_clips)
    
    for new_clip in new_clips:
        if new_clip not in video_graph.text_nodes_by_clip:
            new_memories[new_clip] = [f"CLIP_{new_clip} not found in memory bank, please search for other information"]
        else:
            related_nodes = video_graph.text_nodes_by_clip[new_clip]
            new_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes if (not episodic_only or video_graph.nodes[node_id].type != "semantic")])
                        
    # sort related_memories by timestamp
    new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
    new_memories = {f"CLIP_{k}": v for k, v in new_memories.items()}
    
    return new_memories, current_clips, clip_scores

def calculate_similarity(mem, query, related_nodes):
    related_nodes_embeddings = np.array([mem.nodes[node_id].embeddings[0] for node_id in related_nodes])
    query_embedding = np.array(get_embedding_with_retry(query)[0]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, related_nodes_embeddings)[0]
    return similarities.tolist()

def retrieve_all_episodic_memories(video_graph):
    episodic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "episodic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in episodic_memories:
                episodic_memories[clips_id] = []
            episodic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return episodic_memories

def retrieve_all_semantic_memories(video_graph):
    semantic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "semantic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in semantic_memories:
                semantic_memories[clips_id] = []
            semantic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return semantic_memories


if __name__ == "__main__":
    pass