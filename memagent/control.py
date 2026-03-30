import re
import os
import sys
import json
import time
import openai
import argparse
import multiprocessing
import mmagent.videograph
from mmagent.utils.retrieve import search
# from vllm import LLM, SamplingParams

import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages
#from mmagent.utils.prompts import prompt_agent_verify_answer_referencing

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))
model_name = "models/Qwen3-VL-4B-Instruct"
config = json.load(open("configs/api_config.json"))
gpt_model = "gpt-4o-2024-11-20"
client= None

def get_response(messages, timeout=30):
    response = client.chat.completions.create(
        model=gpt_model, messages=messages, temperature=0, timeout=timeout, max_tokens=2048
    )
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(messages, timeout=30):
    for i in range(20):
        try:
            return get_response(messages, timeout)
        except Exception as e:
            time.sleep(20)
            print(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after 5 retries")

system_prompt = "You are given a question, choices and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer, which has to be one of the options provided. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question} \nChoices: {choices}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}
Choices: {{choices}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>."""

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# sampling_params = SamplingParams(
#     temperature=0.6,
#     top_p=0.95,
#     top_k=20,
#     max_tokens=1024
# )
pattern = r"Action: \[(.*)\].*Content: (.*)"

def consumer(data):
    if not data["finish"]:
        before_clip = data.get("before_clip", None)
        response = data["conversations"][-1]["content"]
        match_result = re.search(pattern, response.split("</think>")[-1], re.DOTALL)

        if match_result:
            action = match_result.group(1)
            content = match_result.group(2)
        else:
            action = "Search"
            content = None

        if action == "Answer":
            data["response"] = content
            data["finish"] = True
        else:
            new_memories = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if before_clip is not None:
                    mem_node.truncate_memory_by_clip(before_clip, False)
                mem_node.refresh_equivalences()
                if "character id" in content:
                    memories, _, _ = search(mem_node, content, [], mem_wise=True, topk=20, before_clip=before_clip)
                    new_memories.update(memories)
                else:
                    memories, currenr_clips, _ = search(mem_node, content, data["currenr_clips"], threshold=0.4, topk=processing_config["topk"], before_clip=before_clip)
                    data["currenr_clips"] = currenr_clips
                    new_memories.update(memories)
            search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            
            if len(new_memories) == 0:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
    return data


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, 
        dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # --- 2. The ReAct Loop ---
    user_question = """How many people were called to the human resources office?"""
    choices = """A. 1, B. 2, C. 3, D. 4"""

    data = {
        "question": user_question,
        "mem_path": "data/test/processed/memory_graphs/test.pkl",
        "finish": False,
        "currenr_clips": [],
        "conversations": [
            {"role": "user", "content": system_prompt.format(question=user_question, choices=choices) + instruction}
        ]
    }

    num_tries = 1

    for round_num in range(num_tries):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Format prompt using the official template
        text = processor.apply_chat_template(data["conversations"], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.6)
        generated_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Append response to conversation
        data["conversations"].append({"role": "assistant", "content": generated_text})
        
        # --- CRITICAL STEP: Call your consumer function ---
        print(f"{generated_text}")
        data = consumer(data)
        
        if data["finish"]:
            print(f"\nFINAL ANSWER: {data['response']}")
            break
        else:
            # If not finished, the last item in conversations is now the "Searched knowledge"
            print(f"Agent requested search. Knowledge retrieved: {len(data['conversations'][-1]['content'])} chars.")

    if not data["finish"]:
       # Update the prompt to FORCE a choice from A, B, C, or D
        force_answer_prompt = f"""
            Based on the knowledge retrieved above, you MUST now provide a final answer. No more [Search] action allowed. Only action allowed is [Answer]
            Choose the most likely option from the choices: A, B, C, or D.

            Output the answer in the format:
            Action: [Answer]
            Content: <Choice>
        """
        
        data["conversations"].append({"role": "user", "content": force_answer_prompt})
        
        text = processor.apply_chat_template(data["conversations"], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.1) # Low temp for final pick
        generated_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        print(f"\nFORCED RESPONSE: {generated_text}")
        data = consumer(data)
        
        # Safe access to 'response'
        final_ans = data.get("response", "Agent failed to provide an answer.")
        print(f"\nFINAL ANSWER: {final_ans}")