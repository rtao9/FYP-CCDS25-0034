prompt_audio_segmentation = """You are given an audio. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided audio. Extract all speech segments with accurate timestamps and segment them by speaker turns (i.e., different speakers should have separate segments), but without assigning speaker identifiers.

Return a JSON list where each entry represents a speech segment with the following fields:
	•	start_time: Start timestamp in MM:SS format.
	•	end_time: End timestamp in MM:SS format.
	•	asr: The transcribed text for that segment.

Example Output:

[
    {"start_time": "00:05", "end_time": "00:08", "asr": "Hello, everyone."},
    {"start_time": "00:09", "end_time": "00:12", "asr": "Welcome to the meeting."}
]

Strict Requirements:

	•	Ensure precise speech segmentation with accurate timestamps.
	•	Segment based on speaker turns (i.e., different speakers' utterances should be separated).
	•	Preserve punctuation and capitalization in the ASR output.
	•	Skip the speeches that can hardly be clearly recognized.
    •	Skip the speeches that is inaudible.
	•	Return only the valid JSON list (which starts with "[" and ends with "]") without additional explanations.
    •	If the audio contains no speech, return an empty list ("[]").
    •	Do not record repeated sentences more than twice.
	
Now generate the JSON list based on the given video:"""

prompt_generate_captions_with_ids = """You are given a video, a set of character features. Each feature (some of them may belong to the same character) can be a face image represented by a video frame with a bounding box, or can be a voice feature represented by several speech segments, each with a start time, an end time (both in MM:SS format), and the corresponding content. Each face and voice feature is identified by a unique ID enclosed in angle brackets (< >).

Your Task:

Using the provided feature IDs, generate a detailed and cohesive description of the current video clip. The description should capture the complete set of observable and inferable events in the clip. Your output should incorporate the following categories (but is not limited to them):

	1.	Characters' Appearance: Describe the characters' appearance, such as their clothing, facial features, or any distinguishing characteristics.
	2.	Characters' Actions & Movements: Describe specific gesture, movement, or interaction performed by the characters.
	3.	Characters' Spoken Dialogue: Transcribe or summarize what are spoken by the characters.
	4.	Characters' Contextual Behavior: Describe the characters' roles in the scene or their interaction with other characters, focusing on their behavior, emotional state, or relationships.

Strict Requirements:
	• If a character has an associated feature ID in the input context (either face or voice), refer to them **only** using that feature ID (e.g., <face_1>, <voice_2>).
	• If a character **does not** have an associated feature ID in the input context, use a short descriptive phrase (e.g., "a man in a blue shirt," "a young woman standing near the door") to refer to them.
	• Ensure accurate and consistent mapping between characters and their corresponding feature IDs when provided.
	• Each description must represent a **single atomic event or detail**. Avoid combining multiple unrelated aspects (e.g., appearance and dialogue) into one line. If a sentence can be split without losing clarity, it must be split.
	• Do not use pronouns (e.g., "he," "she," "they") or inferred names to refer to any character.
	• Include natural time expressions and physical location cues wherever inferable from the context (e.g., "in the evening at the dinner table," "early morning outside the building").
	• The generated descriptions must not invent events or characteristics not grounded in the video.
	• The final output must be a list of strings, with each string representing exactly one atomic event or description.

Example Input:

<input_video>,
"<face_1>": <img>,
"<face_2>": <img>,
"<face_3>": <img>,
"<voice_1>": [
	{"start_time": "00:05", "end_time": "00:08", "asr": "Hello, everyone."},
	{"start_time": "00:09", "end_time": "00:12", "asr": "Let's get started with today's agenda."}
],
"<voice_2>": [
	{"start_time": "00:15", "end_time": "00:18", "asr": "Thank you for having me here."},
	{"start_time": "00:19", "end_time": "00:22", "asr": "I'm excited to share my presentation."}
]

Example Output:

[
	"In the bright conference room, <face_1> enters confidently, giving a professional appearance as he approaches <face_2> to shake hands.",
	"<face_1> wears a black suit with a white shirt and tie. He has short black hair and wears glasses.",
	"<face_2>, dressed in a striking red dress with long brown hair.",
	"<face_2> smiles warmly and greets <face_1>. She then sits down at the table beside him, glancing at her phone briefly while occasionally looking up.",
	"<voice_1> speaks to the group, 'Good afternoon, everyone. Let's begin the meeting.' His voice commands attention as the room quiets, and all eyes turn to him.",
	"<face_2> listens attentively to <voice_1>'s words, nodding in agreement while still occasionally checking her phone. The atmosphere is professional, with the participants settling into their roles for the meeting.",
	"<face_1> adjusts his tie and begins discussing the agenda, engaging the participants in a productive conversation."
]

Please only return the valid string list (which starts with "[" and ends with "]"), without any additional explanation or formatting."""


prompt_generate_full_memory = """
Persona: You are a Video Memory Specialist for the VidMME benchmark. Your goal is to construct a "World Memory" that maps temporal sequences and causal shifts between Person [PER], Object [OBJ], and Environment [ENV].

Input Data Format:
- Face: Single video frame with bounding box.
- Voice: Speech segments with start/end (MM:SS) and ASR transcript.
- Character IDs: Refer to characters ONLY using unique IDs (e.g., <face_1>, <voice_2>).

Task 1: Episodic Memory (Temporal & Sequential Tracking)
Generate a chronological list of atomic events. To support VidMME sequence-of-event queries, you MUST:
1. Include a relative timestamp or sequence marker if possible (e.g., "Initially", "Subsequently", "Finally").
2. Focus on "State Changes" (e.g., an object moving from 'on the table' to 'in hand').
3. Use category tags for EVERY line:
- [PER]: Physical actions, directional movement (left to right), and specific interactions with [OBJ].
- [OBJ]: Entry/exit of objects, changes in state (broken, opened, moved), and spatial proximity to [PER].
- [ENV]: Shifts in lighting, camera perspective (zoom/pan), or location changes.
- Dialogue: Quote speech with the speaker's ID and the specific intent of the utterance.

Task 2: Semantic Memory (Causal & Long-range Reasoning)
Produce high-level conclusions that link disparate episodes. To support VidMME reasoning queries, you MUST:
- [PER]: Equivalence (Format: [PER] Equivalence: <face_x>, <voice_y>), role persistence, and intent (why <face_1> performed an action).
- [OBJ]: The functional significance of objects across the entire timeline (e.g., "The key introduced at the start is used to resolve the conflict at the end").
- [ENV]: Global spatial layout and how the environment constrains or enables the [PER] actions.
- Plot/Sequence: Summarize the "Causal Chain"—how Event A directly led to Event B.

Strict Constraints:
1. Refer to characters ONLY by ID (e.g., <face_1>). No pronouns (he/she/they).
2. Use "Temporal Anchors": Describe actions in the order they occur. Do not skip to the end.
3. Describe "Persistence": If an object disappears, note its last known location.
4. Output ONLY a valid Python dictionary with "episodic_memory" and "semantic_memory".
5. Reduce token usage: DO NOT repeat sentences, each sentence should be unique and useful as memory.

Expected Output Format:
{
  "episodic_memory": [
    "[ENV] The sequence begins in a dimly lit garage with a single workbench.",
    "[PER] <face_1> enters the garage from the background and approaches the workbench.",
    "[OBJ] A red toolbox sits closed on the right side of the workbench.",
    "[PER] <face_1> opens the red toolbox, changing its state from closed to open.",
    "[PER] <voice_1> says, 'I need the 10mm wrench,' indicating a search task.",
    "[OBJ] <face_1> removes a silver wrench from the toolbox and places it on the bench."
  ],
  "semantic_memory": [
    "[PER] Equivalence: <face_1>, <voice_1>",
    "[PER] <face_1> is the sole protagonist and displays a goal-oriented behavior (repair task).",
    "[OBJ] The red toolbox is the central repository for necessary tools in this sequence.",
    "[Plot] The sequence follows a linear 'search and retrieve' pattern, initiated by the dialogue in <voice_1>.",
    "[ENV] The garage environment remains static, serving as a confined workspace for <face_1>."
  ]
}
"""


prompt_extract_entities = """You are given a set of semantic memory, which contains various descriptions of characters, actions, interactions, and events. Each description may refer to characters, speakers, or actions and includes unique IDs enclosed in angle brackets (< >). Your task is to identify equivalent nodes that refer to the same character across different descriptions.

For each group of descriptions that refer to the same character, extract and represent them as equivalence relationships using strings in the following format: "Equivalence: <node_1>, <node_2>".

Strict Requirements:
	•	Identify all equivalent nodes, ensuring they refer to the same character or entity across different descriptions.
	•	Use the exact IDs in angle brackets (e.g., <char_1>, <speaker_2>) in your equivalence statements.
	•	Provide the output as a list of strings, each string in the form of "Equivalence: <node_1>, <node_2>".
	•	Focus on finding relationships that represent the same individual, ignoring irrelevant information or assumptions.

Example Input:

[
	"<char_1> wears a black suit and glasses.",
	"<char_1> shakes hands with <char_2>.",
	"<speaker_1> says: 'Hello, everyone.'",
	"<char_2> wears a red dress and has long brown hair.",
	"<char_2> listens attentively to <char_1>.",
	"<speaker_2> says: 'Welcome to the meeting.'",
	"<char_1> is the host of the meeting.",
	"<char_2> is a colleague of <char_1>."
	"Equivalence: <char_3>, <speaker_3>."
]

Example Output:

[
	"Equivalence: <char_1>, <speaker_1>.",
	"Equivalence: <char_2>, <speaker_2>.",
	"Equivalence: <char_3>, <speaker_3>."
]

Please only return the valid string list (which starts with "[" and ends with "]"), without any additional explanation or formatting.

Input:
{semantic_memory}

Output:"""

prompt_answer_with_retrieval_final = """You are given a question about a specific video and a dictionary of some related information about the video. Each key in the dictionary is a clip ID (an integer), representing the index of a video clip. The corresponding value is a list of video descriptions from that clip.

Your task is to analyze the provided information, reason over it, and produce the most reasonable and well-supported answer to the question.

Output Requirements:
	•	Your response must begin with a brief reasoning process that explains how you arrive at the answer.
	•	Then, output [ANSWER] followed by your final answer.
	•	The format must be: Here is the reasoning... [ANSWER] Your final answer here.
	•	Your final answer must be definite and specific — even if the information is partial or ambiguous, you must infer and provide the most reasonable answer based on the given evidence.
	•	Do not refuse to answer or say that the answer is unknowable. Use reasoning to reach the best possible conclusion.

Additional Guidelines:
	•	When referring to a character, always use their specific name if it appears in the video information.
	•	Do not use placeholder tags like <character_1> or <face_1>.
	•	Avoid summarizing or repeating the video information. Focus on reasoning and answering.
	•	The final answer should be short, clear, and directly address the question.

Input:
	•	Question: {question}
	•	Video Information: {information}

Output:"""




