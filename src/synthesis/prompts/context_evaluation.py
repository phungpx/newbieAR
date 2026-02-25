CONTEXT_EVALUATION = """Given a context, complete the following task and return the result in VALID JSON format: Evaluate the supplied context and assign a numerical score between 0 (Low) and 1 (High) for each of the following criteria in your JSON response:

- **clarity**: Assess how clear and comprehensible the information is. A score of 1 indicates that the context is straightforward and easily understandable, while a score of 0 reflects vagueness or confusion in the information presented.
- **depth**: Evaluate the extent of detailed analysis and the presence of original insights within the context. A high score (1) suggests a thorough and thought-provoking examination, while a low score (0) indicates a shallow overview of the subject.
- **structure**: Review how well the content is organized and whether it follows a logical progression. A score of 1 is given to contexts that are coherently structured and flow well, whereas a score of 0 is for those that lack organization or clarity in their progression.
- **relevance**: Analyze the importance of the content in relation to the main topic, awarding a score of 1 for contexts that stay focused on the subject without unnecessary diversions, and a score of 0 for those that include unrelated or irrelevant information.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'clarity', 'depth', 'structure', and 'relevance' keys.

Example context: "Artificial intelligence is rapidly changing various sectors, from healthcare to finance, by enhancing efficiency and enabling better decision-making."
Example JSON:
{{
    "clarity": 1,
    "depth": 0.8,
    "structure": 0.9,
    "relevance": 1
}}

Example context: "Cats are great pets. They like to sleep and play."
Example JSON:
{{
    "clarity": 0.5,
    "depth": 0.3,
    "structure": 0.4,
    "relevance": 0.5
}}

Example context: "Artificial intelligence is rapidly changing various sectors, from healthcare to finance, by enhancing efficiency and enabling better decision-making."
Example JSON:
{{
    "clarity": 1,
    "depth": 0.9,
    "structure": 1,
    "relevance": 1
}}

Example context: "Artificial intelligence is rapidly changing various sectors, from healthcare to finance, by enhancing efficiency and enabling better decision-making."
Example JSON:
{{
    "clarity": 0.4,
    "depth": 0,
    "structure": 0.3,
    "relevance": 0.2
}}

Example context: "The impact of globalization on local cultures is complex, with both positive and negative effects. It can lead to cultural exchange but also to the erosion of local traditions."
Example JSON:
{{
    "clarity": 0.9,
    "depth": 0.8,
    "structure": 0.9,
    "relevance": 1
}}


`clarity`, `depth`, `structure`, and `relevance` MUST be floats from 0 to 1.
Make sure your JSON response is valid and properly formatted.
**

context:
{context}

JSON:
"""
