# evaluate_synthetic_inputs
# evaluate_context
# evaluate_synthetic_scenarios


class FilterTemplate:
    @staticmethod
    def evaluate_synthetic_inputs(query):
        return f"""Evaluate the provided synthetic query (which may be a question, task, or instruction) for clarity and answerability, assuming sufficient domain knowledge. Use the following criteria to guide your assessment:

        1. **Self-Containment**: Can the query be understood and completed without needing additional context or external references not provided within the query itself? It should be self-sufficient, meaning it doesn't depend on specific documents, tables, or prior knowledge not included in the query.
        2. **Clear Objective**: Does the query clearly convey its intent? It should specify what information, action, or response is being requested, allowing for a direct and appropriate answer or execution without ambiguity.

        Based on these criteria, assign a score between 0 and 1, where:
        - "1" means the query is clear, self-contained, and answerable.
        - "0" means the query is vague, relies on external references, or is unclear in its intent.
        - Scores between 0 and 1 indicate partial clarity or answerability, where the query meets some but not all of the criteria.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'feedback' and 'score' keys.

        Example query: "What technological innovations have changed communication over the last 20 years?"
        Example JSON:
        {{
            "feedback": "The query is somewhat vague as it asks about 'technological innovations' without specifying particular areas of communication (e.g., social media, messaging apps). It could be improved by narrowing the focus to a specific type of innovation or timeframe.",
            "score": 0.5
        }}

        Example query: "Explain the impact of renewable energy policies in Germany on local economies in 2021."
        Example JSON:
        {{
            "feedback": "This query clearly specifies the focus (renewable energy policies), the region (Germany), and the timeframe (2021). It is self-contained and answerable without needing additional context, making it clear and effective.",
            "score": 1.0
        }}

        Example query: "What are the main criticisms of the current education system in the United States?"
        Example JSON:
        {{
            "feedback": "The question is broad and lacks specificity, as 'main criticisms' could refer to various aspects (e.g., funding, curriculum, access). To improve clarity, it could specify which aspect of the education system is being critiqued.",
            "score": 0.4
        }}

        Example query: "Discuss the role of AI in healthcare, particularly in diagnostics, as noted in the last report."
        Example JSON:
        {{
            "feedback": "This question refers to 'the last report' without providing context or details, making it unclear and dependent on external information. It would be clearer if it provided some background on the report or defined what aspects of AI in diagnostics to address.",
            "score": 0.3
        }}
                
        The `feedback` MUST be a STRING and `score` must be a float from 0 to 1.
        **
                
        Query:
        {query}

        JSON:
        """

    @staticmethod
    def evaluate_context(context):
        return f"""Given a context, complete the following task and return the result in VALID JSON format: Evaluate the supplied context and assign a numerical score between 0 (Low) and 1 (High) for each of the following criteria in your JSON response:

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

    @staticmethod
    def evaluate_synthetic_scenarios(scenario):
        return f"""Evaluate the provided conversational scenario for clarity, conversational nature, and appropriateness. Use the following criteria:

        1. **Conversational Structure**: Does the scenario describe an actual conversation between identified participants (not just a question or prompt)?
        2. **Participant Clarity**: Are the participants clearly identified with specific roles (e.g., "teacher and student", "doctor and patient")?
        3. **Contextual Setting**: Is there a clear setting or context for when/where/why this conversation occurs?
        4. **Purposeful Interaction**: Does the scenario imply a multi-turn dialogue with a goal or purpose?
        5. **Naturalness**: Could this conversation realistically occur in the described situation?

        Assign a score between 0 and 1:
        - "1" = Perfect conversational scenario with clear participants, setting, and purpose
        - "0.7-0.9" = Good scenario with minor issues (slightly vague participants or context)
        - "0.4-0.6" = Mediocre scenario (missing clear participants OR setting OR purpose)
        - "0-0.3" = Poor scenario (just a question/prompt, or very vague)

        **
        IMPORTANT: Return JSON format only, with 'feedback' and 'score' keys.

        Example scenario: "A student asks about homework"
        Example JSON:
        {{
            "feedback": "This scenario is too vague. It doesn't specify what subject, what specific aspect of homework, who the student is asking (teacher? parent?), or provide conversational context. It reads more like a prompt fragment than a conversational scenario. Needs: specific participants, setting, and what aspect of homework is being discussed.",
            "score": 0.2
        }}

        Example scenario: "A new employee meets with their supervisor for a quarterly performance review to discuss progress over the first three months, areas for improvement, and goals for the next quarter"
        Example JSON:
        {{
            "feedback": "Excellent conversational scenario. Clear participants (new employee and supervisor), specific setting (quarterly performance review), clear purpose (discuss progress, improvements, and goals), and describes a realistic multi-turn conversation. This would naturally involve back-and-forth dialogue.",
            "score": 1.0
        }}

        Example scenario: "A patient explains symptoms to a doctor during a check-up"
        Example JSON:
        {{
            "feedback": "Good conversational scenario with clear participants (patient and doctor) and setting (check-up). It describes a realistic interaction. Could be slightly more specific about what symptoms or the purpose beyond just explaining, but overall solid.",
            "score": 0.8
        }}

        Example scenario: "Explain how photosynthesis works"
        Example JSON:
        {{
            "feedback": "This is an instruction/prompt, not a conversational scenario. It doesn't describe who is talking to whom, what the setting is, or frame it as an interaction. Needs complete rewrite to describe an actual conversation (e.g., 'A biology teacher explains photosynthesis to a student who asks about...').",
            "score": 0.1
        }}

        Example scenario: "Two colleagues discuss the benefits and drawbacks of remote work versus office work during their lunch break"
        Example JSON:
        {{
            "feedback": "Strong conversational scenario. Clear participants (two colleagues), setting (lunch break), specific topic (remote vs office work), and implies a natural back-and-forth discussion. Realistic and purposeful.",
            "score": 0.95
        }}
                
        The `feedback` MUST be a STRING and `score` must be a float from 0 to 1.
        **
                
        Scenario:
        {scenario}

        JSON:
        """
