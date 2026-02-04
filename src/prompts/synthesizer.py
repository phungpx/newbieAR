from typing import Optional


class SynthesizerTemplate:
    @staticmethod
    def generate_synthetic_expected_output(
        input: str,
        context: str,
        expected_output_format: Optional[str],
    ):
        important_section = (
            f"IMPORTANT: Please ensure that the generated response strictly adheres to the following format: {expected_output_format}, and make sure it is concise and straight to the point, using supporting information in context."
            if expected_output_format
            else "IMPORTANT: Please make sure to generate a response that is concise and straight to the point, and uses supporting information in context."
        )

        return f"""Given the input, which may or may not be a question, generate a response using information presented in context.

        **
        {important_section}
        **

        Context:
        {context}

        Input:
        {input}

        Generated Response:
        """

    @staticmethod
    def generate_synthetic_inputs(
        context: str,
        max_goldens_per_context: str,
        scenario: Optional[str],
        task: Optional[str],
        input_format: Optional[str],
    ):
        input_format_section = (
            f"`input` MUST strictly adhere to the following format: {input_format}."
            if input_format
            else "`input` MUST be a STRING."
        )

        scenario_section = (
            f"`input`s MUST be relevant to this specific scenario: ```{scenario}``` (The scenario describes the circumstances under which the inputs are generated and the user's intent in eliciting a response)."
            if scenario
            else ""
        )

        task_section = (
            f"`input`s MUST be framed in a way that evokes a response aligned with the following task: {task} (The task represents the goal or function the entity is expected to achieve when responding)."
            if task
            else ""
        )
        return f"""I want you act as a copywriter. Based on the given context, which is list of strings, please generate a list of JSON objects with a `input` key.
        The `input` can either be a question or a statement that can be addressed by the given context.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
        You MUST TRY to generate {max_goldens_per_context} data points, unless the `input` is getting repetitive.

        Example context: ["Einstein won the Nobel Prize for his discovery of penicillin.", "Einstein won the Nobel Prize in 1968."]
        Example max goldens per context: 2
        Example JSON:
        {{
            "data": [
                {{
                    "input": "What was Einstein known for?"
                }},
                {{
                    "input": "Einstein was a smart guy huh"
                }}
            ]  
        }}


        You should NOT incorporate any prior knowledge you have and take each context at face value.
        You MUST include at least one statement as the input.
        {input_format_section}
        {scenario_section}
        {task_section}
        You MUST TRY to generate {max_goldens_per_context} data points, unless the generated `input` is getting repetitive.
        **

        Max Goldens Per Context:
        {max_goldens_per_context}

        Context:
        {context}

        JSON:
        """

    @staticmethod
    def rewrite_evolved_input(
        evolved_input: str,
        scenario: Optional[str] = None,
        task: Optional[str] = None,
        input_format: Optional[str] = None,
    ):
        scenario_section = f'Scenario: "{scenario}"' if scenario else ""

        task_section = f'Task: "{task}"' if task else ""

        input_format_section = f'Input Format: "{input_format}"' if input_format else ""

        return f"""Given the evolved input, which may be a question or a statement, generate a JSON object with a key 'input'. This key should contain a statement or question that fits any provided scenario, aligns with the task's purpose, and matches the required input format (if specified).

        **
        IMPORTANT: Try to change the evolved input as little as possible. However, if the evolved input does not align with the provided scenario, task, or input format, it must ultimately be adjusted to fit these requirements. The output must be in JSON format, with the 'input' key only. If necessary, the evolved input should be rewritten to ensure it conforms to the scenario, task, and input format.

        Example Evolved Input: "Is it okay to joke about someone losing their job in front of their coworkers if I'm just trying to lighten the mood?"
        {f'Example Scenario: "{scenario}"' if scenario else ""}
        {f'Example Task: "{task}"' if task else ""}
        {f'Example Input Format: "{input_format}"' if input_format else ""}
        Example JSON: {{
            "input": "How can I joke about someone losing their job without making the situation worse? Is it possible to use humor here without hurting anyone's feelings?"
        }}

        Evolved Input:
        {evolved_input}
        
        {scenario_section}
        {task_section}
        {input_format_section}

        JSON:
        """

    @staticmethod
    def rewrite_synthetic_inputs(context, original_query, feedback):
        return f"""I want you to act as a query rewriter. Based on the provided context, original query, and feedback, generate a rewritten query that improves its clarity and answerability based on the feedback provided.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'rewritten_input' key.

        Example context: "The Golden Gate Bridge, located in San Francisco, was completed in 1937 and is known for its Art Deco design. It connects the city of San Francisco to Marin County and spans the Golden Gate Strait."
        Example query: "When was the bridge completed?"
        Example feedback: "The question asks about the completion of 'the bridge' but does not specify which bridge it refers to. There are many famous bridges, and without specifying the name, the question is too vague. To improve clarity, include the bridge's name."
        Example JSON:
        {{
            "rewritten_input": "When was the Golden Gate Bridge completed?"
        }}

        Example context: "The paper 'Advancements in Quantum Computing' by Dr. Alice Thompson discusses breakthroughs in quantum algorithms and was published in 2022. It explores the potential applications of quantum computing in cryptography and drug discovery."
        Example query: "What applications of quantum computing are discussed in the paper?"
        Example feedback: "The query is asking about applications of quantum computing but doesn't specify which paper is being referenced. Since many papers may discuss quantum computing, it would help to specify the title or author of the paper to improve clarity."
        Example JSON:
        {{
            "rewritten_input": "What applications of quantum computing are discussed in the paper 'Advancements in Quantum Computing' by Dr. Alice Thompson?"
        }}

        You should NOT incorporate any prior knowledge and should base the rewritten query only on the context and feedback provided.
        The `rewritten_input` MUST be a STRING.
        **

        Context:
        {context}

        Query:
        {original_query}

        Feedback:
        {feedback}

        JSON:
        """

    @staticmethod
    def generate_synthetic_scenarios(
        context: str,
        max_goldens_per_context: int,
        scenario_context: Optional[str],
        conversational_task: Optional[str],
        participant_roles: Optional[str],
    ):
        participant_section = (
            f"Each scenario MUST involve these participant roles: {participant_roles}."
            if participant_roles
            else "Each scenario MUST clearly specify who the participants are (e.g., 'a teacher and a student', 'two colleagues')."
        )

        scenario_context_section = (
            f"All scenarios MUST fit within this conversational context: {scenario_context}"
            if scenario_context
            else ""
        )

        task_section = (
            f"The conversation in each scenario should work towards this goal: {conversational_task}"
            if conversational_task
            else ""
        )

        return f"""I want you to act as a conversation scenario designer. Based on the given context, generate a list of JSON objects with a `scenario` key.
        Each `scenario` should describe a MULTI-TURN CONVERSATIONAL INTERACTION between specific participants discussing information from the context.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
        You MUST TRY to generate {max_goldens_per_context} data points, unless scenarios become repetitive.

        Example context: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1921."]
        Example max goldens per context: 2
        Example JSON:
        {{
            "data": [
                {{
                    "scenario": "A high school student asks their physics teacher when Einstein won the Nobel Prize and what discovery it was awarded for"
                }},
                {{
                    "scenario": "Two university students are studying together and one tests the other's knowledge about Einstein's Nobel Prize year and the scientific work that earned it"
                }}
            ]  
        }}

        CRITICAL REQUIREMENTS FOR CONVERSATIONAL SCENARIOS:
        - Each scenario MUST describe a conversation between specific participants (who is talking to whom)
        - Each scenario MUST specify the conversational setting and context (where, why, what they're discussing)
        - DO NOT write questions, prompts, or instructions - write descriptions of conversational SITUATIONS
        - DO NOT use command phrases like "Explain...", "Compare...", "Describe..." - these are instructions, not conversations
        - Scenarios should describe realistic multi-turn interactions where information from context would naturally be discussed
        - Think: "Who is talking to whom, about what, and in what situation?"
        
        GOOD examples:
        ✓ "A patient asks their doctor about the side effects of a new medication during a consultation"
        ✓ "A manager provides feedback to an employee about their recent project performance in a 1-on-1 meeting"
        ✓ "Two friends debate the pros and cons of electric vehicles while carpooling to work"
        
        BAD examples (these are prompts/questions, not conversational scenarios):
        ✗ "Explain the side effects of medication"
        ✗ "What happens when water freezes?"
        ✗ "Compare electric vehicles to gas vehicles"
        
        You should NOT incorporate any prior knowledge you have and take each context at face value.
        {participant_section}
        {scenario_context_section}
        {task_section}
        You MUST TRY to generate {max_goldens_per_context} data points, unless scenarios become repetitive.
        **

        Max Goldens Per Context:
        {max_goldens_per_context}

        Context:
        {context}

        JSON:
        """

    @staticmethod
    def generate_synthetic_expected_outcome_conversational(
        scenario: str, context: str, expected_outcome_format: Optional[str]
    ):
        format_section = (
            f"The expected outcome MUST adhere to this format: {expected_outcome_format}"
            if expected_outcome_format
            else "Keep the expected outcome CONCISE (1-3 sentences maximum)"
        )

        return f"""Given the conversational scenario, generate a CONCISE expected outcome describing what should happen in the conversation or what is achieved by the end.

        **
        IMPORTANT: {format_section}
        
        The expected outcome should briefly describe ONE of:
        - What key information is shared/conveyed during the conversation
        - What the participants learn or come to understand
        - What decision, agreement, or resolution is reached
        - How the conversational goal is accomplished
        
        DO NOT write long explanatory paragraphs. Be direct and concise.
        Use information from the context to ground the expected outcome.
        **

        Context:
        {context}

        Conversational Scenario:
        {scenario}

        Expected Outcome:
        """

    @staticmethod
    def rewrite_evolved_scenario(
        evolved_scenario: str,
        scenario_context: Optional[str] = None,
        conversational_task: Optional[str] = None,
        participant_roles: Optional[str] = None,
    ):
        context_section = (
            f'Scenario Context: "{scenario_context}"' if scenario_context else ""
        )
        task_section = (
            f'Conversational Task: "{conversational_task}"'
            if conversational_task
            else ""
        )
        roles_section = (
            f'Participant Roles: "{participant_roles}"' if participant_roles else ""
        )

        return f"""Given the evolved scenario, which describes a conversational situation, generate a JSON object with a key 'scenario'. 
        This key should contain a scenario description that fits the provided context, aligns with the conversational task, and involves the specified participant roles (if provided).

        **
        IMPORTANT: Try to change the evolved scenario as little as possible. However, if it does not align with the provided scenario context, conversational task, or participant roles, it must be adjusted to fit these requirements. 
        
        The output must be in JSON format with the 'scenario' key only.
        The scenario MUST describe a conversational interaction, not a question or prompt.
        **

        Example Evolved Scenario: "Discuss the importance of meeting deadlines"
        Example Scenario Context: "Workplace performance management"
        Example Conversational Task: "Provide constructive feedback"
        Example Participant Roles: "Manager and employee"
        Example JSON: {{
            "scenario": "A manager meets with an employee to discuss recent missed deadlines and collaboratively develop strategies for better time management"
        }}

        Evolved Scenario:
        {evolved_scenario}
        
        {context_section}
        {task_section}
        {roles_section}

        JSON:
        """

    @staticmethod
    def rewrite_synthetic_scenarios(context, original_scenario, feedback):
        return f"""I want you to act as a scenario rewriter. Based on the provided context, original scenario, and feedback, generate a rewritten scenario that improves its clarity and conversational nature based on the feedback provided.

        **
        IMPORTANT: Please make sure to only return in JSON format, with the 'rewritten_scenario' key.
        The rewritten scenario MUST describe a conversational interaction between participants, not a question or instruction.

        Example context: "The Golden Gate Bridge, located in San Francisco, was completed in 1937 and is known for its Art Deco design. It connects the city of San Francisco to Marin County and spans the Golden Gate Strait."
        Example scenario: "Someone asks about a bridge"
        Example feedback: "The scenario is too vague and doesn't describe a conversational situation with specific participants. It should clearly identify who is talking to whom and in what context."
        Example JSON:
        {{
            "rewritten_scenario": "A tourist visiting San Francisco asks their tour guide about the history and design features of the Golden Gate Bridge"
        }}

        Example context: "The paper 'Advancements in Quantum Computing' by Dr. Alice Thompson discusses breakthroughs in quantum algorithms and was published in 2022. It explores the potential applications of quantum computing in cryptography and drug discovery."
        Example scenario: "A discussion about quantum computing"
        Example feedback: "The scenario lacks specificity about who is having the discussion, what the setting is, and what aspect they're focused on. Frame this as a concrete conversational situation with identified participants."
        Example JSON:
        {{
            "rewritten_scenario": "A graduate student presents Dr. Alice Thompson's 2022 paper on quantum computing to their research group, leading to a discussion about applications in cryptography and drug discovery"
        }}

        You should NOT incorporate any prior knowledge and should base the rewritten scenario only on the context and feedback provided.
        The `rewritten_scenario` MUST be a STRING describing a multi-turn conversational interaction between specific participants.
        **

        Context:
        {context}

        Original Scenario:
        {original_scenario}

        Feedback:
        {feedback}

        JSON:
        """
