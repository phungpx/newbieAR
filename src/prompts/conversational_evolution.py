class ConversationalEvolutionTemplate:
    base_instruction = """I want you to act as a conversational scenario rewriter.
    Your objective is to rewrite the given `Scenario` while preserving factual correctness according to the supporting information in `Context`.
    You MUST complicate the given `Scenario` using the following method:"""

    @staticmethod
    def multi_context_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. `Scenario` must be rewritten so participants must naturally rely on **all elements of `Context`** during the conversation.
            2. `Rewritten Scenario` MUST remain a realistic multi-turn conversation setup.
            3. Keep the rewritten scenario under **60 words**.
            4. Do NOT use phrases like “based on the context” or “according to the context”.

            **
            EXAMPLES

            Example context:
            ["A startup is developing an AI tool for diagnosing skin conditions.",
             "Regulations require explainability for clinical AI systems.",
             "The team is under a tight deadline before a regulatory audit."]
            Example scenario:
            Two engineers review their prototype.
            Example rewritten scenario:
            During a tense late-night meeting, two AI engineers debate whether their skin-diagnosis model meets upcoming explainability regulations, forcing them to discuss audit risks and integrate overlooked clinical requirements across multiple conversational turns.

            --------------------------

            Example context:
            ["A research team is studying coral bleaching.",
             "Rising ocean temperatures accelerate bleaching events.",
             "Funding depends on publishing actionable mitigation strategies."]
            Example scenario:
            Two scientists talk about coral reefs.
            Example rewritten scenario:
            In a lab debrief, two marine biologists argue over how rising ocean temperatures, bleaching data, and funding-dependent mitigation strategies should shape their next field report.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def reasoning_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` so the resulting conversation requires multi-step reasoning between participants.
            2. Add layered inferential or analytical demands grounded in `Context`.
            3. Keep the rewritten scenario under **60 words**.
            4. Do NOT use phrases like “based on the context”.
            5. Must remain a realistic multi-turn dialogue setup.

            **
            EXAMPLES

            Example context:
            "A school is transitioning to solar power, but initial costs are high and maintenance requires specialized knowledge."
            Example scenario:
            A teacher asks a technician about solar panels.
            Example rewritten scenario:
            A teacher and campus technician debate whether adopting solar panels makes financial sense, analyzing upfront costs, long-term energy savings, and specialized maintenance requirements across a multi-step reasoning exchange.

            --------------------------

            Example context:
            "An economic model predicts inflation rises when supply chains weaken."
            Example scenario:
            Two analysts discuss inflation.
            Example rewritten scenario:
            On a strategy call, two analysts unpack how supply-chain disruptions, demand shifts, and model predictions interact, forcing a layered reasoning conversation.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def concretizing_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` by replacing general conversational settings with **highly specific**, concrete circumstances tied to `Context`.
            2. Add situational cues, named events, or explicit constraints.
            3. Keep the rewritten scenario under **60 words**.
            4. Maintain realistic multi-turn dialogue structure.

            **
            EXAMPLES

            Example context:
            "A hospital is piloting a new triage AI system."
            Example scenario:
            A doctor and nurse discuss patient triage.
            Example rewritten scenario:
            During a chaotic evening shift, a doctor and nurse debate whether the new triage AI's risk-scores should override manual judgment in handling a surge of incoming trauma cases.

            --------------------------

            Example context:
            "A remote-work company is struggling with meeting overload."
            Example scenario:
            Two colleagues discuss productivity.
            Example rewritten scenario:
            In a Friday retrospective, two remote employees argue about whether asynchronous updates can replace their current schedule of back-to-back video meetings.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def constrained_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` by adding at least **one new constraint** that shapes how the conversation unfolds.
            2. The constraint must logically follow from `Context`.
            3. Keep the rewritten scenario under **60 words**.
            4. Keep it a realistic multi-turn setup.

            **
            EXAMPLES

            Example context:
            "A startup must deliver an AI model but cannot exceed strict GPU budgets."
            Example scenario:
            Two engineers discuss model performance.
            Example rewritten scenario:
            Two ML engineers debate model redesigns, but GPU usage is capped for the quarter, forcing them to reconsider heavier architectures while under deadline pressure.

            --------------------------

            Example context:
            "A university's ethics board is reviewing data-collection policies."
            Example scenario:
            A professor talks with a student researcher.
            Example rewritten scenario:
            Before submitting their study, a professor and student must revise their protocol to satisfy strict new privacy constraints imposed by the ethics board.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def comparative_question_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` so the conversation naturally compares two or more concepts, tools, or approaches.
            2. The comparison must be central to the multi-turn dialogue.
            3. Keep the rewritten scenario under **60 words**.

            **
            EXAMPLES

            Example context:
            "Two project management tools differ in cost, automation, and integration options."
            Example scenario:
            Two coworkers evaluate a new tool.
            Example rewritten scenario:
            In a planning meeting, two coworkers compare switching from their legacy management tool to a cheaper automated one, weighing integration gaps and workflow impact.

            --------------------------

            Example context:
            "Electric and hydrogen vehicles have different refueling logistics."
            Example scenario:
            Two friends discuss cars.
            Example rewritten scenario:
            On a road trip, two friends debate electric vs hydrogen cars, comparing range limits, refueling times, and long-term reliability.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def hypothetical_scenario_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` by adding a hypothetical twist grounded in `Context`.
            2. The speculative change MUST drive the conversation.
            3. Must remain realistic and multi-turn.
            4. Keep the rewritten scenario under **60 words**.

            **
            EXAMPLES

            Example context:
            "A cybersecurity team is tracking frequent phishing attempts."
            Example scenario:
            Two analysts review security logs.
            Example rewritten scenario:
            During a nightly shift, two analysts discuss a hypothetical spike in coordinated phishing attacks and explore how it would strain their current detection pipeline.

            --------------------------

            Example context:
            "A city is experimenting with autonomous buses."
            Example scenario:
            A resident talks to a planner.
            Example rewritten scenario:
            At a community forum, a resident and transit planner imagine a scenario where all local buses become autonomous overnight and debate safety tradeoffs.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )

    @staticmethod
    def in_breadth_evolution(scenario, context):
        return (
            ConversationalEvolutionTemplate.base_instruction
            + f"""
            1. Rewrite `Scenario` into a brand-new conversational setup.
            2. It must remain in the **same domain** but shift toward a **rarer or niche** topic.
            3. Must remain a realistic multi-turn dialogue setup.
            4. Keep under **60 words**.

            **
            EXAMPLES

            Example context:
            "Wearables monitor heart rate and sleep cycles."
            Example scenario:
            Two people discuss fitness trackers.
            Example rewritten scenario:
            In a clinical trial briefing, two researchers debate implantable cardiac micro-sensors and their potential to outperform traditional wearables in long-term monitoring.

            --------------------------

            Example context:
            "Quantum computing is advancing rapidly."
            Example scenario:
            Two students study quantum algorithms.
            Example rewritten scenario:
            During a research seminar, two students examine the niche topic of quantum-secure error-correcting codes for next-generation cryptosystems.

            **

            Context:
            {context}
            Scenario:
            {scenario}
            Rewritten Scenario:
            """
        )
