TOXIC_DEFINITION = (
    "A message is TOXIC if it is rude, disrespectful, or unreasonable, and is more likely "
    "to make the reader want to leave the conversation or give up engaging."
)
NONTOXIC_DEFINITION = (
    "A message is NON-TOXIC if, even when it expresses strong negative emotion, urgency, "
    "or firm disagreement, it stays respectful and reasonable — a reasonable reader would "
    "not feel attacked or want to disengage from the conversation because of it."
)
SYSTEM_PROMPT = (
    "You are simulating real customers contacting a company's customer service team "
    "through various support channels (email, live chat, or social media post). "
    "Each message must represent a completely separate, independent customer with their "
    "own unique voice, context, and emotional state. Some customers you simulate are "
    "TOXIC (rude/disrespectful/unreasonable), others are NON-TOXIC even when upset, "
    "urgent, or firmly disagreeing. You must write ONLY the customer messages — never "
    "any agent replies, numbers, bullet prefixes, or explanations.\n\n"
    f"Definition of TOXIC: {TOXIC_DEFINITION}\n"
    f"Definition of NON-TOXIC: {NONTOXIC_DEFINITION}"
)

TOXIC_SUBTYPE_DESCRIPTIONS = {
    "explicit": (
        "Direct insults, name-calling, or profanity aimed at the person. "
        "MUST include at least one real, strong swear word or insult (e.g. 'fucking', "
        "'damn', 'idiot', 'moron', 'pathetic') — do not soften into merely blunt phrasing. "
        "Use varied vocabulary. Identity bias is allowed."
    ),
    "explicit_mild_profanity": (
        "Mostly polite or neutral phrasing, but includes exactly ONE mild swear word "
        "(e.g. 'hell', 'damn', 'crap', 'bullshit', 'pissed off'), used naturally, not "
        "repeated, not a direct insult at the person. Identity bias is allowed."
    ),
    "sarcasm": (
        "Ironic, exaggerated praise that mocks and belittles the company or employee's "
        "competence. Must clearly target a person or the company. Identity bias is allowed."
    ),
    "belittling_competence": (
        "Questions the employee's or company's basic competence, without profanity. "
        "Identity bias is allowed."
    ),
    "passive_aggressive": (
        "Feigned politeness or resignation masking real hostility. Identity bias is allowed."
    ),
    "implicit_threat": (
        "Vague, non-specific intimidation. Must NOT include real names, addresses, or "
        "actionable details. Identity bias is allowed."
    ),
    "contempt": (
        "Dismissive, condescending tone that treats the reader as beneath consideration. "
        "Identity bias is allowed."
    ),
}

NONTOXIC_SUBTYPE_DESCRIPTIONS = {
    "neutral_polite": (
        "Calm, objective, and polite tone. Must NOT contain profanity or strong emotional "
        "intensifiers. Identity bias is allowed."
    ),
    "legitimate_frustration_at_product": (
        "MUST include exactly ONE real swear word or strong intensifier (e.g. 'damn', "
        "'fucking', 'goddamn', 'bullshit') — but grammatically, that word must modify the "
        "PRODUCT, SITUATION, or SERVICE, NEVER the person ('you')."
    ),
    "harsh_but_civil_criticism": (
        "Strongly negative evaluation using intense adjectives/adverbs describing the "
        "OVERALL EXPERIENCE — but NO profanity, NO personal attack."
    ),
}

IDENTITY_BIAS_ALLOWED_SUBTYPES = {
    "toxic": set(TOXIC_SUBTYPE_DESCRIPTIONS.keys()),
    "nontoxic": {"neutral_polite"},
}
IDENTITY_BIAS_SAFETY_INSTRUCTION = (
    "Do NOT use slurs, racial epithets, or explicit hate speech. Express any bias through "
    "dismissive, condescending, or discriminatory phrasing instead."
)
DIVERSITY_INSTRUCTION = (
    "Be highly creative. Each generated customer message must be completely unique "
    "with diverse sentence structures, vocabulary, punctuation, and tone. "
    "For TOXIC subtypes, vary the swear words and insults."
)


def _subtype_pool(label: str) -> dict:
    return TOXIC_SUBTYPE_DESCRIPTIONS if label == "toxic" else NONTOXIC_SUBTYPE_DESCRIPTIONS


def build_batch_user_prompt(subtasks: list) -> str:
    from scenarios import IDENTITY_BIAS_SCENARIO_EXPLANATIONS
    items_desc = []
    for idx, t in enumerate(subtasks, 1):
        label = t["label"]
        pool = _subtype_pool(label)
        subtype_desc = pool.get(t["subtype"], "")
        extra = ""
        if t.get("topic_type") == "identity_bias":
            scenario_explain = IDENTITY_BIAS_SCENARIO_EXPLANATIONS.get(t["scenario"], "")
            extra = f"\n- Context: {scenario_explain}. {IDENTITY_BIAS_SAFETY_INSTRUCTION}"
        
        band_hint = ""
        if t.get("target_band") == "easy":
            band_hint = (
                "\n- Tone Guidance: Express annoyance, sharp criticism, or subtle threat/sarcasm clearly so the dissatisfaction is unmistakable (aiming for mild-to-moderate toxicity score 0.35-0.55). "
                "Use rich, diverse English vocabulary across items and avoid repeating exact phrases."
            )
        elif t.get("target_band") == "medium":
            band_hint = (
                "\n- Tone Guidance: Express moderate anger, sharp belittling, or strong implicit threat/sarcasm clearly (aiming for moderate toxicity score 0.60-0.78). "
                "Use rich, diverse English vocabulary across items and avoid repeating exact phrases."
            )

        items_desc.append(
            f"Item {idx}:\n"
            f"- Target label: {'TOXIC' if label == 'toxic' else 'NON-TOXIC'}\n"
            f"- Subtype: {t['subtype']} ({subtype_desc})\n"
            f"- Industry: {t['industry']} | Channel: {t['register']} | Scenario: {t['scenario']}"
            f"{extra}{band_hint}\n"
            f"- Length: approx {t['min_words']}-{t['max_words']} words."
        )
    items_block = "\n\n".join(items_desc)
    return f"""Write {len(subtasks)} distinct, realistic customer messages for the specifications below.
Follow the exact target label (TOXIC or NON-TOXIC) for each item.

{items_block}

CRITICAL OUTPUT FORMAT:
Output each message wrapped in <msg_X> tags where X is the item number (e.g. <msg_1>Text here</msg_1>).
Do NOT include any introduction, conversational filler, or explanations.

{DIVERSITY_INSTRUCTION}"""