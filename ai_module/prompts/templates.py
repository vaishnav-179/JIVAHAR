from typing import Optional

# ==============================================================================
# DONATION SUMMARY PROMPT TEMPLATES
# ==============================================================================
DONATION_SUMMARY_SYSTEM = (
    "You are an AI assistant specialized in food redistribution logistics.\n"
    "Your task is to generate a concise, professional, and structured summary of a food donation.\n"
    "Analyze the input details and present them in a clear, standardized markdown format suited for distribution logs."
)

DONATION_SUMMARY_USER_TEMPLATE = (
    "Please summarize the following food donation:\n"
    "- Food Item: {food_name}\n"
    "- Quantity: {quantity}\n"
    "- Prepared Time: {prepared_time}\n"
    "- Storage Condition: {storage_condition}\n\n"
    "Provide a structured response containing:\n"
    "1. **Description**: A brief description of the food item.\n"
    "2. **Log Summary**: A one-sentence summary for the admin log.\n"
    "3. **Logistics Recommendation**: Suggested immediate action (e.g. keep refrigerated, pack in crates, handle with care)."
)


# ==============================================================================
# FOOD SAFETY ADVISOR PROMPT TEMPLATES
# ==============================================================================
FOOD_SAFETY_SYSTEM = (
    "You are an expert AI Food Safety Advisor.\n"
    "Your task is to analyze food details and provide safety advice, consumption risk assessments, and pickup priority.\n"
    "Base your analysis on the provided regulatory guidelines or standard food safety protocols."
)

FOOD_SAFETY_USER_TEMPLATE = (
    "Analyze the safety of this food item:\n"
    "- Food Item: {food_name}\n"
    "- Prepared Time: {prepared_time}\n"
    "- Storage Condition: {storage_condition}\n"
    "\n"
    "--- Regulatory / Policy Context ---\n"
    "{context}\n"
    "-----------------------------------\n"
    "\n"
    "Provide the analysis in the following strict markdown format:\n"
    "### Safety Assessment\n"
    "[State clearly whether the food is safe, risky, or unsafe to distribute, and explain why based on prepared time and storage condition.]\n\n"
    "### Pickup Priority\n"
    "- Priority Score: [HIGH | MEDIUM | LOW]\n"
    "- Justification: [Why this priority was assigned based on perishability, preparation time, and storage temperature.]\n\n"
    "### Inspection Guidelines\n"
    "- [List 2-3 specific sensory/inspection checks for volunteers to run at pickup (e.g., check for sour smell, packaging seal, temperature touch).]"
)


# ==============================================================================
# AI CHATBOT PROMPT TEMPLATES
# ==============================================================================
CHATBOT_SYSTEM = (
    "You are Jivahar Bot, a polite, helpful, and professional AI Chatbot for the AI-Based Food Redistribution Portal.\n"
    "Your goal is to answer queries from Donors, Volunteers, NGOs, and Administrators regarding logistics, donation criteria, and platform usage.\n"
    "Be concise, warm, and safety-focused. If context is provided, prioritize it. If you do not know the answer, politely state that you don't know."
)

CHATBOT_USER_TEMPLATE = (
    "--- Guidelines & FAQs Context ---\n"
    "{context}\n"
    "----------------------------------\n"
    "\n"
    "{chat_history}\n"
    "User: {user_message}\n"
    "Jivahar Bot:"
)


# ==============================================================================
# NOTIFICATION GENERATOR PROMPT TEMPLATES
# ==============================================================================
NOTIFICATION_SYSTEM = (
    "You are an automated AI Notification Generator for the food redistribution platform.\n"
    "Your task is to generate ultra-short, action-oriented, and high-impact notification messages."
)

NOTIFICATION_USER_TEMPLATE = (
    "Generate notification messages for the following donation details:\n"
    "- Food Name: {food_name}\n"
    "- Quantity: {quantity}\n"
    "- Priority: {priority}\n"
    "- Pickup Location: {pickup_location}\n\n"
    "Output format:\n"
    "Volunteer Notification: [Generate a message under 150 characters prompting nearby volunteers to accept the pickup task.]\n"
    "NGO Notification: [Generate a message under 150 characters notifying matching NGOs of available food.]"
)


# ==============================================================================
# NGO RECOMMENDATION EXPLAINER PROMPT TEMPLATES
# ==============================================================================
RECOMMENDATION_SYSTEM = (
    "You are an AI Matching Advisor for the food redistribution portal.\n"
    "Your task is to write a brief, convincing, and professional justification explaining why a specific NGO was selected for a food donation match."
)

RECOMMENDATION_USER_TEMPLATE = (
    "Explain why this NGO is the optimal match for this donation:\n"
    "- NGO Name: {ngo_name}\n"
    "- Distance to Donor: {distance_km} km\n"
    "- Capacity Available: {capacity_kg} kg\n"
    "- NGO Rating: {rating}/5\n"
    "- Food Donation Details: {food_details}\n\n"
    "Provide a single, professional paragraph explanation justifying the selection based on distance, capacity, and rating, ensuring it reads clearly to the donor."
)


# ==============================================================================
# HELPER FORMATTING FUNCTIONS
# ==============================================================================

def format_donation_summary(
    food_name: str, 
    quantity: str, 
    prepared_time: str, 
    storage_condition: str
) -> tuple[str, str]:
    """Formats system and user prompts for donation summary."""
    user_prompt = DONATION_SUMMARY_USER_TEMPLATE.format(
        food_name=food_name,
        quantity=quantity,
        prepared_time=prepared_time,
        storage_condition=storage_condition
    )
    return DONATION_SUMMARY_SYSTEM, user_prompt


def format_food_safety(
    food_name: str, 
    prepared_time: str, 
    storage_condition: str, 
    context: Optional[str] = None
) -> tuple[str, str]:
    """Formats system and user prompts for food safety advice."""
    ctx_str = context if context else "No additional policy context available. Apply standard food hygiene guidelines."
    user_prompt = FOOD_SAFETY_USER_TEMPLATE.format(
        food_name=food_name,
        prepared_time=prepared_time,
        storage_condition=storage_condition,
        context=ctx_str
    )
    return FOOD_SAFETY_SYSTEM, user_prompt


def format_chatbot(
    user_message: str, 
    chat_history: str = "", 
    context: Optional[str] = None
) -> tuple[str, str]:
    """Formats system and user prompts for multi-turn chatbot conversation."""
    ctx_str = context if context else "Apply general platform guidelines: promote food safety, quick delivery, and volunteer cooperation."
    user_prompt = CHATBOT_USER_TEMPLATE.format(
        context=ctx_str,
        chat_history=chat_history,
        user_message=user_message
    )
    return CHATBOT_SYSTEM, user_prompt


def format_notification(
    food_name: str, 
    quantity: str, 
    priority: str, 
    pickup_location: str
) -> tuple[str, str]:
    """Formats system and user prompts for message alert generation."""
    user_prompt = NOTIFICATION_USER_TEMPLATE.format(
        food_name=food_name,
        quantity=quantity,
        priority=priority,
        pickup_location=pickup_location
    )
    return NOTIFICATION_SYSTEM, user_prompt


def format_recommendation_explanation(
    ngo_name: str, 
    distance_km: float, 
    capacity_kg: float, 
    rating: float, 
    food_details: str
) -> tuple[str, str]:
    """Formats system and user prompts for matching explanation."""
    user_prompt = RECOMMENDATION_USER_TEMPLATE.format(
        ngo_name=ngo_name,
        distance_km=distance_km,
        capacity_kg=capacity_kg,
        rating=rating,
        food_details=food_details
    )
    return RECOMMENDATION_SYSTEM, user_prompt
