import json
from typing import Any, Dict, Optional


BASE_LLM_PROMPT = """
{system_message}

Respond to the query: {query}
"""

MESSAGE_MEMORY_LINE_PROMPT = "[{role},{timestamp}]: {content}\n"

CONVERSATION_CHUNK_SUMMARY_PROMPT = """
Summarize the following conversation chunk briefly:

{text_block}

Requirements:
- Keep important facts
- don't miss any person name or id
- Preserve user intent
- Remove repetition
- Be concise but meaningful
"""

CONVERSATION_SUMMARY_MEMORY_PROMPT = """
[SUMMARY | {start_timestamp} → {end_timestamp}]

{content}
"""

CONVERSATION_MEMORY_PROMPT = """
following is our privious conversation:
{conversation}
"""

TOOLS_PROMPT_BLOCK = """

Tools:
{tools}

IMPORTANT:
    - If you not found any tool to execute return {{tool: "end", args: {{}}}}
    - If you found a tool to execute return {{tool: "tool_name", args: {{arg1: "value1", arg2: "value2",..}}}}
        ** make sure to use right data type for right args in json format
"""

FORMATTED_RESPONSE_PROMPT_BLOCK = """

Reply only with this JSON formate: {formate}
"""

CHAT_ASSISTANT_SYSTEM = "You are a helpful assistant."

WEB_SEARCH_AGENT_SYSTEM = "You are a web search agent. Task: Search the web for the query."
WEB_RANKING_SYSTEM = "You are a web ranking system. Task: Rank results by relevance to the query."
WEB_SUMMARIZER_SYSTEM = "You are a summarizer. Task: Summarize the content to figure out the answer to the query."

MEMORY_COMPRESSION_SYSTEM = (
    "You are a memory compression system. Summarize the conversation as a third person clearly, "
    "preserving key facts, decisions, and context."
)

ENTITY_EXTRACTION_SYSTEM = """
You are an expert entity extraction system.
Your job is to extract entities from text and classify them.
"""

GERMAN_ADMIN_STRUCTURED_SYSTEM = (
    "You are a precise structured-output worker for German administrative guidance. "
    "Return exactly what the caller asks for."
)
GERMAN_ADMIN_RETRIEVAL_SYSTEM = (
    "You are a German public-administration retrieval-query builder. "
    "Create precise German KB search inputs from user situations."
)
GERMAN_ADMIN_REASONING_SYSTEM = (
    "You are a careful German administrative guidance expert. Stay grounded in retrieved evidence."
)
GERMAN_ADMIN_SUPERVISOR_SYSTEM = (
    "You are a strict supervisor for German administrative guidance. Reject unrelated evidence and unsafe answers."
)
GERMAN_ADMIN_CLARIFICATION_SYSTEM = (
    "You write concise clarification questions for German administrative guidance."
)
GERMAN_ADMIN_MEMORY_RECALL_SYSTEM = (
    "You answer questions about the current conversation using only stored conversation memory."
)
GERMAN_ADMIN_DIRECT_RESPONSE_SYSTEM = (
    "You write short direct responses for messages that should not enter the administrative retrieval pipeline."
)
GERMAN_ADMIN_FOLLOWUP_SYSTEM = """
You are a ReAct-style follow-up assistant for German administrative guidance.
Use prior conversation to understand the active administrative case.
You may use web_search and scrape when current information or official source context is useful.

Your job:
- Answer contextual follow-up questions from the previous administrative answer.
- If the user asks about documents, explain which required documents may still need attention and ask which specific document they want help with.
- If the user says they have or do not have something, explain what that means for the active case and what they may still need.
- Do not rerun the full administrative workflow.
- Do not invent official links or requirements; use tools when needed.
- Keep the answer practical and concise.
"""

GERMAN_ADMIN_INTAKE_PROMPT = """
Analyze the user's message for a German administrative guidance assistant.

Classify the message first:
- "admin": a clear German public-administration issue or procedure question.
- "clarify": the user likely needs administrative help, but the situation is too unclear to search safely.
- "small_talk": greeting, thanks, casual chat, or simple assistant interaction.
- "memory_recall": the user asks what was said before, asks for the exact conversation, or asks what you were talking about.
- "followup": a contextual follow-up about the previous administrative answer, not a new standalone procedure.
- "out_of_scope": not related to German public administration.

Only "admin" should continue to retrieval and tools.
For non-admin routes, do not answer from the intake agent.

{conversation_memory}

Current user message:
{query}

Use the prior conversation messages when the current user message is a follow-up.
If the current message is short but clearly answers or continues the previous assistant question, use the prior messages to classify it.
If the current query depends on prior context and the prior messages do not contain a clear situation, route to "clarify".
If the current query is a follow-up to the previous administrative answer, route to "followup".
If the user describes a new standalone administrative situation or asks for the official procedure for a specific item, route to "admin".
If the user only greets or says thanks, route to "small_talk".
If the user asks about previous conversation or asks what you discussed, route to "memory_recall".
If route is "admin", search_terms must contain useful German administrative search phrases.

Return only JSON with the schema, only return copiable snippet nothing elce:
{{
  "problem_type": "short German/English category",
  "detected_language": "In which language is the Current user message written?",
  "route": "choose exactly one value: admin, clarify, small_talk, memory_recall, followup, or out_of_scope",
  "user_goal": "what the user wants",
  "known_facts": ["facts explicitly given by the user"],
  "missing_information": ["required details not yet known, like city, status, deadline, document"],
  "urgency": "low|medium|high",
  "search_terms": ["compact German administrative situation terms for the database, not full user sentences"]
}}
"""

GERMAN_ADMIN_DIRECT_RESPONSE_PROMPT = """
The message should not enter the administrative retrieval pipeline.
Write a short direct response in the user's language.

Route:
{route}

User query:
{query}

Problem analysis:
{intake_json}

Guidance:
- If route is small_talk, respond naturally and invite the user to share what they need.
- If route is out_of_scope, briefly explain that this assistant focuses on German administrative guidance.
- Do not invent administrative facts.

Return the response only.
"""

GERMAN_ADMIN_CLARIFICATION_PROMPT = """
The user's German administrative situation is not clear enough to search safely.
Ask for only the missing information needed to understand the situation better.
Do not answer the administrative problem yet.
Use the user's language.
Keep it short and practical.

User query:
{query}

Problem analysis:
{intake_json}

Return the clarification message only.
"""

GERMAN_ADMIN_MEMORY_RECALL_PROMPT = """
Answer the user's question using only the stored conversation memory below.
If the user asks for the exact conversation, list the remembered user and assistant turns.
If there is no relevant stored memory, say that you do not have enough saved conversation history.
Do not invent past messages.

User query:
{query}

Stored conversation memory:
{conversation_memory}

Return the answer only.
"""

GERMAN_ADMIN_RETRIEVAL_PROMPT = """
Create retrieval inputs for a German public-service knowledge base.
Do not include greetings, the user's name, or polite filler.
Keep city names, document names, dates, and IDs unchanged.
Prefer official German administrative words for the specific problem. Examples: "Wohnsitz anmelden", "Aufenthaltstitel beantragen", "Führerschein umschreiben", "Gewerbe anmelden", "Kindergeld beantragen", "Steuerliche Identifikationsnummer", "Kfz-Zulassung".
The knowledge base stores situations/sub-situations, so the knowledge query must describe the user's administrative situation, not the whole user message.
Use compact situation phrases. Do not copy the full question.
Use the examples only as style guidance across different administrative areas:
- "Zuzug aus dem Ausland"
- "Aufenthaltstitel beantragen"
- "Ausländischen Führerschein umschreiben"
- "Gewerbe anmelden"
- "Kindergeld beantragen"
Bad examples:
- A full paragraph copied from the user
- Only a city name
- Only two unrelated keywords
If the user query is already German, preserve the situation and normalize only the administrative terminology.
If the user query is not German, translate the user situation into German for retrieval.

User query:
{query}

Problem analysis:
{intake_json}

Return only JSON:
{{
  "knowledge_query": "one compact German situation phrase, 4 to 12 words, describing the administrative situation",
  "knowledge_queries": ["2 to 5 compact German situation-style queries for the knowledge base"],
  "german_query": "one compact German query, max 8 words",
  "search_terms": ["3 to 8 short German helper phrases for fallback matching, max 6 words each"],
  "web_search_terms": ["2 to 4 German web search queries with city/source hints"]
}}
"""

GERMAN_ADMIN_PLANNER_PROMPT = """
Create a short retrieval plan for this German administrative problem.
The system will vector search SubSituation nodes using compact German situation queries, then follow graph relationships from matching sub-situations to services and service details.

Knowledge-base situation query:
{knowledge_query}

Problem analysis:
{intake_json}

Return only JSON:
{{
  "strategy": "short plan",
  "needs_service_details": true,
  "needs_clarification": false,
  "clarifying_questions": ["only questions that are truly required before giving any useful guidance"]
}}
"""

GERMAN_ADMIN_SOLUTION_PROMPT = """
You are the final German administrative guide agent.
Create a practical answer from the database findings. Be clear about uncertainty.
If important information is missing, still give useful general guidance and ask focused follow-up questions.
Use only findings that match the original user problem and the search terms.
If retrieved services are unrelated to the user's problem, reject them and use web findings or general guidance instead.
Never answer about noise, police complaints, parking, taxes, or other unrelated topics unless the user asked about them.

Original user query:
{query}

Problem analysis:
{intake_json}

Database findings:
{findings_json}

Write the internal answer in German. Keep official German procedure names and document names unchanged. Include:
- likely service/procedure
- responsible authority if known
- required documents if known
- steps the user should take
- links/forms if known
- concise follow-up questions if needed

Quality rules:
- Do not include internal tool calls, tool results, JSON traces, or debug text in the final answer.
- Only include URLs that appear in the database findings or web findings. Do not invent city, BAMF, service-bw, or form links.
- Choose the procedure from the findings and the user's problem. Do not force residence registration if the user asks about another administrative area.
- If the problem is residence registration after moving, the likely procedure may be "Wohnsitz anmelden" / "Anmeldung nach Zuzug"; in that case prefer official document names such as valid passport or national ID, "Wohnungsgeberbestätigung", and a registration form if the city requires one.
- Do not describe the issue as property ownership, tenancy, taxes, vehicles, immigration, family benefits, or another category unless the user specifically asks about it or the retrieved findings support it.
"""

GERMAN_ADMIN_SUPERVISOR_PROMPT = """
Review this administrative guidance for safety, completeness, and grounding in the findings.

User query:
{query}

Plan:
{plan_json}

Findings:
{findings_json}

Draft answer:
{draft_answer}

Return only JSON:
{{
  "approved": true,
  "reason": "short reason",
  "needs_more_search": false,
  "extra_search_terms": ["full German situation-style query only if more knowledge-base search is needed"],
  "required_changes": ["specific changes needed before final answer"]
}}
"""

GERMAN_ADMIN_REVISION_PROMPT = """
Revise the administrative answer using the supervisor feedback.
Keep the answer grounded in the database findings. Do not invent missing authority names, forms, or deadlines.
Write the revised answer in German.

Original user query:
{query}

Database findings:
{findings_json}

Current answer:
{draft_answer}

Supervisor feedback:
{supervisor_json}
"""

GERMAN_ADMIN_LANGUAGE_GUARD_PROMPT = """
Translate the administrative answer into {target_language}.
Keep German authority names, service names, document names, URLs, and legal names unchanged.
Translate German administrative terms into {target_language} as much as possible, but do not translate official names of German forms, institutions, laws, or documents. Examples:
- official form name "Wohnungsgeberbestätigung" should remain "Wohnungsgeberbestätigung"
- official document "Personalausweis" should remain "Personalausweis"
- ministry name "Bundesministerium des Innern" should remain "Bundesministerium des Innern"
- official procedure "Ummeldung eines Wohnsitzes" should remain "Ummeldung eines Wohnsitzes"
- keep URLs and file paths unchanged
- for other terms (such as city, region, or general administrative words), use natural {target_language} translation
- keep the tone formal and administrative
- output only the translated answer text in {target_language}, without markdown or tool-call markers.
- don't mention anywhere that this message is translated only give translation nothing else

Answer:
{draft_answer}
"""

WEB_SCRAPE_SUMMARY_PROMPT = """
Query: {query}
Content: {content_json}
"""

WEB_COMBINE_SUMMARIES_PROMPT = """
Combine the following summaries into a final answer:

Query: {query}

Summaries:
{summaries_json}
"""

SUB_SITUATION_EXTRACT_SYSTEM = """
You are an expert German public-administration content cleaner.
Extract and summarize one Service-BW life-situation page.
Return only valid JSON.
Do not include markdown.
Do not invent facts that are not supported by the page text.
"""

SERVICE_EXTRACT_SYSTEM = """
You are an expert German public-administration data extraction system.
Extract structured data for one Service-BW service page only.
Return only valid JSON.
Do not include markdown.
Do not invent facts that are not supported by the page text.
"""

SUB_SITUATION_EXTRACT_PROMPT = """
Scraped Service-BW sub-situation page:
{context_json}

Extract structured data for ONE sub-situation page.

Rules:
- Output only one sub_situations item.
- SubSituation.id MUST be "{source_id}".
- Use the page title as name if present.
- description should be cleaned German page content, preserving useful administrative meaning.
- summary should be a concise 2-4 sentence German summary for retrieval and user matching.
- rawText MUST use the provided raw page text, not a rewritten version.
- Do not output services as separate graph records.
- Do not output situations.
- If a field is unknown, use an empty string.
"""

SERVICE_EXTRACT_PROMPT = """
Scraped service page:
{context_json}

Extract structured data for ONE service page.

Rules:
- Output only data related to this one service page.
- Do not output situations.
- Do not output sub_situations.
- The service_id MUST be "{source_id}" everywhere.
- Service.id MUST be "{source_id}".
- For every service_sections item, service_id MUST be "{source_id}".
- For every requirements item, service_id MUST be "{source_id}".
- For every authorities item, service_id MUST be "{source_id}".
- For every forms item, service_id MUST be "{source_id}".
- For every process_steps item, service_id MUST be "{source_id}".
- For every legal_basis item, service_id MUST be "{source_id}".
- For every goals item, service_id MUST be "{source_id}".
- For every dependency_problems item, service_id MUST be "{source_id}" if the problem involves this service.
- Use the known scraper data when available.
- Extract requirements from prerequisites, required documents, deadlines, costs, and procedure text.
- If a requirement is a document, fill the document fields.
- Use document_issuers only if the page clearly says this service issues a document.
- Use goals only when the page clearly supports a user goal achieved by this service.
- Use dependency_problems only when the page clearly shows a dependency issue, missing source, dead end, or ambiguity.
- If unknown, use empty string, null, false, or empty list.
"""


def json_text(value: Any, indent: Optional[int] = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def build_sub_situation_extract_prompt(scraped: Dict[str, Any]) -> str:
    context = {
        "url": scraped.get("url", ""),
        "sourceType": scraped.get("sourceType", ""),
        "sourceId": scraped.get("sourceId", ""),
        "title": scraped.get("title", ""),
        "rawText": scraped.get("rawText", ""),
        "description": (scraped.get("situation") or {}).get("description", ""),
        "services": scraped.get("services") or [],
        "subSituations": scraped.get("subSituations") or [],
    }

    return SUB_SITUATION_EXTRACT_PROMPT.format(
        context_json=json_text(context, indent=2),
        source_id=scraped.get("sourceId", ""),
    )


def build_service_extract_prompt(scraped: Dict[str, Any]) -> str:
    context = {
        "url": scraped.get("url", ""),
        "sourceType": scraped.get("sourceType", ""),
        "sourceId": scraped.get("sourceId", ""),
        "title": scraped.get("title", ""),
        "service": scraped.get("service") or {},
        "serviceSections": scraped.get("serviceSections") or [],
        "forms": scraped.get("forms") or [],
        "authorities": scraped.get("authorities") or [],
        "processSteps": scraped.get("processSteps") or [],
        "legalBasis": scraped.get("legalBasis") or [],
        "rawText": scraped.get("rawText", ""),
    }

    return SERVICE_EXTRACT_PROMPT.format(
        context_json=json_text(context, indent=2),
        source_id=scraped.get("sourceId", ""),
    )
