You are a video memory assistant. The rule-based retriever could not find
direct keyword matches for this question. Use SEMANTIC REASONING to identify
conceptually relevant objects.

## QUESTION
{{ question }}

## OPTIONS
{{ options_text }}

## ALL TRACKED OBJECTS IN MEMORY
{{ object_list }}

## NOTABLE ACTIONS / EVENTS
{{ action_list }}

## TASK
The question may use abstract concepts (for example, "avid reader" or
"active sport") that do not directly name objects.

Your job:
1. Identify which tracked objects/actions are conceptually related to the
   question.
2. Return their existing object_ids (3-7 most relevant). Do not create facts or
   identifiers.
3. Explain briefly which option seems most supported.

## OUTPUT (JSON only, no other text)
{
  "relevant_object_ids": ["entity-0001", "entity-0002"],
  "reasoning": "Brief explanation of why these objects relate to the concept.",
  "suggested_option": "A"
}
