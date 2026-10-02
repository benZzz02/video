Analyze this segment for budgeted object-state memory.

Time range: {{ segment_range }}
Selected frame mapping: {{ frame_table }}

{{ writing_budget }}

Existing object catalog (identity guidance only):
{{ entity_catalog }}

The budget controls writing detail, not object discovery.

Required behavior:
1. DETAILED objects: if visible, write structured state with location, holder,
   state, relation, interaction, state_change, evidence_summary, and
   evidence_frames.
2. COMPACT objects: if visible, write one compact note with
   location/state/relation and evidence_frames.
3. Always discover newly visible manipulated objects, tools, containers,
   appliances, cooking surfaces, sinks, screens/text, and unusual objects.
   Put them in compact_objects unless they are central to the action, then put
   them in detailed_objects.
4. Do not ignore new or rare objects just because they are not listed in the
   budget.
5. Do not duplicate the same physical object across detailed_objects and
   compact_objects.

Return ONLY valid JSON:
{
  "time": "start-end",
  "detailed_objects": [
    {
      "name": "object name",
      "aliases": ["optional established alias"],
      "category": "category",
      "attributes": ["attr1"],
      "location": "where it is",
      "holder": "none or holder",
      "state": "current visible state",
      "relations": [
        {"relation": "on/in/next_to/used_with/etc", "target": "other object"}
      ],
      "interactions": [
        {"type": "held/moved/used/etc", "with": "person/tool", "summary": "description"}
      ],
      "state_change": "what changed since previous segment, or stable",
      "visible_text": "exact legible text or empty string",
      "evidence_summary": "specific visible evidence",
      "evidence_frames": [0],
      "confidence": 0.85
    }
  ],
  "compact_objects": [
    {
      "name": "non-focus visible object",
      "category": "category",
      "location": "compact location",
      "state": "compact state",
      "relation": "relation to action or detailed object",
      "brief": "one short useful memory note",
      "evidence_frames": [0],
      "confidence": 0.7
    }
  ],
  "events": [
    {
      "event_type": "action type",
      "summary": "what happened",
      "participants": ["obj1", "obj2"],
      "changed_objects": ["obj1"],
      "evidence_frames": [0],
      "confidence": 0.8
    }
  ]
}

Rules:
1. Only list visible or partially visible objects.
2. Maximum {{ max_objects }} total objects across both object lists and
   {{ max_events }} events.
3. Prefer detailed budget for repeated or actively manipulated focus objects.
4. Preserve compact notes for tools, containers, appliances, cooking surfaces,
   sinks, screens/text, and newly appearing objects even when they are not the
   action center.
5. evidence_frames uses local indices from Selected frame mapping; use an empty
   list if you cannot localize the evidence.
