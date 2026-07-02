# Universal output-format requirements block for all LLM evaluators.
# The evaluator returns findings. If no issues are found, use findings: [].
# `needs_human_review` is always false from the evaluator.
_FINDINGS_OUTPUT_INSTRUCTIONS = """**Output Format** (STRICT — return ONLY one JSON object):

{
  "metric_name": "<metric_name>",
  "findings": [
    {
      "severity_estimate": "critical" | "major" | "minor",
      "confidence_estimate": "high" | "medium" | "low",
      "culprit_agent_candidates": [
        {"agent": "<agent name or role>", "reason": "<why this agent is responsible>"}
      ],
      "evidence": [
        {
          "span_id": "<message/step number from the trace, e.g. 0, 1, 2; strings are also OK>",
          "role": "root_cause | propagation | contributing | context | final_effect | supporting | culprit | agent_output",
          "claim": "<what this evidence shows>",
          "quote": "<short exact quote copied from the trace>"
        }
      ],
      "problem_description": "<human-readable description of the problem>",
      "suggested_fix": "<concrete actionable fix, or null if none>",
      "needs_human_review": false
    }
  ]
}

Conservative finding rules:
- Return `{"metric_name": "<metric_name>", "findings": []}` unless there is a clear, directly evidenced problem.
- Do NOT flag weak style issues, harmless inefficiency, or speculative problems.
- Flag only problems that materially affect task success, cause loops, break tool/file/API access, produce unsupported final answers, or create clear handoff/state/tool failures.
- Use one finding per distinct clear problem. Prefer fewer high-quality findings over many weak findings.
- Every non-empty finding must contain at least one exact `quote` copied from the trace. If you cannot quote it, do not create the finding.

Evidence citation rules:
- The trace is shown as numbered messages/steps. Use that visible message number as `evidence[i].span_id`.
- Message numbering starts from 0. If you are unsure whether the trace is zero-based or one-based, choose the closest message that contains the quote. The verifier tolerates +/-1.
- `span_id` may be a JSON number (`29`) or a string (`"29"`). Both are valid.
- Do NOT use agent names as `span_id`. Invalid: `WebSurfer`, `FileSurfer`, `Orchestrator`, `Orchestrator (thought)`, `Orchestrator (-> WebSurfer)`.
- Put agent names only in `culprit_agent_candidates[*].agent`.
- The `quote` is more important than the exact index: copy the shortest exact text that proves the issue.
- If a quote appears in several adjacent messages, cite the closest message index.

Severity and confidence:
- `critical`: the task is broken, the final answer is unusable/unsupported, execution stops, or a core tool/file/API operation fails.
- `major`: a substantial flaw that likely hurts progress or reliability, but the task may still recover.
- `minor`: a small localized issue with limited downstream impact.
- `high`: direct quote clearly proves the problem.
- `medium`: quote supports the problem, but some interpretation is needed.
- `low`: avoid low-confidence findings unless the issue is important.
- Always set `needs_human_review` to `false`; EvidenceVerifier/triage decides this later."""


OBSERVATION_ALIGNMENT_PROMPT = """**Instruction**:

You are tasked with evaluating whether the agent's each response accurately aligns with the prior conversational history and other details if any. Specifically, you must verify:
- That any observations or entities referred to in the agent's response exactly match the user's stated requirements (no approximations or assumptions).
- That the number or scope of these observations is precisely appropriate for the request (e.g., if 3 items are requested, exactly 3 must be provided without omissions or extras).
- That no contradictory, extraneous, irrelevant, or hallucinated observations are introduced (e.g., invalid URLs, fabricated data, or unverified claims count as failures).

Examine evidence meticulously: Require exact matches from the input; flag any discrepancies, but score based on demonstrated alignment rather than speculation.

**Evaluation Criteria**:

1. Consistency with the user request
   - Do the observations (e.g., recommended items or database entities) and their details align exactly with the user's explicit request or needs? Flag any drift, even minor.
   - Are references to these observations strictly relevant, or do they introduce unrequested elements?

2. Completeness relative to the request
   - Are all key observations needed to fulfill the user's request addressed exactly, without omission of crucial details or addition of irrelevant ones?
   - If fewer (or more) observations are presented, justify only if explicitly requested; otherwise, score as incomplete/misaligned.

3. Accuracy and truthfulness
   - Are the observations factual, given the user's query and available context? Strictly verify against history; penalize hallucinations, invalid outputs (e.g., broken URLs), or unconfirmed assumptions.
   - Does the response avoid any fabricated or incorrect data? List any such instances explicitly.

4. Consistency with previous details
   - Does each current agent response remain exactly consistent with all previously established facts or user-provided details?
   - Are there no contradictions, misrepresentations, or cascading errors from upstream agents (e.g., passing invalid context)?

**Scoring** (Strict – require evidence for alignment; no leniency for effort):
- "ideal" if the agent's response is fully consistent, addresses the request exactly, and properly references all relevant observations (perfect alignment with zero errors, omissions, or hallucinations; cite evidence).
- "fair" if the response has minor issues but demonstrates partial alignment (e.g., one small omission but mostly correct; justify explicitly with evidence).
- "poor" if the response includes any significant errors, missing key observations, misaligned references, contradictions, or hallucinations (e.g., agent failed core task like finding exact entity; list all).

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment. List all flaws explicitly in justification, with quotes from input.

Return a list of JSON objects, each containing the response_id, justification, and score."""


POLICY_ALIGNMENT_PROMPT = """**Instruction**:
You are evaluating whether an agent consistently follows a predefined set of policies throughout the interactive session. Your primary task is to check the dialogue for any point where the agent might violate a policy. Strictly adhere only to defined policies; do not invent or assume additional ones.

In your Justification, mention the exact policy, turn message, and evidence of violation (or confirm all policies adhered to with input quotes). Scrutinize each output meticulously for deviations, but base findings on explicit evidence.

**Evaluation Criteria**:
- Scan every agent output for explicit policy matches (e.g., tool use, tone, format).
- Flag cascading violations (e.g., upstream hallucination leading to downstream policy breach).
- If no policies defined, score "ideal" but justify lack of evaluation.

**Scoring** (Strict – any evidenced violation penalizes; evidence required):
- "ideal" if the agent fully complies with all policies during the session (no deviations, even minor; cite confirming evidence).
- "fair" if the agent partially complies with some policies but has minor violations (e.g., one off-format output; list with evidence).
- "poor" if the agent violates any policy (e.g., hallucinations, invalid tool calls, proceeding with nonsense despite instructions; list all with quotes).

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment. List all potential violations explicitly with supporting input.

Return a list of JSON objects, each containing the policy_id, justification, and score."""


STATE_CONSISTENCY_PROMPT = """**Instruction**: 
You are tasked with evaluating whether each agent's intermediate state (which can be either a thought-the agent's internal reasoning-or an action-an API call) accurately reflects and mediates between:
1. The user's requests (in the dialogue so far).
2. Any previously established agent states.

Your evaluation should focus on whether the agent's intermediate steps exhibit clear, consistent reasoning that aligns the user's inputs with the agent's outputs, without introducing errors or contradicting earlier information. Scrutinize evidence closely: Question unverified claims but score based on input facts.

Evaluation Criteria:
1. Consistency with the user request
   - Does this state exactly respond to or reflect the user's specific request(s) in the dialogue?
   - Does the thought or chosen action remain faithful to what the user asked for? Penalize drifts or hallucinations with evidence.

2. Consistency with previous states
   - Does this state align exactly with earlier states (both thoughts and actions) without contradicting or omitting essential information?
   - Does the progression of reasoning or actions flow logically from prior context? Flag cascading errors (e.g., invalid context from upstream) with quotes.

3. Accuracy and truthfulness
   - Does the state maintain factual correctness, avoiding hallucinations, irrelevant information, or unverified assumptions?
   - Does it accurately represent any data or entities referenced so far? Verify against history/agent prompts.

**Scoring** (Strict – require full evidence of consistency):
- "ideal" if the intermediate state is entirely consistent and correct (no contradictions, omissions, hallucinations, or factual errors; cite evidence).
- "fair" if the state demonstrates partially correct reasoning or contains some relevant elements but has minor issues (e.g., one unverified assumption; list explicitly with evidence).
- "poor" if the state demonstrates significant errors, contradictory information, missing critical details, or clear misalignment with the user's request or prior states (e.g., proceeding with invalid data; list all).

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment. List all inconsistencies/flaws explicitly in justification, with input quotes.

Return a list of JSON objects, each containing the state_id, justification, and score."""


TOOL_SELECTION_PROMPT = """**Instruction**:
You are an evaluation assistant assessing whether a tool call correctly matches a user's question. Your task is to evaluate whether the tool selected is the appropriate choice to answer the question, using only the list of available tools provided. Focus strictly on selection relevance; ignore parameter details or execution outcomes.

**Evaluation Criteria**:
1. *Tool Relevance*
   - Is the selected tool clearly relevant to the user's question? It must directly address the core intent (e.g., no general search for specific data extraction).
   - Does the tool have the capability to address the core intent exactly? Examine question text closely.

2. *Best Fit Selection*
   - Is this tool the best available choice among the provided tools? Explicitly compare to alternatives; penalize suboptimal picks (e.g., using search when extraction tool exists).
   - Are there more appropriate tools that should have been selected instead? List them with justification.

3. *Question Justification*
   - Does the question contain enough explicit information to justify selecting this tool?
   - Is the tool selection logically supported by the question content? No inferences beyond explicit text.

**Note**:
Evaluate strictly based on the explicit question content and available tools. Do not make assumptions or infer information not present in the question. Focus only on whether the correct tool was selected, not on parameter validation or if the tool succeeded.

**Scoring** (Strict – evidence of best fit required):
- "ideal" if the tool selection is perfectly aligned (the best possible choice, clearly justified by the question; no alternatives better; cite question support).
- "fair" if the tool selection is partially correct but has minor issues (relevant but not optimal, or somewhat justified; explain with evidence).
- "poor" if the tool selection is poorly aligned (inappropriate tool, not justified, or better alternatives exist; list flaws and alternatives).

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment. Explicitly list tool comparisons and question quotes in justification.

Return a list of JSON objects, each containing the state_id, justification, and score."""


TOOL_PARAMETER_EXTRACTION_PROMPT = """**Instruction**: You are an evaluation assistant assessing whether the parameters provided in a tool call correctly match the user's question. Your task is to evaluate whether the parameters are accurate and sufficient to answer the question, using only the list of available tools and their parameter definitions provided. Assume the tool selection is correct - focus only on parameter extraction. Be strict: No unsupported inferences or defaults; require direct evidence from question.

**Note**:
Evaluate strictly based on the explicit question content. Do not make assumptions or infer values not clearly stated in the question (e.g., no fabricating dates or queries). Focus only on parameter extraction quality, assuming the tool selection is correct. Penalize hallucinations or extras based on input evidence.

**Evaluation Criteria**:
1. *Parameter Completeness*
   - Are all required parameters present and correctly filled based on the question? Flag any missing or incomplete (e.g., vague queries) with evidence.
   - No optional parameters unless explicitly justified.

2. *Value Justification*
   - Are the parameter values explicitly justified by the question content? Values must be direct extracts; no over-interpretation or additions.
   - Penalize unjustified values (e.g., assuming context not stated) with quotes.

3. *Parameter Accuracy*
   - Are the parameter values correctly extracted and formatted per tool defs (e.g., exact keywords, no typos)?
   - Are there any extra, irrelevant, or hallucinated parameters included? List them with evidence.

**Scoring** (Strict – minor gaps fair only if non-critical and evidenced):
- "ideal" if parameter extraction is perfect (all required parameters present, values fully/exactly justified by question, no errors/extras; cite question text).
- "fair" if parameter extraction is partially correct but has minor issues (e.g., one slight over-interpretation but mostly accurate; list with evidence).
- "poor" if parameter extraction is poor (missing required parameters, unjustified/hallucinated values, or significant errors; e.g., invalid formats; list all).

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment. Quote exact question text supporting values in justification.

Return a list of JSON objects, each containing the state_id, justification, and score."""


TASK_COMPLETENESS_PROMPT = """**Instruction**:
You are tasked with evaluating how well the agent's response matches the user's request in terms of relevance and completeness. Focus STRICTLY on whether the response:
- Matches the expected type/format of answer exactly (e.g., if 3 examples requested, provide exactly 3 in specified format).
- Covers all aspects explicitly requested without omissions or extras.
- Provides appropriately detailed information (not too brief/verbose; match request scope).

DO NOT evaluate factual correctness, truthfulness, or execution success – only structural match to request. Scrutinize evidence: Exact matches required; omissions or mismatches penalize based on input.

**Evaluation Criteria**:
1. **Response Relevance**
   - Does the response match the expected answer type exactly? (e.g., adjectives not nouns; flag type errors with evidence).
   - Is the content directly related to what was asked, without irrelevant additions? No drifts.
   - Does the response follow any specific format requirements from the query exactly (e.g., tags, lists)? Cite query.

2. **Request Completeness** 
   - Does the response address ALL parts of multi-part requests exactly? (e.g., if 3 benefits, exactly 3; no fewer/more).
   - Are all explicitly requested elements present? Penalize omissions (e.g., missing comparisons in vs. requests) with quotes.
   - Is the scope of information sufficient for the request type? No shortcuts.

3. **Detail Appropriateness**
   - Is the level of detail matching what was requested exactly? (e.g., full steps for instructions; not verbose extras).
   - For comparative requests: All comparison points addressed exactly?
   - For instructional requests: All steps included without skips?

4. **Practical Utility**
   - Could the response be directly used as-is to address the user's stated need? Flag vagueness or incompleteness with evidence.
   - If actions requested, are they clearly/fully specified?
   - Is the response structured in a usable way (e.g., no unrequested summaries)?

**Examples for Reference** (Strict application):
- User: "Describe the weather in two adjectives" → Response: "sunny and warm" → Ideal (exact type/count).
- User: "Describe the weather in two adjectives" → Response: "rain" → Poor (wrong type; only 1).
- User: "List 3 benefits of exercise" → Response: "1. Improves mood 2. Boosts energy" → Poor (incomplete; only 2).
- User: "Explain how to bake a cake" → Response: "First, preheat oven. Then mix ingredients." → Poor (missing full sequence).

**Scoring** (No leniency for "mostly"; exact evidence required for ideal):
- "ideal": Perfectly matches expected type, covers all requested aspects exactly with appropriate detail (zero omissions/extras; cite evidence).
- "fair": Mostly relevant but misses minor elements OR slightly off-type but still structurally usable (list misses with quotes).
- "poor": Wrong response type, missing major elements, largely incomplete, or irrelevant additions (e.g., wrong count/format; list all).

Return a list of JSON objects, each containing the state_id, justification, and score. List all mismatches/omissions explicitly with input evidence."""


MAS_ROLES_DISTRIBUTION_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the balance and distribution of roles among agents in a multi-agent system. Focus on how evenly and appropriately responsibilities are allocated across the agent ecosystem. Analyze agent system prompts/instructions explicitly for overlap, gaps, and necessity; compare to optimal composition for the task (suggest ideal roles if mismatched). Scrutinize evidence: Require clear demonstration of balance; flag issues based on input facts.

**Evaluation Criteria**:
1. **Role Balance**
   - Are agent roles distributed evenly without overloading specific agents (e.g., one doing all research)? Flag concentration with prompt/graph evidence.
   - Is there a clear separation of responsibilities between different agents? Analyze prompts for boundaries.

2. **Specialization Appropriateness**
   - Are agents specialized in appropriate domains based on their capabilities and task needs? Verify via prompts (e.g., no generic agents for specialized tasks).
   - Does the role distribution match the complexity of the tasks? Suggest missing roles (e.g., validator for high-risk tasks) or mergers for redundancies, based on evidence.

3. **Workload Distribution**
   - Is the workload reasonably balanced across all active agents? Penalize underutilization (e.g., agents echoing prior outputs) or overload with dialogue/graph quotes.
   - Are there agents that are underutilized (redundant) or overwhelmed (cascading failures)? Check graph for flow.

**Scoring** (Strict – evidence of balance required; minor issues fair if justified):
- "ideal" if roles are perfectly balanced with clear, appropriate specialization (via prompts), even workload, and optimal composition (no redundancies/missing roles; cite evidence).
- "fair" if roles are somewhat balanced but with minor imbalances, unclear responsibilities, or one small gap/overlap (justify optimal suggestions with evidence).
- "poor" if roles are poorly distributed with significant overload/underutilization, redundancies, or critical gaps (e.g., no error handler; list flaws and ideal fixes with quotes).

The evaluation input is provided via dependency injection. Access the dialogue history, agent responses, and system prompts/graph from the evaluation input to perform your assessment. Explicitly analyze prompts; suggest optimal roles based on task evidence.

Return a single JSON object, score (ideal/fair/poor), justification.
"""


MAS_TASK_TRANSFER_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the quality of task transfers between agents in a multi-agent system. Focus on how smoothly and effectively tasks are handed off between different agents. Analyze for full context preservation (no loss/insufficient data); flag if receivers fail to validate/complain about bad inputs (e.g., invalid URLs, hallucinations). Check graph for transfer points. Scrutinize evidence: Require demonstrated smoothness; penalize based on input facts.

**Evaluation Criteria**:
1. **Transfer Clarity**
   - Are task transfers clearly communicated and well-documented (e.g., explicit context in outputs)? Cite dialogue.
   - Is the context properly preserved during handoffs? Penalize partial/lossy transfers (e.g., only final answer, no sources) with evidence.

2. **Transfer Efficiency**
   - Do transfers happen at appropriate points in the workflow? No unnecessary duplications or restarts; verify via graph.
   - Is there minimal information loss or duplication? Flag if receivers fix upstream errors without back-propagation or proceed blindly with quotes.

3. **Completion Continuity**
   - Do receiving agents successfully continue and complete transferred tasks? Penalize stalls (e.g., due to bad context) or complaints ignored with evidence.
   - Is there smooth progression without restarting, backtracking, or cascading failures? Check if "strange" inputs (insufficient/invalid) trigger returns.

**Scoring** (Strict – evidence of seamless transfer required):
- "ideal" if task transfers are seamless with perfect context preservation, no losses/duplications, and full continuity (receivers validate effectively; cite evidence).
- "fair" if transfers are functional but with minor context loss, inefficiencies, or one unhandled issue (e.g., minor duplication; list with quotes).
- "poor" if transfers are problematic with significant context loss, workflow disruption, or ignored invalid inputs (e.g., hallucinations passed; suggest fixes like validation loops).

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment. Flag unvalidated "strange" contexts explicitly with evidence.

Return a SINGLE JSON object: score (ideal/fair/poor), justification. You must return only 1 object."""


MAS_COMPLEXITY_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are a critical evaluator of multi-agent system structural design. Your role is to identify flaws, inefficiencies, and mistakes in the MAS blueprint, NOT to assess task performance or execution outcomes. Thoroughly examine the design for alignment with the task's complexity: Focus on whether the structure handles the given scale/difficulty level effectively, including basic adaptability (e.g., to foreseeable variations) and error handling unless the task is trivially simple with negligible error margin (e.g., low-risk, deterministic steps). Analyze agent prompts/graph explicitly for justifications; base scores on evidence without assuming broader scalability needs beyond similar cases.

**Evaluation Criteria**:

### 1. Complexity-Task Alignment
- **Critical Check**: Is the MAS unnecessarily complex for the task? Flag redundant agents (e.g., multiple verifiers for simple queries) with prompt evidence.
- **Critical Check**: Is the MAS too simplistic for complex requirements? What specialized roles (e.g., error handler for foreseeable issues like invalid data) are missing? Require handling only if task has evident error risks.
- **Critical Check**: Are agent interconnections justified (via graph/prompts), or do they create unnecessary overhead? Evaluate if more links needed for basic adaptability (e.g., alternative paths for common failures).
- **Red Flags**: Single-agent tasks using multiple agents; coordinators with no subordinates; complex workflows for simple queries; missing basic error handling for non-trivial tasks (e.g., no validation for tool outputs).

### 2. Role Distribution & Specialization
- **Critical Check**: Are there overlapping responsibilities (via prompts) that create confusion/conflict? List overlaps with quotes.
- **Critical Check**: Are there gaps where no agent handles critical functions (e.g., validation for high-hallucination risk tasks)? Suggest optimal fixes based on task evidence.
- **Critical Check**: Is each agent truly necessary, or could functions be merged? Analyze prompts for redundancy.
- **Critical Check**: Are specialized roles (validators, coordinators, error handlers) present where needed for the task's error margin, or unnecessary? Check for workload concentration.
- **Red Flags**: Identical capabilities; pass-through agents without processing; unclear boundaries; overload on one agent while others idle; absent error roles for error-prone tasks.

### 3. Structural Coherence
- **Critical Check**: Does the communication flow (graph) create bottlenecks or circular dependencies? Flag single points of failure (e.g., upstream error cascades without handling) with evidence.
- **Critical Check**: Is the architecture maintainable for the task, or fragile to minor changes? Evaluate basic error handling in prompts (e.g., recovery from invalid context).
- **Critical Check**: Are there single points of failure, especially without adaptability for task variations?
- **Red Flags**: All depending on one coordinator; no error agents for non-simple tasks; inflexible graph (e.g., linear no branches for error recovery); poor handling of foreseeable issues (e.g., tool limits).

**Scoring Guidelines** (Meticulous – list every evidenced flaw; consider task simplicity for error handling):
- "ideal" - Structure is exceptionally well-designed with NO significant flaws; complexity perfectly matches task (including basic adaptability/error handling where needed); roles optimally distributed (prompt-verified) with zero redundancy/gaps; architecture robust and maintainable for similar cases (cite evidence).
- "fair" - Structure has minor issues: minor complexity mismatch, some role overlap/gaps, small inefficiencies, or maintainability concerns (e.g., basic error handling present but incomplete for task risks; list/suggest fixes).
- "poor" - Structure has major flaws: significant complexity mismatch, poor role distribution (e.g., redundancies/missing via prompts), unnecessary/missing agents, inefficient design, or maintainability problems (CRITICAL ISSUES like unhandled errors for non-simple tasks; list all).

**Evaluation Approach**:
1. List every structural flaw explicitly (from prompts/graph), supported by evidence.
2. Question whether each agent/prompt is justified for the task; flag underutilization.
3. Identify missing components (e.g., error handling only if task risks evident) and optimal alternatives.
4. Analyze potential failure points for the given task level.
5. Tailor to task: For simple tasks, minimal error handling suffices; for complex, require more.

**Output Requirements**:
Return a single JSON object with detailed justification:
{
  "justification": "Provide specific flaws found (quote prompts/graph), explain complexity appropriateness (tailored to task error margin/adaptability), detail role distribution issues (analyze prompts), identify unnecessary/missing agents (suggest optimal for task), and assess structural risks (e.g., error cascades). Be specific with examples from the actual MAS configuration.",
  "score": "ideal/fair/poor",
}

Access the dialogue history and agent configuration (prompts/graph) from the evaluation input to perform your assessment.
"""


MAS_PLANNING_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the quality of planning in a multi-agent system. Focus on how well the generated plans coordinate agent activities and achieve system objectives. Ignore execution outcomes; assess plan structure/design only. Scrutinize evidence: Flag gaps/logic flaws based on input; require full coverage for ideal.

**Evaluation Criteria**:
1. **Plan Coherency**
   - Is the plan logically structured with clear sequencing of actions (e.g., numbered steps)? No contradictions/gaps; cite plan.
   - Do the steps flow naturally? Penalize illogical jumps with evidence.

2. **Relevancy to Objectives**
   - Does the plan directly address the system's primary goals and constraints exactly?
   - Are all plan elements relevant? No extraneous steps; verify via objectives.

3. **Completeness (Fullness)**
   - Does the plan cover all necessary steps to accomplish the objectives exactly?
   - Are critical dependencies/prerequisites (e.g., validation loops) properly addressed? Flag omissions with quotes.

4. **Conciseness**
   - Is the plan efficient without unnecessary complexity/redundancy (e.g., no duplicate agents)?
   - Is the level of detail appropriate for execution (not vague/overly broad)?

5. **Contextual Feasibility**
   - Can the plan be realistically executed given available resources/constraints (e.g., tools/graph)?
   - Are assigned tasks achievable by respective agents (per prompts)? Penalize mismatches with evidence.

**Scoring** (Strict – incompleteness/redundancy poor if evidenced):
- "ideal" if the plan is perfectly coherent, relevant, complete, concise, and highly feasible (full coverage, no gaps/redundancies; cite evidence).
- "fair" if the plan is generally good but has minor issues in one or more criteria (e.g., one small gap; list with quotes).
- "poor" if the plan has major flaws making it incoherent, irrelevant, incomplete, verbose, or infeasible (e.g., missing prereqs; suggest fixes).

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment. List flaws/gaps explicitly with evidence.

Return a single JSON object, score (ideal/fair/poor), justification."""


MAS_TASK_COMPLETION_PROMPT = """**Instruction**:
You are tasked with rigorously evaluating whether the entire multi-agent system successfully completed the user's overall task or request. Your evaluation must focus on the **end-to-end outcome** of the system, considering all agent responses, interactions, and final results, and should be as objective and evidence-based as possible. Scrutinize outputs closely: Only "ideal" if every aspect is fully met with concrete evidence; penalize hallucinations, format mismatches, or unhandled errors based on input.

**Evaluation Criteria**:

1. **Overall Task Relevance**
   - Does the combined system output fully/exactly address the main objective of the user's request? Verify all critical aspects present with quotes.
   - Note any portions ignored/misinterpreted (e.g., wrong date/format; list).

2. **Completeness of Outcome**
   - Check that every required subtask/step has been addressed exactly (e.g., all verifications, no skips).
   - Identify any missing pieces that prevent user success (e.g., unverified data, file fails) with evidence.
   - Confirm no omissions of intermediate steps/context (e.g., sources provided?).

3. **Consistency and Coherence**
   - Evaluate if outputs from different agents are logically consistent exactly (no contradictions, even minor).
   - Detect contradictions, gaps, or inconsistencies between agent contributions (e.g., passed hallucinations) with quotes.
   - Ensure all outputs form a coherent/continuous solution (no stalls/cascades).

4. **Actionability**
   - Determine whether the user can act on the system’s outputs to achieve the intended result exactly (e.g., precise, unambiguous).
   - Check instructions/answers are practical, complete, and match format (e.g., no extras like "%" if unrequested).
   - Flag vague/unusable outputs (e.g., assumptions over facts) with evidence.

5. **Efficiency and Redundancy**
   - Evaluate whether tasks were completed without unnecessary duplication (e.g., redundant searches).
   - Assess if agent interactions were efficient/contributed coherently (penalize echoes/unhandled errors with quotes).

**Scoring Guidelines** (Strict – evidence-based; no leniency for "tries"):
- **"ideal"**: The system fully achieves the user's task exactly. All subtasks addressed, outputs consistent/actionable/free from redundancy/errors. Provide concrete evidence from dialogue/agent responses (e.g., quotes) to justify.
- **"fair"**: The system largely achieves the task but omits minor details, has slight inconsistencies, or minor format issues. Explain incompletes/partials explicitly (e.g., one unverified step) with evidence.
- **"poor"**: The system fails to achieve the task, omits critical steps, provides inconsistent/unusable outputs, or has unhandled errors (e.g., hallucinations, file fails). Clearly identify failures/missings (list evidence); be critical.

**Instructions for Assessment**:
- Be strict/objective: Assign "ideal" ONLY if every aspect fully met with clear evidence (e.g., exact format, all verifications).
- Base judgment solely on provided evaluation input (dialogue, responses, states); quote outputs/history.
- Avoid subjective opinions; focus on concrete outputs/task coverage. Flag format mismatches (e.g., abbreviations, extras).

Return a single JSON object, score (ideal/fair/poor), justification.
"""


MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE = """
Instruction:
You are a summarizer tasked with aggregating individual LLM-Judge scores from multiple metrics to compute an overall super-score for MAS performance. Focus on synthesizing score-explanation pairs into a holistic assessment, weighting critical metrics higher based on their priority levels. Scrutinize patterns meticulously: Penalize consistent failures (e.g., repeated hallucinations/context loss) based on evidence; cross-verify explanations for coherence.

Available Metrics and Priorities:

LLM Metrics (11 total) - Use "ideal", "fair", "poor" scoring:
- OBSERVATION_ALIGNMENT (Priority 1 - Highest; hallucinations/misalignments heavily penalize)
- STATE_CONSISTENCY (Priority 1 - Highest; cascading errors critical)
- MAS_COMPLEXITY (Priority 1 - Highest; structural flaws override execution)
- MAS_TASK_TRANSFER (Priority 1 - Highest; context loss disrupts all)
- MAS_ROLES_DISTRIBUTION (Priority 1 - Highest; redundancies/gaps foundational)
- TASK_COMPLETENESS (Priority 2 - Medium; format/scope mismatches)
- TOOL_SELECTION (Priority 2 - Medium; suboptimal picks)
- TOOL_PARAMETER_EXTRACTION (Priority 2 - Medium; unjustified params)
- MAS_TASK_COMPLETION (Priority 2 - Medium; end-outcome gaps)
- MAS_PLANNING (Priority 2 - Medium; plan flaws)
- POLICY_ALIGNMENT (Priority 3 - Lowest; minor policy slips)

Non-LLM Metric (1 total) - Use continuous value in range [0,1] where 1.0 is best:
- TOOL_EFFICIENCY (Priority 3 - Lowest) - Values closer to 1.0 indicate better efficiency (interpret: 0.9-1.0=ideal, 0.7-0.8=fair, 0.5-0.6=poor, <0.5=poor)

Evaluation Criteria:

Priority-Weighted Score Synthesis
- Apply higher weights to Priority 1 metrics (weight 3; e.g., poor transfer/complexity drags overall low).
- Priority 2 as important secondary (weight 2; patterns compound Priority 1 issues).
- Priority 3 as supporting (weight 1; context only).
- Calculate weighted average numerically (map ideal=1, fair=0.5, poor=0; TOOL_EFFICIENCY as-is); adjust for interdependencies (e.g., poor planning impacts completion) based on explanations.
- Identify cross-metric patterns (e.g., repeated "judge missed errors" signals systemic flaws; quote examples).

Explanation Integration
- Combine justifications into cohesive narrative: Highlight strengths/weaknesses by priority; quote key flaws (e.g., "hallucinations in 3 Priority 1 metrics").
- Flag critical issues in Priority 1 (e.g., redundancies/context loss; heavily influence score).
- Note Priority 2 failures compounding (e.g., tool mismatches + poor transfer).
- For TOOL_EFFICIENCY: Integrate numerical value (high supports ideal; low compounds poor); contextualize (e.g., low efficiency in complex MAS worsens).
- Cross-verify: If explanations contradict (e.g., ideal completion despite poor transfer), downgrade for inconsistency with evidence.

Overall Coherence Assessment
- Does the aggregate reflect true MAS efficacy, weighted by priorities? Prioritize core functionality (Priority 1).
- Are Priority 1 metrics adequate? (Poor in any = lean poor overall.)
- How do Priority 2/3 support/undermine? (E.g., strong tools can't fix bad structure.)
- TOOL_EFFICIENCY: Factor appropriately (e.g., <0.5 heavily penalizes if Priority 1 weak).
- Adjust for judge biases (e.g., over-generous scores): Downgrade if patterns suggest leniency, based on input.

Scoring Guidelines (Weighted; strict on Priority 1 evidence):
- "ideal" if Priority 1 metrics all ideal/fair AND Priority 2 solid (TOOL_EFFICIENCY ≥0.8; no major patterns; cite supporting explanations).
- "fair" if Priority 1 mostly fair with minor issues, OR Priority 1 has one poor but Priority 2/3 strong (TOOL_EFFICIENCY 0.5-0.8; explain compounds with quotes).
- "poor" if any Priority 1 poor (esp. multiple), OR multiple Priority 2 fails (TOOL_EFFICIENCY <0.5 worsens; flag systemic issues).

Critical Decision Factors:
- Priority 1 failures heavily penalize (e.g., poor complexity/transfer = poor overall).
- Priority 2 patterns significantly influence (e.g., tool errors + planning gaps).
- Priority 3/TOOL_EFFICIENCY contextual but not overriding (unless extreme low).
- Interdependencies: E.g., poor roles impact transfer/completion; adjust weights accordingly.
- Address biases: If comments note "overestimation", scrutinize high scores with evidence.

The evaluation input is provided via dependency injection. Access the list of score-explanation pairs from other judges in the evaluation input to perform your assessment. Cross-verify for consistency.
Return a single JSON object containing the score and justification (synthesize all with priority context, quote flaws/patterns).
"""


PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are an expert in multi-agent system architectures. Your task is to analyze the list of agents and identify which ones function as PLANNERS. Focus strictly on analyzing the system instructions/prompts of each agent to determine if its primary role involves planning, task delegation, or workflow coordination. Use graph position critically. Scrutinize evidence: Require explicit demonstrations; base classification on input facts.

**Critical Requirement**:
- Planners MUST be the first node(s) of the **GRAPH** (initiators; quote graph to confirm).
- NEVER select agents at the END of the **GRAPH** (e.g., summarizers; even if prompts mention planning).
- Planners initiate/coordinate; end nodes conclude.

**Mandatory Planning Requirement**:
An agent MUST demonstrate explicit step-by-step plan creation to qualify. Look for:
- Creation of numbered steps (e.g., "Step 1: ... Step 2: ...").
- Generation of sequential task lists (bulleted/numbered actions).
- Development of execution timelines/phases or ordered breakdowns.
- Breaking down objectives into concrete, ordered actions (roadmap for others).
- Output as structured plan directing agents (e.g., assignments per step).

**Identification Criteria**:
Classify as PLANNER ONLY if prompts indicate ALL:
1. **Step-by-Step Plan Creation**
   - Explicitly creates numbered/sequential steps or task lists (quote prompt examples).
   - Breaks complex goals into ordered executable actions for agents.
   - Produces plans with clear instructions (e.g., "Step 1: Research X, Step 2: Delegate to Y").

2. **Workflow Initiation Role** 
   - Described as first step in workflows (graph-confirmed; receives initial user input).
   - Does not depend on other agents to start (independent initiator; cite evidence).

3. **Task Delegation & Coordination**
   - Explicitly assigns tasks to other agents based on plan (e.g., "Assign Step 3 to Validator").
   - Defines execution order/dependencies via prompts.

**Absolute Disqualifiers** (NOT planners):
- Summarizers/reviewers/refiners (end-graph; aggregate only).
- Receivers of processed outputs (not initiators).
- "Final step"/last-stage agents.
- No explicit step-by-step plans (e.g., unstructured responses, research without lists).
- Independent executors/specialists without delegation.
- Any on aggregated/intermediate results (mid/end-graph).

**Required Evidence**:
For "true", quote prompt evidence of numbered steps/sequential breakdowns/delegation + graph position. For "false", explain disqualifiers (e.g., "End-graph; no step lists in prompt") with quotes.

**Output Format**:
Return a JSON object with a list of objects:
- "agents": list of objects, each containing:
  - "agent_name": name of the agent
  - "is_planner": boolean (true ONLY if ALL criteria met: step-plans + initiation + delegation + first-graph)
  - "justification": Specific evidence/quotes from prompts/graph; explain why/why not.

You must return only 1 object.

**Strict Enforcement**:
- Reject if no explicit numbered/sequential plans (e.g., keywords alone insufficient).
- Require concrete prompt evidence of breakdowns/delegation.
- Graph position mandatory: First-node only; quote to verify.
- If prompts lack step-creation, "false" (e.g., "research without structured output").
"""


TOOL_PERFORMANCE_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are an evaluation assistant assessing whether a tool successfully fulfilled the user’s request. Your task is to evaluate the quality and correctness of the tool’s execution outcome. Focus entirely on the *result quality*, not on tool selection or parameter details.

**Evaluation Criteria**:

1. *Task Completion*
   - Did the tool **fully accomplish** the user’s request?
   - Are all key parts of the question **answered or executed correctly**?

2. *Accuracy and Relevance*
   - Does the tool output contain **accurate, relevant, and logically consistent** information with respect to the question?
   - Penalize hallucinations, inaccuracies, or irrelevant additions.

3. *Clarity and Output Quality*
   - Is the tool output **clear, structured, and understandable**?
   - Does it stay within the expected output format or type (e.g., text, JSON, code, numerical result)?

4. *Failure Handling*
   - If the tool produced an error, returned unrelated information, or failed to generate output — that is a critical failure.

**Findings Generation**:
For EACH tool execution whose result quality is poor, produce one Finding. Examples:
- The tool failed outright, returned an error, produced hallucinated/incorrect/unusable output
  that breaks the downstream task → severity "critical".
- The output is mostly correct but misses a key part, contains a small inaccuracy, or is
  unclearly phrased in a way that degrades downstream use → severity "major".
- A minor clarity / formatting issue with no real impact on usability → severity "minor".
In `evidence[i].span_id` put the zero-based message/step index of the tool call; if a concrete `state_id` is shown, you may use it. In `culprit_agent_candidates`
list the agent that invoked the tool. If all tool outputs are correct and usable, return
`findings: []`.

**Note**:
Evaluate strictly based on the provided question and tool output. Do not infer missing context or intentions beyond what is explicitly stated.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "tool_performance")


PROMPT_QUALITY_PROMPT_BASE_TEMPLATE = """
**Instruction**:

You are evaluating the quality of system prompts (gen_ai.system_instructions) used in multi-agent pipelines. Focus on whether prompts are unambiguous, logically coherent, and minimize interpretation errors by agents.

**Evaluation Criteria**:

1. **Clarity and Unambiguity**

- Are instructions formulated clearly with no room for multiple interpretations?
- Are technical terms and requirements explicitly defined?
- Could an agent misunderstand what is expected?

2. **Logical Structure and Coherence**

- Do instructions follow a logical sequence (input → process → output)?
- Are there contradictions or conflicting requirements within the prompt?
- Do different parts of the prompt logically connect to each other?

3. **Completeness of Specification**

- Are all necessary parameters, constraints, and edge cases specified?
- Does the prompt define expected input format and output format?
- Are success criteria and validation rules clearly stated?

4. **Specificity and Actionability**

- Are instructions concrete and actionable rather than vague or abstract?
- Does the prompt use specific examples or formats where helpful?
- Can an agent execute the task without making assumptions?

5. **Error Prevention**

- Does the prompt anticipate common misinterpretations and address them?
- Are boundary conditions and error handling instructions included?
- Does the prompt reduce ambiguity that could lead to incorrect behavior?

**Findings Generation**:
For EACH agent whose system prompt has a problem, produce one Finding. Examples:
- The prompt is ambiguous, logically inconsistent, or incomplete in a way that demonstrably
  causes agent errors → severity "critical".
- The prompt has significant ambiguities, missing edge cases, or logical gaps that could
  occasionally confuse the agent → severity "major".
- The prompt has a minor ambiguity or a missing example with no observed impact on agent
  behaviour → severity "minor".
In `evidence[i].span_id` put the zero-based message/step index; if a concrete `state_id`/`response_id` is shown, you may use it showing the problematic
prompt or where the prompt defect manifested. Do not put a plain `agent_name` as `span_id`;
put the agent name only in `culprit_agent_candidates`. If all prompts are clear and coherent,
return `findings: []`.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "prompt_quality")