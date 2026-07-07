"""Prompts for Multi-Agent System (MAS) evaluation metrics."""

# Universal output-format requirements block (shared by all MAS evaluators).
# See individual.py for field documentation.
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
          "idx": "<zero-based message/step number from the trace, e.g. 0, 1, 2; strings are also OK>",
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
- The trace is shown as numbered messages/steps. Use that visible message number as `evidence[i].idx`.
- Message numbering starts from 0. If you are unsure whether the trace is zero-based or one-based, choose the closest message that contains the quote. The verifier tolerates +/-1.
- `idx` may be a JSON number (`29`) or a string (`"29"`). Both are valid.
- Do NOT use agent names as `idx`. Invalid: `WebSurfer`, `FileSurfer`, `Orchestrator`, `Orchestrator (thought)`, `Orchestrator (-> WebSurfer)`.
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


# LLM Evaluation Prompt for MAS Roles Distribution
MAS_ROLES_DISTRIBUTION_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the balance and distribution of roles among agents in a multi-agent system.
Focus on how evenly and appropriately responsibilities are allocated across the agent ecosystem.

**Evaluation Criteria**:
1. **Role Balance**
   - Are agent roles distributed evenly without overloading specific agents?
   - Is there a clear separation of responsibilities between different agents?

2. **Specialization Appropriateness**
   - Are agents specialized in appropriate domains based on their capabilities?
   - Does the role distribution match the complexity of the tasks being handled?

3. **Workload Distribution**
   - Is the workload reasonably balanced across all active agents?
   - Are there agents that are underutilized or overwhelmed?

**Findings Generation**:
For EACH structural problem with role distribution, produce one Finding. Examples:
- A critical role (e.g., validation for error-prone tasks) is entirely missing, or one agent
  is overloaded while the task fails downstream → severity "critical".
- Significant role overlap / redundancy, or one agent underutilised while others overloaded
  → severity "major".
- Minor imbalance, one unclear responsibility boundary, or a small redundancy with limited
  impact → severity "minor".
In `culprit_agent_candidates`, for MAS-level issues, list the agent(s) whose role design is
the cause, or the closest responsible role (e.g., "system architect"). In `evidence[i].idx`
put the visible message number that shows the imbalance in action. Put agent names only in `culprit_agent_candidates`.
If roles are well balanced,
return `findings: []`.

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "mas_roles_distribution")


# LLM Evaluation Prompt for MAS Agents Coordination - Task Transfer
MAS_TASK_TRANSFER_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the quality of task transfers between agents in a multi-agent system.
Focus on how smoothly and effectively tasks are handed off between different agents.

**Evaluation Criteria**:
1. **Transfer Clarity**
   - Are task transfers clearly communicated and well-documented?
   - Is the context properly preserved during handoffs?

2. **Transfer Efficiency**
   - Do transfers happen at appropriate points in the workflow?
   - Is there minimal information loss or duplication during transfers?

3. **Completion Continuity**
   - Do receiving agents successfully continue and complete transferred tasks?
   - Is there smooth progression without restarting or backtracking?

**Findings Generation**:
For EACH problematic task transfer, produce one Finding. Examples:
- An upstream agent passes an invalid/hallucinated context that the downstream agent consumes
  without validation, breaking the task → severity "critical" (`culprit_agent_candidates` =
  upstream agent; `evidence` with `root_cause` on the upstream item and `propagation` on the
  downstream item).
- Significant context loss during a handoff, or a receiver proceeding blindly with bad inputs,
  degrading but not destroying the outcome → severity "major".
- Minor duplication, a small unnecessary restart, or a single unvalidated but harmless input
  → severity "minor".
In `evidence[i].idx` put the zero-based message/step index (or concrete `state_id`/`response_id`) of the offending handoff; add a
second evidence entry for the propagation point if applicable. In `culprit_agent_candidates`
list the upstream agent that originated the bad context and/or the downstream agent that
accepted it unvalidated. If all transfers are seamless, return `findings: []`.

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "mas_task_transfer")


# LLM Evaluation Prompt for MAS Complexity
MAS_COMPLEXITY_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the complexity and interconnectedness of a multi-agent system.
Focus on the density of agents and the quality of their relationships.

**Evaluation Criteria**:
1. **Agent Density**
   - Is the number of agents appropriate for the system's scope and tasks?
   - Does the agent density match the complexity requirements?

2. **Interconnection Quality**
   - Are agent connections well-designed and efficient?
   - Is the communication network optimal for the workflow?

3. **System Scalability**
   - Can the system complexity accommodate growth and new requirements?
   - Is the architecture maintainable and extensible?

**Findings Generation**:
For EACH structural complexity problem, produce one Finding. Examples:
- The MAS is critically over- or under-complex for the task (e.g., a single-agent task forced
  into many agents with cascading failures, or a complex task with no error handling) →
  severity "critical".
- Significant redundant interconnections, missing basic error-handling roles for non-trivial
  tasks, or a fragile single point of failure → severity "major".
- A redundant agent, a minor unnecessary link, or a small maintainability concern with no
  real impact on this task → severity "minor".
In `culprit_agent_candidates`, for MAS-level structural issues, list the closest responsible
role (e.g., "system architect") or the agent whose design causes the problem. In
`evidence[i].idx` put a concrete `state_id`/`response_id` that demonstrates the issue.
Do not put a plain `agent_name` as `idx`; put agent names only in `culprit_agent_candidates`. If no explicit id is available, cite the zero-based message index.
If the complexity
is appropriate, return `findings: []`.

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "mas_complexity")


# LLM Evaluation Prompt for MAS Planning Quality
MAS_PLANNING_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are tasked with evaluating the quality of planning in a multi-agent system.
Focus on how well the generated plans coordinate agent activities and achieve system objectives.

**Evaluation Criteria**:
1. **Plan Coherency**
   - Is the plan logically structured with clear sequencing of actions?
   - Do the steps flow naturally without contradictions or gaps in logic?

2. **Relevancy to Objectives**
   - Does the plan directly address the system's primary goals and constraints?
   - Are all plan elements relevant to achieving the desired outcomes?

3. **Completeness (Fullness)**
   - Does the plan cover all necessary steps to accomplish the objectives?
   - Are critical dependencies and prerequisites properly addressed?

4. **Conciseness**
   - Is the plan efficient without unnecessary complexity or redundant steps?
   - Is the level of detail appropriate for execution?

5. **Contextual Feasibility**
   - Can the plan be realistically executed given available resources and constraints?
   - Are the assigned tasks achievable by the respective agents?

**Findings Generation**:
For EACH flaw in the plan, produce one Finding. Examples:
- The plan is incoherent, misses critical prerequisites, or assigns unreachable tasks, causing
  the workflow to fail → severity "critical".
- The plan has major gaps, redundant/duplicate steps, or assigns tasks to agents that cannot
  perform them → severity "major".
- A minor extraneous step, a small verbosity issue, or an unclear but functional step
  → severity "minor".
In `culprit_agent_candidates` list the planning agent (the planner) that produced the flawed
plan. In `evidence[i].idx` put the zero-based message/step index or concrete `state_id`/`response_id` of the plan element (and, if
the planner is identified, cite a concrete `state_id`/`response_id` produced by the planner).
Do not put a plain `agent_name` as `idx`. If no planning flaws are found, return
`findings: []`.

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "mas_planning")


MAS_TASK_COMPLETION_PROMPT = """**Instruction**:
You are tasked with rigorously evaluating whether the entire multi-agent system successfully completed the user's overall task or request.
Your evaluation must focus on the **end-to-end outcome** of the system, considering all agent responses, interactions, and final results, and should be as objective and evidence-based as possible.

**Evaluation Criteria**:

1. **Overall Task Relevance**
   - Does the combined system output fully address the main objective of the user's request?
   - Verify that all critical aspects of the task are present in the system output.
   - Note any portions of the task that were ignored or misinterpreted.

2. **Completeness of Outcome**
   - Check that every required subtask or step has been addressed.
   - Identify any missing pieces that would prevent the user from successfully achieving their goal.
   - Confirm that the solution does not omit important intermediate steps or context.

3. **Consistency and Coherence**
   - Evaluate if outputs from different agents are logically consistent.
   - Detect contradictions, gaps, or inconsistencies between agent contributions.
   - Ensure that all outputs together form a coherent and continuous solution.

4. **Actionability**
   - Determine whether the user can act on the system's outputs to achieve the intended result.
   - Check that instructions, answers, or outputs are practical, complete, and unambiguous.
   - Flag any vague or unusable outputs.

5. **Efficiency and Redundancy**
   - Evaluate whether tasks were completed without unnecessary duplication.
   - Assess whether agent interactions were efficient and contributed to a coherent final result.

**Findings Generation**:
For EACH way in which the system FAILED to complete the user's task, produce one Finding.
Examples:
- The system fails outright, omits a critical step, returns an unusable/contradictory final
  output, or propagates an unhandled hallucination → severity "critical".
- The system largely completes the task but a subtask is incomplete, outputs are slightly
  inconsistent, or one step is unverified → severity "major".
- A minor omission (e.g., a missing source citation) or a slight format mismatch that does
  not affect task usability → severity "minor".
In `culprit_agent_candidates` list the agent(s) whose contribution caused the failure (e.g.,
the agent that produced the wrong final answer, or the one that introduced a propagated
error). In `evidence[i].idx` put the zero-based message/step index or concrete `response_id`/`state_id` of the offending output
(and add a `propagation` evidence entry if the error cascaded). If the system fully and
correctly completes the task, return `findings: []`.

**Instructions for Assessment**:
- Be strict and objective.
- Base your judgment solely on the **provided evaluation input**, including dialogue history, agent responses, and state information.
- Avoid subjective opinions; focus on concrete outputs and task coverage.

""" + _FINDINGS_OUTPUT_INSTRUCTIONS.replace("<metric_name>", "mas_task_completion")