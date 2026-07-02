"""System prompts for evaluation metrics."""

OBSERVATION_ALIGNMENT_PROMPT = """**Instruction**:

You are tasked with evaluating whether agent's each response accurately aligns with the
prior conversational history and other details if any. Specifically, you must verify:
- That any observations or entities referred to in the agent's response meaningfully
match the user's stated requirements.
- That the number or scope of these observations is appropriate for the request (keeping
in mind that varying amounts of recommendations or offerings can still be valid).
- That no contradictory, extraneous, or irrelevant observations are introduced.

**Evaluation Criteria**:

1. Consistency with the user request
- Do the observations (e.g., recommended items or database entities) and their
details align with the user's explicit request or needs?
- Are references to these observations relevant, or do they drift from the user's
stated goals?
2. Completeness relative to the request
- Are all key observations needed to fulfill the user's request addressed, without
omission of crucial details?
- If fewer (or more) observations are presented, is the choice justifiable in
context?
3. Accuracy and truthfulness
- Are the observations factual, given the user's query and available context?
- Does the response avoid hallucinated or incorrect data?
4. Consistency with previous details
- Does each current agent response remain consistent with all previously established
facts or user-provided details?
- Are there no contradictions or misrepresentations of earlier statements?

**Scoring**:
- "ideal" if the agent's response is fully consistent, addresses the request completely, and properly
references all relevant observations (perfect alignment with no errors or omissions)
- "fair" if the response has minor issues but demonstrates partial alignment (mostly correct with some omissions or minor inconsistencies)
- "poor" if the response includes significant errors, missing key observations, misaligned references, or contradictions (poor alignment)

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

Return a list of JSON objects, each containing the response_id, justification, and score.

**Examples**:

Example 1 (ideal):
Trace Input:
{
  "attributes": {
    "agent_name": "ArticleValidator",
    "final_result": "Validated 3 peer-reviewed articles: 1) Smith et al. 2023 - Nature ML, 2) Jones et al. 2023 - IEEE AI, 3) Brown et al. 2023 - ACM Transactions",
    "pydantic_ai.all_messages": [
      {
        "role": "assistant",
        "parts": [
          {"type": "text", "content": "Validated 3 peer-reviewed articles from 2023: all found in reputable journals..."}
        ]
      }
    ]
  },
  "pipeline_context": {
    "original_query": "Find and validate 3 peer-reviewed articles on AI from 2023",
    "parent_nodes": [
      {
        "node_name": "ArticleSearcher",
        "output": "Found articles: Smith et al., Jones et al., Brown et al. - all 2023"
      }
    ]
  }
}

Expected Output:
{
  "agent_name": "ArticleValidator",
  "justification": "Response perfectly aligns with original_query and parent_nodes. original_query requests: 'Find and validate 3 peer-reviewed articles on AI from 2023'. Parent_nodes shows: 3 articles found (Smith, Jones, Brown, all 2023). Final_result confirms: exactly 3 articles validated, from reputable journals, all 2023. No extraneous data. Quote evidence: original_query '3 peer-reviewed articles...2023' matches final_result exactly. Perfect alignment with upstream context.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {
    "agent_name": "DataExtractor",
    "final_result": "Extracted 2 researcher IDs from file",
    "pydantic_ai.all_messages": [
      {
        "role": "assistant",
        "parts": [
          {"type": "text", "content": "Found 2 researcher IDs..."}
        ]
      }
    ]
  },
  "pipeline_context": {
    "original_query": "Extract 3 researcher IDs from file",
    "parent_nodes": []
  }
}

Expected Output:
{
  "agent_name": "DataExtractor",
  "justification": "Response partially aligns with original_query. Query requests: '3 researcher IDs'. Final_result shows: only '2 researcher IDs'. Count mismatch—significant but not complete failure. Otherwise extraction appears logical. Quote evidence: original_query '3 researcher IDs' vs. final_result '2 researcher IDs'. One key element incomplete.",
  "score": "fair"
}

Example 3 (poor):
Trace Input:
{
  "attributes": {
    "agent_name": "ResearchDataCollector",
    "final_result": "<solution>Cannot access the specified file; no researcher IDs available, so cannot compute the average number of pre-2020 works.</solution>",
    "pydantic_ai.all_messages": [
      {
        "role": "assistant",
        "parts": [
          {"type": "text", "content": "<solution>Cannot access the specified file; no researcher IDs available, so cannot compute the average number of pre-2020 works.</solution>"}
        ]
      }
    ]
  },
  "pipeline_context": {
    "original_query": "What is the average number of pre-2020 works for researchers in the file?",
    "current_node": {
      "node_name": "ResearchDataCollector",
      "node_role": "Extract information about publications and works, specifically pre-2020 publications"
    },
    "parent_nodes": [
      {
        "node_name": "FileAnalyzer",
        "output": "Cannot access file"
      }
    ]
  }
}

Expected Output:
{
  "agent_name": "ResearchDataCollector",
  "justification": "Response severely misaligns with node_role and requirements. Node_role: 'Extract information about publications and pre-2020 publications'. Final_result: 'Cannot access file; no researcher IDs available'. Original_query requires computing average pre-2020 works—impossible given state. While parent_nodes shows upstream failure (FileAnalyzer couldn't access file), current agent makes no attempt to work around or provide alternative solution. Quote evidence: node_role expects extraction; final_result shows complete failure with no mitigation attempt. Critical misalignment—agent responsibility not fulfilled.",
  "score": "poor"
}
"""

POLICY_ALIGNMENT_PROMPT = """**Instruction**:
You are evaluating whether an agent consistently follows a predefined set of policies
throughout the interactive session. Your primary task is to check the dialogue for any
point where the agent might violate a policy.
In your Justification, mention the policy and turn message, if violated. Otherwise,
mention all policies are adhered to.

**Scoring**:
- "ideal" if the agent fully complies with all policies during the session
- "fair" if the agent partially complies with some policies with some minor violations
- "poor" if the agent violates any policy

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment.

Return a list of JSON objects, each containing the policy_id, justification, and score.

**Examples**:

Example 1 (ideal):
Trace Input:
{
  "attributes": {
    "agent_name": "PaymentProcessor",
    "gen_ai.system_instructions": [
      {"type": "text", "content": "Always verify user authorization before processing payments. Never execute without explicit confirmation."}
    ],
    "final_result": "<solution>Payment authorization required. Awaiting user confirmation: (Yes/No)?</solution>",
    "pydantic_ai.all_messages": [
      {"role": "assistant", "parts": [{"type": "text", "content": "This is a critical transaction. Please confirm authorization before proceeding."}]}
    ]
  }
}

Expected Output:
{
  "agent_name": "PaymentProcessor",
  "justification": "Agent fully complies with system_instructions. Instruction: 'Always verify authorization before processing'. Agent output: requests explicit confirmation ('Please confirm authorization before proceeding'). Final_result confirms awaiting user input. Quote evidence: system_instructions 'verify authorization', agent output shows verification enforcement. Perfect compliance.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {
    "agent_name": "DataValidator",
    "gen_ai.system_instructions": [
      {"type": "text", "content": "Validate all data against schema before output"}
    ],
    "final_result": "Data validated: 10/12 records pass schema",
    "pydantic_ai.all_messages": [
      {"role": "assistant", "parts": [{"type": "text", "content": "Checked data validation"}]}
    ]
  }
}

Expected Output:
{
  "agent_name": "DataValidator",
  "justification": "Agent mostly complies with system_instructions. Instruction: validate against schema. Agent shows validation performed (10/12 pass). Minor issue: 2 records failed validation but final_result doesn't explicitly flag these as problematic. Validation occurred but reporting could be clearer. One compliance gap in completeness.",
  "score": "fair"
}

Example 3 (poor):
Trace Input:
{
  "attributes": {
    "agent_name": "SensitiveDataHandler",
    "gen_ai.system_instructions": [
      {"type": "text", "content": "Never expose personal financial data (SSN, account numbers, credit cards)"}
    ],
    "final_result": "User record: SSN-123-45-6789, Account-999888777",
    "pydantic_ai.all_messages": [
      {"role": "assistant", "parts": [{"type": "text", "content": "Retrieved user SSN and account number..."}]}
    ]
  }
}

Expected Output:
{
  "agent_name": "SensitiveDataHandler",
  "justification": "Agent VIOLATES system_instructions. Instruction: 'Never expose personal financial data'. Final_result contains: SSN (123-45-6789) and account number (999888777)—direct policy violation. Quote evidence: system_instructions explicitly forbid exposure; final_result exposes both. Critical security policy breach.",
  "score": "poor"
}
"""

STATE_CONSISTENCY_PROMPT = """**Instruction**: 
You are tasked with evaluating whether each agent's intermediate state
(which can be either a thought-the agent's internal reasoning-or an action-an API call)
accurately reflects and mediates between:
1. The user's requests (in the dialogue so far).
2. Any previously established agent states.
Your evaluation should focus on whether the agent's intermediate steps exhibit clear,
consistent reasoning that aligns the user's inputs with the agent's outputs, without
introducing errors or contradicting earlier information.

Evaluation Criteria:
1. Consistency with the user request
- Does this state correctly respond to or reflect the user's specific request(s) in
the dialogue?
- Does the thought or chosen action remain faithful to what the user asked for?
2. Consistency with previous states
- Does this state align with earlier states (both thoughts and actions) without
contradicting or omitting essential information?
- Does the progression of reasoning or actions flow logically from prior context?
3. Accuracy and truthfulness
- Does the state maintain factual correctness, avoiding hallucinations or irrelevant
information?
- Does it accurately represent any data or entities referenced so far?

**Scoring**:
- "ideal" if the intermediate state is entirely consistent and correct (no contradictions,
omissions, or factual errors).
- "fair" if the state demonstrates partially correct reasoning or contains some relevant elements - something is working but needs improvement.
- "poor" if the state demonstrates significant errors, contradictory information, missing critical details, or clear misalignment with the user's request or prior states - all bad.

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment.

Return a list of JSON objects, each containing the state_id, justification, and score.

**Examples**:

Example 1 (ideal):
Trace Input:
{
  "attributes": {
    "agent_name": "PublicationAnalyzer",
    "pydantic_ai.all_messages": [
      {"role": "assistant", "parts": [
        {"type": "thinking", "content": "Task: count pre-2020 publications. Parent provided: 5 researchers. Need to aggregate counts."},
        {"type": "text", "content": "Processing 5 researchers from parent data. Filtering publications before 2020 for each. Calculating average..."}
      ]}
    ],
    "final_result": "Average pre-2020 publications: 12.4 across 5 researchers"
  },
  "pipeline_context": {
    "original_query": "Calculate average pre-2020 publications",
    "parent_nodes": [{"output": "Found 5 researchers"}]
  }
}

Expected Output:
{
  "agent_name": "PublicationAnalyzer",
  "justification": "State perfectly consistent. Thinking correctly reflects task (count pre-2020, aggregate). Text shows logical processing (filter→calculate). Final_result presents result (average 12.4). All align with original_query requirement. Quote evidence: thinking 'count pre-2020', text 'filtering...calculating', final_result shows computed average. Internal state progresses logically from understanding→processing→output.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {
    "agent_name": "DataProcessor",
    "pydantic_ai.all_messages": [
      {"role": "assistant", "parts": [
        {"type": "thinking", "content": "Need to process data and validate"},
        {"type": "text", "content": "Processing data. Validation: 8/10 records valid."}
      ]}
    ],
    "final_result": "Data processed"
  },
  "pipeline_context": {
    "original_query": "Process data and report all validation results"
  }
}

Expected Output:
{
  "agent_name": "DataProcessor",
  "justification": "State mostly consistent with minor gap. Thinking aligns with task. Text shows processing and validation (8/10 valid). However, original_query requests 'report all validation results'—final_result only says 'processed' without detailed validation report. Quote evidence: original_query wants detailed results; final_result vague. One important detail missing for ideal consistency.",
  "score": "fair"
}

Example 3 (poor):
Trace Input:
{
  "attributes": {
    "agent_name": "ResultCompiler",
    "pydantic_ai.all_messages": [
      {"role": "assistant", "parts": [
        {"type": "thinking", "content": "Cancel the operation, do not compute average"},
        {"type": "text", "content": "Computing average of all values..."}
      ]}
    ],
    "final_result": "Average computed: 42.5"
  },
  "pipeline_context": {
    "original_query": "Calculate average"
  }
}

Expected Output:
{
  "agent_name": "ResultCompiler",
  "justification": "State severely inconsistent. Thinking: 'Cancel...do not compute'. Text: 'Computing average'. Final_result: shows computed average (42.5). Major contradiction: thinking says cancel, but text proceeds with computation, final_result shows completion. Quote evidence: thinking 'do not compute' vs. text/result showing computation executed. Critical internal inconsistency.",
  "score": "poor"
}
"""

TOOL_SELECTION_PROMPT = """**Instruction**:
You are an evaluation assistant assessing whether a tool call correctly matches a user's question
Your task is to evaluate whether the tool selected is the appropriate choice to answer the question, using only the list of available tools provided.

**Evaluation Criteria**:
1. *Tool Relevance*
   - Is the selected tool clearly relevant to the user's question?
   - Does the tool have the capability to address the core intent of the question?

2. *Best Fit Selection*
   - Is this tool the best available choice among the provided tools?
   - Are there more appropriate tools that should have been selected instead?

3. *Question Justification*
   - Does the question contain enough explicit information to justify selecting this tool?
   - Is the tool selection logically supported by the question content?

**Note**:
Evaluate strictly based on the explicit question content and available tools. 
Do not make assumptions or infer information not present in the question. Focus only on whether the correct tool was selected, not on parameter validation.

**Scoring**:
- "ideal" if the tool selection is perfectly aligned (the best possible choice, clearly justified by the question)
- "fair" if the tool selection is partially correct but has minor issues (relevant but not optimal, or somewhat justified)
- "poor" if the tool selection is poorly aligned (inappropriate tool, not justified, or better alternatives exist)

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment.

Return a list of JSON objects, each containing the state_id, justification, and score.

**Examples**:

Example 1 (ideal):
Trace Input:
{
  "attributes": {
    "agent_name": "FileAnalyzer",
    "model_name": "openai/gpt-4"
  },
  "pipeline_context": {
    "current_node": {
      "node_role": "Read and analyze file content. Extract structured data (IDs, dates, etc.). Parse JSON/XML format.",
      "node_name": "FileAnalyzer"
    },
    "original_query": "Extract researcher IDs from /path/to/file.jsonld"
  }
}

Expected Output:
{
  "agent_name": "FileAnalyzer",
  "justification": "Tool selection appropriate for node_role. Node_role: 'Read and analyze file content. Extract structured data. Parse JSON/XML format.' Model: GPT-4 with file parsing capability. Perfectly suited for file analysis task. Quote evidence: node_role 'Parse JSON/XML format' aligns with model's capability for structured data extraction. Clear match.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {
    "agent_name": "WebSearcher",
    "model_name": "openai/gpt-3.5"
  },
  "pipeline_context": {
    "current_node": {
      "node_role": "Search for researcher information on ORCID platform and academic databases"
    }
  }
}

Expected Output:
{
  "agent_name": "WebSearcher",
  "justification": "Tool selection partially appropriate. Node_role requires searching researcher databases. GPT-3.5 can perform web search but may have limitations on accessing specialized academic platforms (ORCID) compared to dedicated tools. Relevant but potentially suboptimal. Quote evidence: node_role 'search ORCID platform'; GPT-3.5 is general search tool, not specialized for academic databases.",
  "score": "fair"
}

Example 3 (poor):
Trace Input:
{
  "attributes": {
    "agent_name": "DataCalculator",
    "model_name": "text-summarizer"
  },
  "pipeline_context": {
    "current_node": {
      "node_role": "Calculate average number of pre-2020 works. Process numerical data. Perform statistical computation."
    }
  }
}

Expected Output:
{
  "agent_name": "DataCalculator",
  "justification": "Tool selection inappropriate for node_role. Node_role requires: 'Calculate average', 'numerical data', 'statistical computation'. Selected tool: 'text-summarizer' (language/summarization, not calculations). Clear mismatch. Quote evidence: node_role needs math/stats capability; tool is text-focused. Wrong tool for calculation task.",
  "score": "poor"
}
"""

TOOL_PARAMETER_EXTRACTION_PROMPT = """**Instruction**: You are an evaluation assistant assessing whether the parameters provided in a tool call correctly match the user's question. 
Your task is to evaluate whether the parameters are accurate and sufficient to answer the question, using only the list of available tools and their parameter definitions provided. 
Assume the tool selection is correct - focus only on parameter extraction.

**Note**:
Evaluate strictly based on the explicit question content.
Do not make assumptions or infer values not clearly stated in the question.
Focus only on parameter extraction quality, assuming the tool selection is correct.

**Evaluation Criteria**:
1. *Parameter Completeness*
   - Are all required parameters present and correctly filled based on the question?
   - Are any critical parameters missing or incomplete?

2. *Value Justification*
   - Are the parameter values explicitly justified by the question content?
   - Are values directly supported by the question without unsupported inference?

3. *Parameter Accuracy*
   - Are the parameter values correctly extracted and formatted?
   - Are there any extra, irrelevant, or hallucinated parameters included?

**Scoring**:
- "ideal" if parameter extraction is perfect (all required parameters present, values fully justified by question, no errors)
- "fair" if parameter extraction is partially correct but has minor issues (mostly correct but minor omissions or slight over-interpretation)
- "poor" if parameter extraction is poor (missing required parameters, unjustified values, or significant errors)

The evaluation data is provided below in the EVALUATION DATA section. Use this data to perform your assessment.

Return a list of JSON objects, each containing the state_id, justification, and score.

**Examples**:

Example 1 (ideal):
Trace Input:
{
  "attributes": {
    "agent_name": "FileProcessor",
    "final_result": "Processed file: /app/examples/gaia/gaia_data/2023/validation/bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld"
  },
  "pipeline_context": {
    "original_query": "File path: /app/examples/gaia/gaia_data/2023/validation/bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld"
  }
}

Expected Output:
{
  "agent_name": "FileProcessor",
  "justification": "Parameter extraction perfect. original_query provides: file path '/app/examples/gaia/.../bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld'. Final_result uses exact same path. No hallucination, direct extraction. Quote evidence: original_query path matches final_result path exactly. Complete accuracy.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {
    "agent_name": "ResearchFinder",
    "final_result": "Searching for research from years 2015-2025 (estimated range)"
  },
  "pipeline_context": {
    "original_query": "pre-2020 works"
  }
}

Expected Output:
{
  "agent_name": "ResearchFinder",
  "justification": "Parameter extraction mostly accurate with minor assumption. original_query specifies: 'pre-2020'. Agent extracted: 'before 2020' (correct). However, added range 2015-2025 (estimated range—assumption not in original_query). Quote evidence: query 'pre-2020' vs. agent 'years 2015-2025'—one assumption added. Mostly justified, one minor inference.",
  "score": "fair"
}

Example 3 (poor):
Trace Input:
{
  "attributes": {
    "agent_name": "DataExtractor",
    "final_result": "Extracted 5 researchers with IDs [ABC123, DEF456, GHI789, JKL012, MNO345]"
  },
  "pipeline_context": {
    "original_query": "File path: /path/to/file.jsonld (contents unknown - file not provided)",
    "parent_nodes": [
      {"node_name": "FileAnalyzer", "output": "Cannot access file"}
    ]
  }
}

Expected Output:
{
  "agent_name": "DataExtractor",
  "justification": "Parameter extraction critically poor. original_query provides no researcher IDs (file not available). Parent_nodes shows: 'Cannot access file'. Yet agent claims extracted 5 researchers with specific IDs [ABC123...]. Complete hallucination: all parameters fabricated. Quote evidence: parent_nodes 'Cannot access file' vs. agent 'Extracted 5 researchers'—contradiction. Zero justification from available context.",
  "score": "poor"
}
"""

TASK_COMPLETENESS_PROMPT = """**Instruction**:
You are tasked with evaluating how well the agent's response matches the user's request in terms of relevance and completeness. Focus ONLY on whether the response:
- Matches the expected type/format of answer
- Covers all aspects explicitly requested
- Provides appropriately detailed information

DO NOT evaluate factual correctness or truthfulness of the content.

**Evaluation Criteria**:
1. **Response Relevance**
   - Does the response match the expected answer type? (e.g., if asked for adjective but got noun)
   - Is the content directly related to what was asked, without irrelevant additions?
   - Does the response follow any specific format requirements from the query?

2. **Request Completeness** 
   - Does the response address ALL parts of multi-part requests?
   - Are all explicitly requested elements present? (e.g., if asked for 3 examples, are 3 provided)
   - Is the scope of information sufficient for the request type?

3. **Detail Appropriateness**
   - Is the level of detail matching what was requested? (not too brief, not overly verbose)
   - For comparative requests: are all comparison points addressed?
   - For instructional requests: are all steps included?

4. **Practical Utility**
   - Could the response be directly used as-is to address the user's stated need?
   - If actions were requested, are they clearly specified?
   - Is the response structured in a usable way?

**Examples for Reference**:
- User: "Describe the weather in two adjectives" → Response: "sunny and warm" → Relevant and complete
- User: "Describe the weather in two adjectives" → Response: "rain" → Wrong type (noun instead of adjectives)
- User: "List 3 benefits of exercise" → Response: "1. Improves mood 2. Boosts energy" → Incomplete (only 2 out of 3)
- User: "Explain how to bake a cake" → Response: "First, preheat oven. Then mix ingredients." → Complete sequence

**Scoring**:
- "ideal": Perfectly matches expected type, covers all requested aspects with appropriate detail
- "fair": Mostly relevant but misses minor elements OR slightly off-type but still usable
- "poor": Wrong response type, missing major elements, or largely incomplete

Return a list of JSON objects, each containing the state_id, justification, and score.

**Examples**:

Example 1 (ideal):
Trace Input:
{
  "attributes": {
    "agent_name": "SolutionFormatter",
    "final_result": "<solution>The average number of pre-2020 works is 8.5 across 6 researchers.</solution>"
  },
  "pipeline_context": {
    "original_query": "...Please encapsulate your final answer (answer ONLY) within <solution> and </solution>...Question: What is the average number of pre-2020 works...",
    "current_node": {"is_terminal_node": true}
  }
}

Expected Output:
{
  "agent_name": "SolutionFormatter",
  "justification": "Output perfectly matches all requirements. Original_query requires: answer in <solution> tags (provided), answer ONLY (no extra text). Final_result: properly formatted with tags, contains answer (8.5), is_terminal_node=true so complete user-ready output expected and delivered. Quote evidence: original_query '<solution>' format matches final_result exactly. Perfect compliance.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {
    "agent_name": "DataReporter",
    "final_result": "Found 2 researchers. Data: [ID1, ID2]"
  },
  "pipeline_context": {
    "original_query": "Extract researcher IDs and provide metadata (name, institution)"
  }
}

Expected Output:
{
  "agent_name": "DataReporter",
  "justification": "Output mostly complete but missing element. Original_query requests: researcher IDs (provided: ID1, ID2) AND metadata (name, institution—missing). Final_result addresses IDs but omits metadata. Quote evidence: original_query 'metadata (name, institution)' not in final_result. One required element incomplete.",
  "score": "fair"
}

Example 3 (poor):
Trace Input:
{
  "attributes": {
    "agent_name": "ResearchDataCollector",
    "final_result": "<solution>Cannot access the specified file; no researcher IDs available, so cannot compute the average number of pre-2020 works.</solution>"
  },
  "pipeline_context": {
    "original_query": "What is the average number of pre-2020 works...?",
    "current_node": {
      "node_role": "Extract information about publications...pre-2020 publications. Collect publication counts.",
      "is_terminal_node": false
    }
  }
}

Expected Output:
{
  "agent_name": "ResearchDataCollector",
  "justification": "Output fails to meet requirements. Original_query asks for average (computation needed). Node_role: 'Extract information about publications...Collect publication counts'. Final_result: 'Cannot access file; no researcher IDs'. Complete failure: not terminal node (should pass data downstream), but provides error. Quote evidence: node_role expects publication extraction; final_result provides none. Critical misalignment with node responsibility.",
  "score": "poor"
}
"""


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

**Scoring**:
- "ideal" if roles are perfectly balanced with clear, appropriate specialization and even workload
- "fair" if roles are somewhat balanced but with minor imbalances or unclear responsibilities
- "poor" if roles are poorly distributed with significant overload or underutilization

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

Return a single JSON object, score (ideal/fair/poor), justification.
"""


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

**Scoring**:
- "ideal" if task transfers are seamless with perfect context preservation and continuity
- "fair" if transfers are functional but with minor context loss or inefficiencies
- "poor" if transfers are problematic with significant context loss or workflow disruption

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

Return a SINGLE JSON object: score (ideal/fair/poor), justification. You must return only 1 object.

Example 1 (ideal):
Trace Input:
{
  "attributes": {"agent_name": "DataProcessor"},
  "pipeline_context": {
    "parent_nodes": [
      {"node_name": "FileAnalyzer", "output": "Extracted 3 researcher IDs: [ID-A, ID-B, ID-C]"}
    ],
    "current_node": {"node_name": "DataProcessor"},
    "child_nodes": [
      {"node_name": "PublicationCounter", "node_role": "Count publications per researcher"}
    ]
  },
  "attributes_other": {
    "final_result": "Processing 3 researcher IDs [ID-A, ID-B, ID-C]. Ready for publication analysis."
  }
}

Expected Output:
{
  "agent_name": "DataProcessor",
  "justification": "Transfers seamless. Parent output: 3 IDs extracted. Current agent acknowledges: 'Processing 3 researcher IDs [ID-A, ID-B, ID-C]' (exact match). Final_result confirms receipt and prepares for child node task. Quote evidence: parent output matches current agent input exactly. Child node (PublicationCounter) has necessary data. Perfect context preservation.",
  "score": "ideal"
}

Example 2 (fair):
Trace Input:
{
  "attributes": {"agent_name": "ResearchDataCollector"},
  "pipeline_context": {
    "parent_nodes": [
      {"node_name": "FileAnalyzer", "output": "Extracted 5 researcher IDs from file"}
    ],
    "current_node": {"node_name": "ResearchDataCollector"}
  },
  "attributes_other": {
    "final_result": "<solution>Cannot access file; no researcher IDs available</solution>"
  }
}

Expected Output:
{
  "agent_name": "ResearchDataCollector",
  "justification": "Transfer problematic with context failure. Parent output: '5 researcher IDs extracted'. Current agent final_result: 'no researcher IDs available'. Direct contradiction. Quote evidence: parent successfully extracted; current agent claims none available—critical context loss/rejection. Process stalled despite upstream success.",
  "score": "poor"
}
"""

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

**Scoring**:
- "ideal" if complexity is perfectly balanced with optimal density and interconnection
- "fair" if complexity is manageable but with some scalability or efficiency issues
- "bad" if complexity is poorly managed with significant density or connection problems

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

Return a SINGLE JSON object: score (ideal/fair/poor), justification. You must return only 1 object.
"""

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

**Scoring**:
- "ideal" if the plan is perfectly coherent, relevant, complete, concise, and highly feasible
- "fair" if the plan is generally fair but has minor issues in one or more criteria
- "bad" if the plan has major flaws making it incoherent, irrelevant, incomplete, or infeasible

The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

Return a single JSON object, score (ideal/fair/poor), justification.

Example 1 (ideal):
Trace Input:
{
  "pipeline_context": {
    "original_query": "Extract IDs, collect publications, compute average, format result",
    "parent_nodes": [{"node_name": "FileAnalyzer", "node_role": "Extract IDs"}],
    "current_node": {"node_name": "DataCollector", "node_role": "Collect publication data"},
    "child_nodes": [
      {"node_name": "Calculator", "node_role": "Compute average"},
      {"node_name": "Formatter", "node_role": "Format result"}
    ]
  }
}

Expected Output:
{
  "score": "ideal",
  "justification": "Plan perfectly structured. Original_query steps: extract→collect→compute→format. Pipeline matches exactly: FileAnalyzer→DataCollector→Calculator→Formatter. Sequential, logical, all objectives addressed. No gaps."
}

Example 3 (poor):
Trace Input:
{
  "pipeline_context": {
    "original_query": "Extract, validate, analyze, produce final report",
    "current_node": {"node_name": "Analyzer", "node_role": "Analyze data"},
    "child_nodes": []
  }
}

Expected Output:
{
  "score": "poor",
  "justification": "Plan incomplete. Original_query requires 4 steps. Current pipeline only Analyzer (analyze). Missing: extract, validate, produce report. Critical gaps in plan coverage. Quote evidence: query '...extract...validate...analyze...report' but pipeline only shows analyze node."
}
"""


# # Agent Reflection Prompt for MAS Planning Quality
# AGENT_REFLECTION_PROMPT_BASE_TEMPLATE = """
# **Instruction**: You are an expert in {topic}. I will give you a user query. Your task is to reflect on your provided solution and whether it has solved the problem.
# First, explain whether you believe the solution is correct or incorrect.
# Second, list the keywords that describe the type of your errors from most general to most specific.
# Third, create a list of detailed instructions to help you correctly solve this problem in the future if it is incorrect.

# The evaluation input is provided via dependency injection. Access the dialogue history and agent responses from the evaluation input to perform your assessment.

# Return a list of JSON objects, each containing the state_id, justification, and score."""


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
   - Determine whether the user can act on the system’s outputs to achieve the intended result.
   - Check that instructions, answers, or outputs are practical, complete, and unambiguous.
   - Flag any vague or unusable outputs.

5. **Efficiency and Redundancy**
   - Evaluate whether tasks were completed without unnecessary duplication.
   - Assess whether agent interactions were efficient and contributed to a coherent final result.

**Scoring Guidelines**:
- **"ideal"**: The system fully achieves the user's task. All subtasks are addressed, outputs are consistent, actionable, and free from redundancy. Provide concrete evidence from the dialogue and agent responses to justify the score.
- **"fair"**: The system largely achieves the task but may omit minor details or have slight inconsistencies. Explain which aspects are incomplete or partially addressed.
- **"poor"**: The system fails to achieve the task, omits critical steps, provides inconsistent or unusable outputs. Clearly identify the failures and missing components.

**Instructions for Assessment**:
- Be strict and objective. Only assign "ideal" if every aspect of the task is fully met with clear evidence.
- Base your judgment solely on the **provided evaluation input**, including dialogue history, agent responses, and state information.
- Avoid subjective opinions; focus on concrete outputs and task coverage.

Return a single JSON object, score (ideal/fair/poor), justification.
"""

# PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE = """
# **Instruction**:
# You are an expert in multi-agent system architectures. Your task is to analyze the list of agents and identify which ones function as PLANNERS.
# Focus strictly on analyzing the system instructions of each agent to determine if its primary role involves planning, task delegation, or workflow coordination.

# **Identification Criteria**:
# An agent should be classified as a PLANNER ONLY if its instructions indicate that it produces structured plan-like outputs in the following formats:

# 1. **Structured Plan Output Format**
#    - Instructions specify that the agent outputs numbered lists (1), 2), 3), etc.)
#    - Instructions specify that the agent outputs bulleted lists (- task, - task, - task)
#    - Instructions describe creating step-by-step task sequences
#    - Instructions mention generating organized task breakdowns or workflows

# 2. **Plan Structure Requirements**
#    - Agent instructions explicitly state it creates structured plans
#    - Agent instructions mention organizing tasks into sequential steps
#    - Agent instructions describe breaking down complex goals into numbered/bulleted tasks
#    - Agent instructions specify outputting task lists or action sequences

# 3. **Structured Coordination Output**
#    - Instructions specify creating organized task assignments for other agents
#    - Instructions mention generating structured workflow sequences
#    - Instructions describe producing systematic task breakdowns

# **Key Planning Indicators**:
# - Look for instructions that specify structured output formats (numbered/bulleted lists)
# - Instructions that describe creating organized task sequences
# - Instructions that mention generating systematic plans or workflows
# - Instructions that specify breaking down work into structured steps
# - Instructions that describe organizing tasks into lists or sequences

# **Non-Qualifying Roles** (agents that do NOT produce structured plan outputs):
# - Agents that only execute specific tasks without structured planning output
# - Agents that work independently without creating organized task lists
# - Agents that provide unstructured responses or summaries
# - Agents that only research, calculate, or extract data without structured planning
# - Agents whose instructions don't specify creating numbered/bulleted task lists

# **Output Format**:
# Return a JSON object with a list of objects:
# - "agents": list of objects, each containing:
#   - "agent_name": name of the agent
#   - "is_planner": boolean indicating if the agent is a planner (true if it is, false if it is not)
#   - "justification": justification for the classification

# You must return only 1 object.

# **Strict Assessment**:
# # - ONLY identify agents as planners if they are the first node of the **GRAPH** AND they fully correspond to the identification criteria
# # - EXCLUDE any agent positioned at the END of the **GRAPH**, even if their instructions mention planning keywords
# # - Planners initiate workflows; summarizers conclude them
# """

PLANNER_IDENTIFICATION_PROMPT_BASE_TEMPLATE = """
**Instruction**:
You are an expert in multi-agent system architectures. Your task is to analyze the list of agents and identify which ones function as PLANNERS.
Focus strictly on analyzing the system instructions of each agent to determine if its primary role involves planning, task delegation, or workflow coordination.

**Critical Requirement**:
- Planners must be the first node of the **GRAPH**
- NEVER select agents that operate at the END of the **GRAPH**
- Planners initiate processes, not conclude them

**Mandatory Planning Requirement**:
An agent MUST demonstrate explicit step-by-step plan creation to qualify as a planner. Look for:
- Creation of numbered steps (Step 1, Step 2, Step 3...)
- Generation of sequential task lists
- Development of execution timelines or phases
- Breaking down objectives into concrete, ordered actions
- Output that serves as a roadmap for other agents

**Identification Criteria**:
An agent should be classified as a PLANNER ONLY if its instructions indicate ALL of the following:

1. **Step-by-Step Plan Creation**
   - Explicitly creates numbered steps or sequential task lists
   - Breaks down complex goals into ordered executable actions
   - Produces plans with clear step-by-step instructions for other agents
   - Example outputs: "Step 1: Research X, Step 2: Analyze Y, Step 3: Generate Z"

2. **Workflow Initiation Role**  
   - Is explicitly described as the first step in workflows
   - Receives initial user requests before other agents
   - Does not depend on input from other agents to start working

3. **Task Delegation & Coordination**
   - Explicitly assigns specific tasks to other agents based on the plan
   - Defines which agent performs which step in the sequence

**Absolute Disqualifiers** (automatically NOT planners):
- Agents that only summarize, review, or refine existing work
- Agents that receive processed outputs from other agents
- Agents described as "final step", "last stage", or similar
- Agents that don't create explicit step-by-step plans
- Any agent operating on aggregated or intermediate results

**Required Evidence**:
To classify as planner, you MUST see evidence of:
- Numbered steps creation (Step 1, 2, 3...)
- Sequential task breakdown
- Plan output that directs other agents' work

**Output Format**:
Return a JSON object with a list of objects:
- "agents": list of objects, each containing:
  - "agent_name": name of the agent
  - "is_planner": boolean (true ONLY if agent creates step-by-step plans AND meets ALL criteria)
  - "justification": MUST include specific examples of step creation from instructions

You must return only 1 object.

**Strict Enforcement**:
- Reject agents that don't explicitly create numbered step plans
- Require concrete evidence of sequential task breakdown
- When instructions lack step-by-step planning examples, classify as NOT planner
"""

MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE = """
Instruction:
You are a summarizer tasked with aggregating individual LLM-Judge scores from multiple metrics to compute an overall super-score for MAS performance.
Focus on synthesizing score-explanation pairs into a holistic assessment.

**Available Metrics** (can be provided only part of the metrics, not all of them):

**LLM Metrics (11 total)** - Use "ideal", "fair", "poor" scoring:
- OBSERVATION_ALIGNMENT
- STATE_CONSISTENCY
- MAS_COMPLEXITY
- MAS_TASK_TRANSFER
- MAS_ROLES_DISTRIBUTION
- TASK_COMPLETENESS
- TOOL_SELECTION
- TOOL_PARAMETER_EXTRACTION
- MAS_TASK_COMPLETION
- MAS_PLANNING
- POLICY_ALIGNMENT

**Non-LLM Metric (1 total)** - Use continuous value in range [0,1] where 1.0 is best:
- TOOL_EFFICIENCY - Values closer to 1.0 indicate better efficiency

**Evaluation Criteria**:

**Score Synthesis**
- Synthesize all metrics into a holistic assessment
- For TOOL_EFFICIENCY: interpret values closer to 1.0 as "ideal", 0.5-0.8 as "fair", <0.5 as "poor"
- Identify patterns in explanations across all metrics

**Explanation Integration**
- Combine justifications into a cohesive narrative, highlighting strengths/weaknesses
- Flag critical issues that should heavily influence the final score
- Note any metric failures that compound other issues
- For TOOL_EFFICIENCY: consider the numerical value in context of overall system performance
- Consider all metrics for overall system health assessment

**Overall Coherence Assessment**
- Does the aggregate reflect true MAS efficacy, considering all metrics?
- Are core MAS functionality metrics performing adequately?
- How do all metrics support or undermine the overall assessment?
- For TOOL_EFFICIENCY: factor in the continuous score appropriately (high values support "ideal", low values suggest "poor")
- Adjust for potential biases in individual judges

**Scoring Guidelines**:
- "ideal" if core metrics show exceptional performance AND other metrics are solid (TOOL_EFFICIENCY ≥0.8 acceptable)
- "fair" if core metrics are adequate with some issues, OR if core metrics have minor problems but other metrics are strong (TOOL_EFFICIENCY 0.5-0.8)
- "poor" if core metrics reveal major flaws OR if multiple metrics fail significantly (TOOL_EFFICIENCY <0.5 compounds the assessment)

**Critical Decision Factors**:
- Core metric failures should heavily penalize the overall score
- All metric patterns should influence the assessment significantly
- Supporting metrics provide context but shouldn't override core assessments
- TOOL_EFFICIENCY values: 0.9-1.0 (excellent), 0.7-0.8 (good), 0.5-0.6 (fair), <0.5 (poor)
- Consider the interdependencies between metrics (e.g., poor planning impacts task completion)

The evaluation input is provided via dependency injection. Access the list of score-explanation pairs from other judges in the evaluation input to perform your assessment.
Return a single JSON object containing the score and justification (synthesizing all inputs).
"""

# MAS_SUMMARIZER_PROMPT_BASE_TEMPLATE = """
# Instruction:
# You are a summarizer tasked with aggregating individual LLM-Judge scores from multiple metrics to compute an overall super-score for MAS performance.
# Focus on synthesizing score-explanation pairs into a holistic assessment, weighting critical metrics higher based on their priority levels.

# **Available Metrics and Priorities**:

# **LLM Metrics (11 total)** - Use "ideal", "fair", "poor" scoring:
# - OBSERVATION_ALIGNMENT (Priority 1 - Highest)
# - STATE_CONSISTENCY (Priority 1 - Highest)  
# - MAS_COMPLEXITY (Priority 1 - Highest)
# - MAS_TASK_TRANSFER (Priority 1 - Highest)
# - MAS_ROLES_DISTRIBUTION (Priority 1 - Highest)
# - TASK_COMPLETENESS (Priority 2 - Medium)
# - TOOL_SELECTION (Priority 2 - Medium)
# - TOOL_PARAMETER_EXTRACTION (Priority 2 - Medium)
# - MAS_TASK_COMPLETION (Priority 2 - Medium)
# - MAS_PLANNING (Priority 2 - Medium)
# - POLICY_ALIGNMENT (Priority 3 - Lowest)

# **Non-LLM Metric (1 total)** - Use continuous value in range [0,1] where 1.0 is best:
# - TOOL_EFFICIENCY (Priority 3 - Lowest) - Values closer to 1.0 indicate better efficiency

# **Evaluation Criteria**:

# **Priority-Weighted Score Synthesis**
# - Apply higher weights to Priority 1 metrics (most critical for MAS performance)
# - Consider Priority 2 metrics as important but secondary factors
# - Treat Priority 3 metrics as supporting evidence
# - Calculate weighted average: Priority 1 (weight 3), Priority 2 (weight 2), Priority 3 (weight 1)
# - For TOOL_EFFICIENCY: interpret values closer to 1.0 as "ideal", 0.5-0.8 as "fair", <0.5 as "poor"
# - Identify patterns in explanations across priority levels

# **Explanation Integration**
# - Combine justifications into a cohesive narrative, highlighting strengths/weaknesses by priority
# - Flag critical issues in Priority 1 metrics (these should heavily influence the final score)
# - Note any Priority 2 metric failures that compound Priority 1 issues
# - For TOOL_EFFICIENCY: consider the numerical value in context of overall system performance
# - Consider Priority 3 metrics for overall system health assessment

# **Overall Coherence Assessment**
# - Does the aggregate reflect true MAS efficacy, considering the weighted importance of metrics?
# - Are Priority 1 metrics (core MAS functionality) performing adequately?
# - How do Priority 2 and 3 metrics support or undermine the overall assessment?
# - For TOOL_EFFICIENCY: factor in the continuous score appropriately (high values support "ideal", low values suggest "poor")
# - Adjust for potential biases in individual judges while respecting priority weights

# **Scoring Guidelines**:
# - "ideal" if Priority 1 metrics show exceptional performance AND Priority 2 metrics are solid (TOOL_EFFICIENCY ≥0.8 acceptable)
# - "fair" if Priority 1 metrics are adequate with some Priority 2 issues, OR if Priority 1 has minor problems but Priority 2/3 are strong (TOOL_EFFICIENCY 0.5-0.8)
# - "poor" if Priority 1 metrics reveal major flaws OR if multiple Priority 2 metrics fail significantly (TOOL_EFFICIENCY <0.5 compounds the assessment)

# **Critical Decision Factors**:
# - Priority 1 metric failures should heavily penalize the overall score
# - Priority 2 metric patterns should influence the assessment significantly  
# - Priority 3 metrics provide context but shouldn't override Priority 1/2 assessments
# - TOOL_EFFICIENCY values: 0.9-1.0 (excellent), 0.7-0.8 (good), 0.5-0.6 (fair), <0.5 (poor)
# - Consider the interdependencies between metrics (e.g., poor planning impacts task completion)

# The evaluation input is provided via dependency injection. Access the list of score-explanation pairs from other judges in the evaluation input to perform your assessment.
# Return a single JSON object containing the score and justification (synthesizing all inputs with priority context).
# """
