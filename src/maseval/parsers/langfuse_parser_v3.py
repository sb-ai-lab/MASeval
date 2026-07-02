"""Example trace parser for Langfuse format."""

from __future__ import annotations

import ast
import json
import logging
import re

from dotenv import load_dotenv
from langfuse.api.resources.commons.types.observations_view import \
    ObservationsView
from rich import print

# Configure logging
logger = logging.getLogger(__name__)

from maseval.models import (AgentDescription, AgentError, AgentResponse,
                            AgentsPool, AgentState, DialogueMessage,
                            EvaluationInput, InvalidToolCall, Latency,
                            MessageRole, StateType, TokensInfo, ToolCall,
                            ToolDefinition, ToolsInfo)


def parse_langfuse_task(trace: ObservationsView) -> EvaluationInput:
    """Parse one task of a Langfuse trace into our EvaluationInput format.

    Args:
        trace_data: Raw trace data from Langfuse

    Returns:
        EvaluationInput: Parsed evaluation input
    """

    observations = trace.observations
    # def decode_unicode(s):
    #     if isinstance(s, str):
    #         return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)
    #     elif isinstance(s, dict):
    #         return {k: decode_unicode(v) for k, v in s.items()}
    #     return s

    # decoded_observations = [
    #     type(obs).parse_obj({
    #         **obs.dict(),
    #         'input': decode_unicode(obs.input) if obs.input else obs.input,
    #         'output': decode_unicode(obs.output) if obs.output else obs.output,
    #         'metadata': decode_unicode(obs.metadata) if obs.metadata else obs.metadata
    #     })
    #     for obs in observations
    # ]

    # Get agents list for single task from observations
    agents = _get_single_task_info(observations)

    # If agents are found, parse the task
    if agents != []:
        # Parse dialogue history from observations
        dialogue_history = _parse_dialogue_from_observations(
            observations, used_agents=agents
        )

        # Parse agent responses and states from observations
        agent_responses, agent_states = _parse_observations(
            observations, used_agents=agents
        )

        # Extract agent tools info
        agents_tools_info = _extract_tool_info(
            observations, agent_states, used_agents=agents
        )

        # Extract agent tokens and latency info
        agents_tokens_info, agents_latency_info = _extract_agents_info(
            observations, used_agents=agents
        )

        # Extract agents pool info
        agents_pool = _get_agents_pool(observations)

        return EvaluationInput(
            user_query=trace.output.get("question"),
            trace_id=trace.id,
            # session_id=trace.get("sessionId"), # didn't find in trace
            environment=trace.environment,
            dialogue_history=dialogue_history,
            agent_responses=agent_responses,
            agent_states=agent_states,
            agents_tools_info=agents_tools_info,
            agents_tokens_info=agents_tokens_info,
            agents_latency_info=agents_latency_info,
            agents_pool=agents_pool,
            metadata={
                "project_id": trace.projectId,
                "latency": trace.latency,
                "created_at": trace.createdAt,
            },
        )

    # If agents are not found, parse the errors
    else:
        agents_errors = _get_agents_errors(observations)

        return EvaluationInput(
            trace_id=trace.id,
            environment=trace.environment,
            agents_errors=agents_errors,
            metadata={
                "project_id": trace.projectId,
                "latency": trace.latency,
                "created_at": trace.createdAt,
            },
        )


def _get_single_task_info(observations: list[ObservationsView]) -> list[str]:
    """
    Extract information about a single task from observations.

    Args:
        observations: List of observation objects to search through
        task_id: The ID of the task to find information for

    Returns:
        list[str]: Names of agents used for the task
    """
    used_agents = []
    has_output = False

    for obs in observations:
        # Check if this is a single_task observation
        obs_name = getattr(obs, "name", None)

        if not (
            getattr(obs, "type", None) == "SPAN"
            and isinstance(obs_name, str)
            and obs_name
            and "_task_" in obs_name
        ):
            continue

        output = getattr(obs, "output", None)
        if output is not None:
            has_output = True

            pool = {}
            if isinstance(output, dict):
                pool = output.get("pool", {})
            else:
                logger.warning("Observation output is not a dict for task.")
                continue

            if not isinstance(pool, list):
                logger.warning(f"Pool is not a list for task: {pool}")
                continue

            for agent in pool:
                if not isinstance(agent, dict):
                    logger.warning(f"Agent entry is not a dictionary: {agent}")
                    continue

                agent_name = agent.get("name")
                if not isinstance(agent_name, str) or not agent_name.strip():
                    logger.warning(f"Invalid agent name: {agent_name}")
                    continue

                used_agents.append(agent_name.strip())

        if not has_output:
            logger.warning(f"Task has no output. Status message: {obs.status_message}")

    return used_agents


def _parse_dialogue_from_observations(
    observations: list[ObservationsView], used_agents: list[str]
) -> list[DialogueMessage]:
    """
    Parse dialogue history from trace input/output.

    Args:
        observations: List of observation objects to search through
        used_agents: List of agent names that were used for the task

    Returns:
        list[DialogueMessage]: List of dialogue messages parsed from the observations
    """
    dialogue = []

    for agent in used_agents:
        for obs in observations:
            if (
                getattr(obs, "type", None) == "SPAN"
                and getattr(obs, "name", None) == f"{agent} run"
            ):
                metadata = getattr(obs, "metadata", {}) or {}
                attributes = (
                    metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
                )
                raw_messages = attributes.get("pydantic_ai.all_messages")
                if not raw_messages:
                    continue
                try:
                    agent_messages = json.loads(raw_messages)
                except (TypeError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing agent messages for {agent}: {e}")
                    continue

                for message in agent_messages:
                    role_value = (message or {}).get("role")
                    parts = (message or {}).get("parts", [])
                    if role_value == "system":
                        first_part = parts[0] if parts else {}
                        content = (first_part or {}).get("content")
                        if content is not None:
                            role = _map_role("system")
                            dialogue.append(DialogueMessage(role=role, content=content))
                    elif role_value in ["user", "assistant"]:
                        for part in parts:
                            part_type = (part or {}).get("type")
                            if part_type not in ["tool_call_response", "tool_call"]:
                                content = (part or {}).get("content")
                                if content is not None:
                                    role = _map_role(role_value)
                                    dialogue.append(
                                        DialogueMessage(role=role, content=content)
                                    )
                    else:
                        logger.warning("Unknown role.")

    return dialogue


def _parse_observations(
    observations: list[ObservationsView], used_agents: list[str]
) -> tuple[list[AgentResponse], list[AgentState]]:
    """
    Parse agent responses and states from observations.

    Args:
        observations: List of observation objects to search through
        used_agents: List of agent names that were used for the task

    Returns:
        tuple: A tuple containing:
            - list[AgentResponse]: List of agent responses parsed from observations
            - list[AgentState]: List of agent states parsed from observations
    """
    agent_responses = []
    agent_states = []

    for agent in used_agents:
        counter_responses = 1
        counter_states = 1

        for obs in observations:
            if (
                getattr(obs, "type", None) == "SPAN"
                and getattr(obs, "name", None) == f"{agent} run"
            ):
                start_time = getattr(obs, "start_time", None)
                metadata = getattr(obs, "metadata", {}) or {}
                attributes = (
                    metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
                )
                raw_messages = attributes.get("pydantic_ai.all_messages")
                if not raw_messages:
                    continue
                try:
                    agent_messages = json.loads(raw_messages)
                except (TypeError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing agent messages for {agent}: {e}")
                    continue

                for message in agent_messages:
                    role = (message or {}).get("role")
                    parts = (message or {}).get("parts", [])

                    if role == "assistant":
                        for part in parts:
                            part_type = (part or {}).get("type")
                            if part_type == "text":
                                content = (part or {}).get("content")
                                if content is None:
                                    continue
                                response = AgentResponse(
                                    response_id=f"{agent}_{counter_responses}",
                                    content=content,
                                    timestamp=start_time,
                                )
                                agent_responses.append(response)
                                counter_responses += 1

                                state = AgentState(
                                    state_id=f"{agent}_{counter_states}",
                                    type=StateType.ASSISTANT,
                                    content=content,
                                    timestamp=start_time,
                                )
                                agent_states.append(state)
                                counter_states += 1

                            elif part_type == "tool_call":
                                tool_call, invalid_tool_call = _process_tool_call(
                                    part, observations
                                )

                                state_type = (
                                    StateType.INVALID_TOOL_CALL
                                    if invalid_tool_call
                                    else StateType.TOOL_CALL
                                )
                                tool_call_obj = (
                                    invalid_tool_call
                                    if invalid_tool_call
                                    else tool_call
                                )

                                state = AgentState(
                                    state_id=f"{agent}_{counter_states}",
                                    type=state_type,
                                    content=(part or {}).get("arguments"),
                                    tool_call=tool_call_obj,
                                    timestamp=start_time,
                                )
                                agent_states.append(state)
                                counter_states += 1

                    elif role == "user":
                        for part in parts:
                            part_type = (part or {}).get("type")
                            if part_type == "text":
                                content = (part or {}).get("content")
                                if content is None:
                                    continue
                                state = AgentState(
                                    state_id=f"{agent}_{counter_states}",
                                    type=StateType.USER,
                                    content=content,
                                    timestamp=start_time,
                                )
                                agent_states.append(state)
                                counter_states += 1

                            elif part_type == "tool_call_response":
                                result = (part or {}).get("result")
                                state = AgentState(
                                    state_id=f"{agent}_{counter_states}",
                                    type=StateType.TOOL_CALL_RESPONSE,
                                    content=result,
                                    timestamp=start_time,
                                )
                                agent_states.append(state)
                                counter_states += 1

                    elif role == "system":
                        if parts:
                            content = (
                                (parts[0] or {}).get("content")
                                if isinstance(parts[0], dict)
                                else None
                            )
                            if content is not None:
                                state = AgentState(
                                    state_id=f"{agent}_{counter_states}",
                                    type=StateType.SYSTEM,
                                    content=content,
                                    timestamp=start_time,
                                )
                                agent_states.append(state)
                                counter_states += 1

    return agent_responses, agent_states


def _process_tool_call(
    part: dict, observations: list[ObservationsView]
) -> tuple[ToolCall | None, InvalidToolCall | None]:
    """
    Process a tool call part and return the appropriate tool call objects.

    Args:
        part: The tool call part from the message
        observations: List of observations to find tool execution details

    Returns:
        tuple: A tuple containing (tool_call, invalid_tool_call)
    """
    try:
        for obs in observations:
            if getattr(obs, "type", None) != "SPAN":
                continue
            part_name = (part or {}).get("name")
            expected_name = f"running tool: {part_name}" if part_name else None
            if getattr(obs, "name", None) != expected_name:
                continue

            attributes = (getattr(obs, "metadata", {}) or {}).get("attributes", {})
            obs_call_id = attributes.get("gen_ai.tool.call.id")
            if obs_call_id != (part or {}).get("id"):
                continue

            arguments_value = (part or {}).get("arguments", "{}")
            try:
                parsed_args = (
                    ast.literal_eval(arguments_value)
                    if isinstance(arguments_value, str)
                    else arguments_value
                )
            except (ValueError, SyntaxError):
                parsed_args = arguments_value

            tool_response = attributes.get("tool_response")

            tool_call = ToolCall(
                id=(part or {}).get("id"),
                name=part_name,
                parameters=parsed_args,
                result=tool_response,
            )

            if getattr(obs, "status_message", None) is not None:
                invalid_tool_call = InvalidToolCall(
                    id=(part or {}).get("id"),
                    name=part_name,
                    parameters=parsed_args,
                    status_message=getattr(obs, "status_message", None),
                    result=tool_response,
                )
                return None, invalid_tool_call
            else:
                return tool_call, None

    except Exception as e:
        logger.error(f"Error processing tool call: {e}")
        return None, None


def _extract_tool_info(
    observations: list[ObservationsView],
    agent_states: list[AgentState],
    used_agents: list[str],
) -> list[ToolsInfo]:
    """
    Extract tools-related information for each used agent.

    Args:
        observations: List of trace observations to inspect for tool metadata and spans.
        agent_states: Parsed agent states, used to collect called tools per agent.
        used_agents: Agent names that participated in the task.

    Returns:
        list[ToolsInfo]: One entry per agent with called tools and available tool definitions.
    """
    agents_tools_info = []

    used_agents_ids = {}

    for agent in used_agents:
        for obs in observations:
            if obs.type == "SPAN" and obs.name == f"{agent} run":
                used_agents_ids[agent] = obs.id

    for agent in used_agents_ids.keys():
        tools_called = []
        available_tools = []

        for state in agent_states:
            if (
                state.type == StateType.TOOL_CALL
                or state.type == StateType.INVALID_TOOL_CALL
            ) and state.state_id.split("_")[0] == agent:
                tools_called.append(state.tool_call)

        for obs in observations:
            if (
                obs.type == "GENERATION"
                and obs.parent_observation_id == used_agents_ids[agent]
            ):
                attributes = (getattr(obs, "metadata", {}) or {}).get("attributes", {})
                raw_params = attributes.get("model_request_parameters")

                params = {}
                if isinstance(raw_params, str):
                    try:
                        params = json.loads(raw_params)
                    except Exception:
                        params = {}
                elif isinstance(raw_params, dict):
                    params = raw_params

                tools = params.get("function_tools") or []

                for tool in tools:
                    try:
                        tool_definition = ToolDefinition(
                            name=tool.get("name"),
                            description=tool.get("description", ""),
                            parameters=(
                                (tool.get("parameters_json_schema") or {}).get(
                                    "properties"
                                )
                            )
                            or {},
                            required_parameters=(
                                (tool.get("parameters_json_schema") or {}).get(
                                    "required"
                                )
                            )
                            or [],
                        )
                        available_tools.append(tool_definition)

                    except Exception:
                        continue

                break

        tool_info = ToolsInfo(
            agent_name=agent, tools_called=tools_called, available_tools=available_tools
        )
        agents_tools_info.append(tool_info)

    return agents_tools_info


def _extract_agents_info(
    observations: list[ObservationsView], used_agents: list[str]
) -> tuple[list[TokensInfo], list[Latency]]:
    """
    Aggregate token usage and latency per agent from Langfuse observations.

    Args:
        observations: Flat list of Langfuse observations for a single trace.
        used_agents: Ordered list of agent names that participated in the task.

    Returns:
        tuple[list[TokensInfo], list[Latency]]: Two lists aligned with agent order:
            - tokens list with input/output/total token counts per agent
            - latency list with total latency in seconds per agent (rounded to 2 decimals)

    Notes:
        - Only observations that are direct children of the agent "<agent> run" span
          are counted.
        - Token usage ignores spans named 'running 1 tool'.
    """
    agents_tokens_info = []
    agents_latency_info = []

    used_agents_ids = {}

    for agent in used_agents:
        for obs in observations:
            if obs.type == "SPAN" and obs.name == f"{agent} run":
                used_agents_ids[agent] = obs.id

    for agent in used_agents_ids.keys():
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        latency = 0

        parent_id = used_agents_ids.get(agent)

        for obs in observations:
            if parent_id is None:
                continue
            if obs.parent_observation_id == parent_id and obs.name != "running 1 tool":
                usage = getattr(obs, "usage_details", None) or {}
                if isinstance(usage, dict):
                    input_tokens += int(usage.get("input", 0) or 0)
                    output_tokens += int(usage.get("output", 0) or 0)
                    total_tokens += int(usage.get("total", 0) or 0)

            if obs.id == parent_id:
                latency += obs.latency / 1000

        tokens_info = TokensInfo(
            agent_name=agent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        agents_tokens_info.append(tokens_info)

        latency_info = Latency(agent_name=agent, latency=round(latency, 2))
        agents_latency_info.append(latency_info)

    return agents_tokens_info, agents_latency_info


def _get_agents_errors(observations: list[ObservationsView]) -> list[AgentError]:
    """
    Extract agent errors from observations.

    Args:
        observations: List of observation objects to search through

    Returns:
        list[AgentError]: List of agent errors found in the observations
    """
    agents_errors = []
    for obs in observations:
        if obs.type == "SPAN" and obs.name.endswith(" run"):
            agents_errors.append(
                AgentError(
                    agent_name=obs.name.split(" run")[0],
                    error_message=obs.status_message,
                )
            )

    return agents_errors


def _get_agents_pool(observations: list[ObservationsView]) -> list[AgentsPool]:
    """
    Extract agents pool information from observations.

    Args:
        observations: List of observation objects to search through

    Returns:
        list[AgentsPool]: List of agents pool information found in the observations
    """
    agents_descriptions = []

    for obs in observations:
        obs_name = getattr(obs, "name", None)
        if not (isinstance(obs_name, str) and "_task_" in obs_name):
            continue

        output = getattr(obs, "output", {}) or {}

        if isinstance(output, dict):
            for agent in output.get("pool", []):
                agents_descriptions.append(
                    AgentDescription(
                        agent_name=agent.get("name"),
                        instructions=agent.get("instructions"),
                    )
                )
            agents_pool = AgentsPool(
                agents=agents_descriptions, graph=output.get("graph") or output.get("mermaid_graph")
            )

    return agents_pool


def _map_role(role_str: str) -> MessageRole:
    """
    Map various role strings to our MessageRole enum.

    Args:
        role_str: The role string to map to a MessageRole enum value

    Returns:
        MessageRole: The corresponding MessageRole enum value, defaults to USER if no mapping found
    """
    role_str = role_str.lower()

    role_mapping = {
        "user": MessageRole.USER,
        "human": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
        "ai": MessageRole.ASSISTANT,
        "system": MessageRole.SYSTEM,
        "tool": MessageRole.TOOL,
    }

    return role_mapping.get(role_str, MessageRole.USER)


if __name__ == "__main__":
    load_dotenv()

    # Import here to avoid circular dependency
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from maseval import get_langfuse_download_client

    # Get Langfuse client for downloading traces
    # This uses LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
    lf = get_langfuse_download_client()

    trace_data = lf.api.trace.get(
        # "92cbd96e51159cee14b51258232e0b80"
        # "b7dbbc128b3f73e47814406b87c04342"
        "2458411fc6f99613108eb51777b5c8dc"
    )  # 63fb7e24f7d9505ed7f4cf5cdce1e61d

    print(type(trace_data))
    # Parse single task
    eval_input = parse_langfuse_task(trace_data)

    if eval_input.agents_errors == None:
        print(f"Parsed trace with {len(eval_input.dialogue_history)} dialogue messages")
        print(f"Found {len(eval_input.agent_responses)} agent responses")
        print(f"Found {len(eval_input.agent_states)} agent states")
        print(f"Found {len(eval_input.agents_tools_info)} agents")

        # Print some details
        for i, msg in enumerate(eval_input.dialogue_history):
            print(f"Message {i}: {msg.role} - {msg.content[:100]}...")

        for i, state in enumerate(eval_input.agent_responses, start=1):
            print(f"State {i}: {state.response_id} - {state.content}")

        for i, agent in enumerate(eval_input.agents_tokens_info, start=1):
            print(
                f"Agent-{i}: {agent.agent_name}, input: {agent.input_tokens}, output: {agent.output_tokens}, TOTAL: {agent.total_tokens}"
            )

        for i, agent in enumerate(eval_input.agents_latency_info, start=1):
            print(f"Agent-{i}: {agent.agent_name}, latency: {agent.latency}")

        print(
            f"Agents: {eval_input.agents_pool.agents}, graph: {eval_input.agents_pool.graph}"
        )
    else:
        print(f"Found {len(eval_input.agents_errors)} agents errors")
        for i, agent in enumerate(eval_input.agents_errors, start=1):
            print(f"Agent-{i}: {agent.agent_name}, error: {agent.error_message}")
