"""Example trace parser for Langfuse format."""

from __future__ import annotations

import ast
import json
import logging

from dotenv import load_dotenv
from langfuse.api.resources.commons.types.observations_view import \
    ObservationsView

# Configure logging
logger = logging.getLogger(__name__)

from maseval.models import (AgentResponse, AgentState,  # ToolDefinition,
                            DialogueMessage, EvaluationInput, InvalidToolCall,
                            MessageRole, StateType, ToolCall)



def parse_langfuse_task(trace: ObservationsView, task_id: str) -> EvaluationInput:
    """Parse one task of a Langfuse trace into our EvaluationInput format.

    Args:
        trace_data: Raw trace data from Langfuse
        task_id: ID for GAIA task

    Returns:
        EvaluationInput: Parsed evaluation input
    """

    observations = trace.observations

    # Get parent_id and agents list for single task from observations
    agents, parent_id = _get_single_task_info(observations, task_id=task_id)

    # Parse dialogue history from observations
    dialogue_history = _parse_dialogue_from_observations(
        observations, used_agents=agents, parent_id=parent_id
    )

    # Parse agent responses and states from observations
    agent_responses, agent_states = _parse_observations(
        observations, used_agents=agents, parent_id=parent_id
    )

    # # Extract available tools (if any)
    # available_tools = _extract_tool_definitions(observations)

    return EvaluationInput(
        trace_id=trace.id,
        # session_id=trace.get("sessionId"), # didn't find in trace
        environment=trace.environment,
        dialogue_history=dialogue_history,
        agent_responses=agent_responses,
        agent_states=agent_states,
        # available_tools=available_tools,
        metadata={
            "project_id": trace.projectId,
            "latency": trace.latency,
            "created_at": trace.createdAt,
        },
    )


def extract_all_tasks(trace: ObservationsView) -> list[str]:
    tasks = []
    for obs in trace.observations:
        if obs.type == "SPAN" and obs.name == "single_task":
            tasks.append(obs.input["args"][0]["task_id"])

    return tasks


def _get_single_task_info(
    observations: list[ObservationsView], task_id: str
) -> tuple[list[str], str]:
    """
    Extract information about a single task from observations.

    Args:
        observations: List of observation objects to search through
        task_id: The ID of the task to find information for

    Returns:
        tuple: A tuple containing:
            - list[str]: Names of agents used for the task
            - str: The parent ID of the task observation
    """
    used_agents = []
    parent_id = None
    has_output = False

    for obs in observations:
        try:
            # Check if this is a single_task observation
            if not (
                hasattr(obs, "type")
                and hasattr(obs, "name")
                and obs.type == "SPAN"
                and obs.name == "single_task"
            ):
                continue

            if obs.output["task_id"] == task_id:
                has_output = True
                parent_id = obs.id

                # Extract agent names from pool
                if "pool" not in obs.output:
                    logger.warning(
                        f"Observation output missing 'pool' field for task {task_id}"
                    )
                    continue

                if not isinstance(obs.output["pool"], list):
                    logger.warning(
                        f"Pool is not a list for task {task_id}: {obs.output['pool']}"
                    )
                    continue

                for agent in obs.output["pool"]:
                    try:
                        if not isinstance(agent, dict):
                            logger.warning(f"Agent entry is not a dictionary: {agent}")
                            continue

                        if "name" not in agent:
                            logger.warning(f"Agent entry missing 'name' field: {agent}")
                            continue

                        agent_name = agent["name"]
                        if not isinstance(agent_name, str) or not agent_name.strip():
                            logger.warning(f"Invalid agent name: {agent_name}")
                            continue

                        used_agents.append(agent_name.strip())

                    except (KeyError, TypeError, AttributeError) as e:
                        logger.error(f"Error processing agent entry {agent}: {e}")
                        continue

        except (AttributeError, TypeError) as e:
            logger.error(f"Error processing observation {obs}: {e}")
            continue

    # Validate results for the specific task
    if not has_output:
        raise ValueError(
            f"Task with ID '{task_id}' has no output. Status message: {obs.status_message}"
        )

    if parent_id is None:
        raise ValueError(f"Parent ID not found for task '{task_id}'")

    if not used_agents:
        logger.warning(f"No agents found for task '{task_id}'")

    return used_agents, parent_id


def _parse_dialogue_from_observations(
    observations: list[ObservationsView], used_agents: list[str], parent_id: str
) -> list[DialogueMessage]:
    """
    Parse dialogue history from trace input/output.

    Args:
        observations: List of observation objects to search through
        used_agents: List of agent names that were used for the task
        parent_id: The parent observation ID to filter observations by

    Returns:
        list[DialogueMessage]: List of dialogue messages parsed from the observations
    """
    dialogue = []

    for agent in used_agents:
        for obs in observations:
            if obs.type == "SPAN" and obs.name == f"{agent} run":
                if parent_id == obs.parent_observation_id:
                    try:
                        agent_messages = json.loads(
                            obs.metadata["attributes"]["pydantic_ai.all_messages"]
                        )

                        for message in agent_messages:
                            try:
                                if message["role"] == "system":
                                    role = _map_role("system")
                                    content = message["parts"][0]["content"]
                                    dialogue.append(
                                        DialogueMessage(role=role, content=content)
                                    )

                                elif message["role"] in ["user", "assistant"]:
                                    for part in message["parts"]:
                                        if part["type"] not in [
                                            "tool_call_response",
                                            "tool_call",
                                        ]:
                                            role = _map_role(message["role"])
                                            content = part["content"]
                                            dialogue.append(
                                                DialogueMessage(
                                                    role=role, content=content
                                                )
                                            )

                                else:
                                    print("Unknown role.")

                            except (KeyError, IndexError, TypeError) as e:
                                print(f"Error parsing message: {e}")
                                continue

                    except (KeyError, json.JSONDecodeError, TypeError) as e:
                        print(f"Error parsing agent messages for {agent}: {e}")
                        continue

    return dialogue


def _parse_observations(
    observations: list[ObservationsView], used_agents: list[str], parent_id: str
) -> tuple[list[AgentResponse], list[AgentState]]:
    """
    Parse agent responses and states from observations.

    Args:
        observations: List of observation objects to search through
        used_agents: List of agent names that were used for the task
        parent_id: The parent observation ID to filter observations by

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
                obs.type == "SPAN"
                and obs.name == f"{agent} run"
                and parent_id == obs.parent_observation_id
            ):
                try:
                    start_time = obs.start_time
                    agent_messages = json.loads(
                        obs.metadata["attributes"]["pydantic_ai.all_messages"]
                    )

                    for message in agent_messages:
                        try:
                            role = message["role"]
                            parts = message.get("parts", [])

                            if role == "assistant":
                                for part in parts:
                                    if part["type"] == "text":
                                        response = AgentResponse(
                                            response_id=f"{agent}_{counter_responses}",
                                            content=part["content"],
                                            timestamp=start_time,
                                        )
                                        agent_responses.append(response)
                                        counter_responses += 1

                                        state = AgentState(
                                            state_id=f"{agent}_{counter_states}",
                                            type=StateType.ASSISTANT,
                                            content=part["content"],
                                            timestamp=start_time,
                                        )
                                        agent_states.append(state)
                                        counter_states += 1

                                    elif part["type"] == "tool_call":
                                        tool_call, invalid_tool_call = (
                                            _process_tool_call(part, observations)
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
                                            content=part["arguments"],
                                            tool_call=tool_call_obj,
                                            timestamp=start_time,
                                        )
                                        agent_states.append(state)
                                        counter_states += 1

                            elif role == "user":
                                for part in parts:
                                    if part["type"] == "text":
                                        state = AgentState(
                                            state_id=f"{agent}_{counter_states}",
                                            type=StateType.USER,
                                            content=part["content"],
                                            timestamp=start_time,
                                        )
                                        agent_states.append(state)
                                        counter_states += 1

                                    elif part["type"] == "tool_call_response":
                                        state = AgentState(
                                            state_id=f"{agent}_{counter_states}",
                                            type=StateType.TOOL_CALL_RESPONSE,
                                            content=part["result"],
                                            timestamp=start_time,
                                        )
                                        agent_states.append(state)
                                        counter_states += 1

                            elif role == "system":
                                if parts:
                                    state = AgentState(
                                        state_id=f"{agent}_{counter_states}",
                                        type=StateType.SYSTEM,
                                        content=parts[0]["content"],
                                        timestamp=start_time,
                                    )
                                    agent_states.append(state)
                                    counter_states += 1

                        except (KeyError, IndexError, TypeError, ValueError) as e:
                            print(f"Error parsing message for {agent}: {e}")
                            continue

                except (KeyError, json.JSONDecodeError, TypeError) as e:
                    print(f"Error parsing agent messages for {agent}: {e}")
                    continue

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
            if (
                obs.type == "SPAN"
                and obs.name == f'running tool: {part["name"]}'
                and obs.metadata["attributes"]["gen_ai.tool.call.id"] == part["id"]
            ):

                tool_call = ToolCall(
                    id=part["id"],
                    name=part["name"],
                    parameters=ast.literal_eval(part["arguments"]),
                    result=obs.metadata["attributes"]["tool_response"],
                )

                if obs.status_message is not None:
                    invalid_tool_call = InvalidToolCall(
                        id=part["id"],
                        name=part["name"],
                        parameters=ast.literal_eval(part["arguments"]),
                        status_message=obs.status_message,
                        result=obs.metadata["attributes"]["tool_response"],
                    )
                    return None, invalid_tool_call
                else:
                    return tool_call, None

    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing tool call: {e}")
        return None, None


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

    trace_data = lf.api.trace.get("40fb8634142c9c2e93dd945660d82584")

    # task_ids = ['ec09fa32-d03f-4bf8-84b0-1f16922c3ae4', 'e1fc63a2-da7a-432f-be78-7c4a95598703', '8e867cd7-cff9-4e6c-867a-ff5ddc2550be']
    task_ids = []

    if task_ids == []:
        task_ids = extract_all_tasks(trace_data)

    for task in task_ids:
        eval_input = parse_langfuse_task(trace_data, task_id=task)
        print(f"Parsed trace with {len(eval_input.dialogue_history)} dialogue messages")
        print(f"Found {len(eval_input.agent_responses)} agent responses")
        print(f"Found {len(eval_input.agent_states)} agent states")
        # print(f"Found {len(eval_input.available_tools)} tools")

        # Print some details
        for i, msg in enumerate(eval_input.dialogue_history):
            print(f"Message {i}: {msg.role} - {msg.content[:100]}...")

        for i, state in enumerate(eval_input.agent_states, start=1):
            print(f"State {i}: {state.type} - {state.state_id}")
