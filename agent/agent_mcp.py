
import asyncio
from typing import Annotated, Any, AsyncGenerator, Generator, Optional, Sequence, TypedDict, Union

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from langchain_core.messages.tool import ToolMessage
import json

nest_asyncio.apply()
############################################
## Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
GENIE_SPACE_ID = "01f101086a1711319ead12a273bb07f9"
VECTOR_SEARCH_SCHEMA = "mkr_gcp_sandbox_euw3/default"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = """
You are a helpful assistant for our sports data customers.
Use the vector search, which containd product documentation, to answer questions our product.
Use the Genie to answer data related questions.
If you don't know the answer, say 'I don't know', don't make up answers.
"""


workspace_client = WorkspaceClient()
host = workspace_client.config.host


## There are three connection types:
## 1. Managed MCP servers — fully managed by Databricks
## 2. External MCP servers — hosted outside Databricks but proxied through a
##    Managed MCP server proxy
## 3. Custom MCP servers — MCP servers hosted as Databricks Apps
##
###############################################################################

def create_tools():
    # Import OBO credentials inside the function to ensure it's available at runtime
    try:
        from databricks_ai_bridge import ModelServingUserCredentials
        obo_workspace_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
    except Exception as e:
        print(f"Warning: Could not create OBO workspace client: {e}")
        # Fallback to default workspace client if OBO fails
        obo_workspace_client = workspace_client

    databricks_mcp_client = DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="obo_vs_client",
                url=f"{host}/api/2.0/mcp/vector-search/{VECTOR_SEARCH_SCHEMA}",
                workspace_client=obo_workspace_client
            ),
            DatabricksMCPServer(
                name="obo_genie_client",
                url=f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}",
                workspace_client=obo_workspace_client
            )
        ]
    )
    return databricks_mcp_client.get_tools()



# The state for the agent workflow, including the conversation and any custom data
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: LanguageModelLike,
    system_prompt: Optional[str] = None,
):
    tools = asyncio.run(create_tools())
    model = model.bind_tools(tools)  # Bind tools to the model

    # Function to check if agent should continue or finish based on last message
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If function (tool) calls are present, continue; otherwise, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    # Preprocess: optionally prepend a system prompt to the conversation history
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | model  # Chain the preprocessor and the model

    # The function to invoke the model within the workflow
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)  # Create the agent's state machine

    workflow.add_node("agent", RunnableLambda(call_model))  # Agent node (LLM)
    workflow.add_node("tools", ToolNode(tools))  # Tools node

    workflow.set_entry_point("agent")  # Start at agent node
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",  # If the model requests a tool call, move to tools node
            "end": END,  # Otherwise, end the workflow
        },
    )
    workflow.add_edge("tools", "agent")  # After tools are called, return to agent node

    # Compile and return the tool-calling agent workflow
    return workflow.compile()


# ResponsesAgent class to wrap the compiled agent and make it compatible with Mosaic AI Responses API
class LangGraphResponsesAgent(ResponsesAgent):
    # def __init__(self, agent):
    #     self.agent = agent

    # Make a prediction (single-step) for the agent
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done" or event.type == "error"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    async def _predict_stream_async(
        self,
        request: ResponsesAgentRequest,
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        agent = create_tool_calling_agent(model=llm, system_prompt=system_prompt)
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        # Stream events from the agent graph
        async for event in agent.astream(
            {"messages": cc_msgs}, stream_mode=["updates", "messages"]
        ):
            if event[0] == "updates":
                # Stream updated messages from the workflow nodes
                for node_data in event[1].values():
                    if len(node_data.get("messages", [])) > 0:
                        all_messages = []
                        for msg in node_data["messages"]:
                            if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                                msg.content = json.dumps(msg.content)
                            all_messages.append(msg)
                        for item in output_to_responses_items_stream(all_messages):
                            yield item
            elif event[0] == "messages":
                # Stream generated text message chunks
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except:
                    pass

    # Stream predictions for the agent, yielding output as it's generated
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        agen = self._predict_stream_async(request)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        ait = agen.__aiter__()

        while True:
            try:
                item = loop.run_until_complete(ait.__anext__())
            except StopAsyncIteration:
                break
            else:
                yield item


# Initialize the entire agent, including MCP tools and workflow
def initialize_agent():
    return LangGraphResponsesAgent()


mlflow.langchain.autolog()
AGENT = initialize_agent()
mlflow.models.set_model(AGENT)
