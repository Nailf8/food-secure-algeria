from typing import Any, Dict, List, Optional
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.schema import NodeWithScore
from llama_index.llms.databricks import Databricks


DEFAULT_CONTEXT_PROMPT = (
    "Here is some context that may be relevant:\n"
    "-----\n"
    "{node_context}\n"
    "-----\n"
    "Please write a response to the following question, using the above context:\n"
    "{query_str}\n"
)


class ChatPipeline():
    def __init__(self, verbose=True):
        """
        Initializes the QueryPipelineBuilder with verbosity settings.
        
        :param verbose: Whether to enable verbose output for the pipeline.
        """
        self.verbose = verbose
        self.pipeline = QueryPipeline(verbose=verbose)

    def add_components(self, input_component, rewrite_template, llm, retriever, reranker, response_component):
        """
        Adds modules to the pipeline and defines the links between them.
        
        :param input_component: The input component of the pipeline.
        :param rewrite_template: The component responsible for rewriting queries.
        :param llm: The language model component.
        :param retriever: The retrieval component.
        :param reranker: The reranker component.
        :param response_component: The component that synthesizes the final response.
        """
        # Add modules to the pipeline
        self.pipeline.add_modules(
            {
                "input": input_component,
                "rewrite_template": rewrite_template,
                "llm": llm,
                "retriever": retriever,
                "reranker": reranker,
                "response_component": response_component,
            }
        )
        # Define links between modules
        self._define_links()

    def _define_links(self):
        """
        Defines the links between the modules in the pipeline.
        """
        p = self.pipeline
        # Linking input to rewrite_template
        p.add_link("input", "rewrite_template", src_key="query_str", dest_key="query_str")
        p.add_link("input", "rewrite_template", src_key="chat_history_str", dest_key="chat_history_str")
        
        # Linking rewrite_template to LLM, and LLM to retriever
        p.add_link("rewrite_template", "llm")
        p.add_link("llm", "retriever")
        
        # Linking retriever to reranker, and LLM to reranker
        p.add_link("retriever", "reranker", dest_key="nodes")
        p.add_link("llm", "reranker", dest_key="query_str")
        
        # Linking reranker to response_component
        p.add_link("reranker", "response_component", dest_key="nodes")
        
        # Linking input to response_component
        p.add_link("input", "response_component", src_key="query_str", dest_key="query_str")
        p.add_link("input", "response_component", src_key="chat_history", dest_key="chat_history")

    def _update_memory(self, pipeline_memory, user_message, response):
        # Store the user's message in memory
        user_msg = ChatMessage(role="user", content=user_message)
        pipeline_memory.put(user_msg)
        
        # Store the final response in memory
        pipeline_memory.put(response)

    def _run(self, msg, pipeline_memory):
        # get memory
        chat_history = pipeline_memory.get()

        # prepare inputs
        chat_history_str = "\n".join([str(x) for x in chat_history])

        # run pipeline
        response = self.pipeline.run(
            query_str=msg,
            chat_history=chat_history,
            chat_history_str=chat_history_str,
        )
        # update memory
        self._update_memory(pipeline_memory, msg, response.message)
        return response.message

    def _execute(self, msg, pipeline_memory):
        # get memory
        chat_history = pipeline_memory.get()

        # prepare inputs
        chat_history_str = "\n".join([str(x) for x in chat_history])

        run_state = self.pipeline.get_run_state(
        query_str=msg,
        chat_history=chat_history,
        chat_history_str=chat_history_str,
    )
        next_module_keys = self.pipeline.get_next_module_keys(run_state)

        while True:
            for module_key in next_module_keys:
                # get the module and input
                module = run_state.module_dict[module_key]
                module_input = run_state.all_module_inputs[module_key]

                # run the module
                output_dict = module.run_component(**module_input)

                # process the output
                self.pipeline.process_component_output(
                    output_dict,
                    module_key,
                    run_state,
                )

            # get the next module keys
            next_module_keys = self.pipeline.get_next_module_keys(
                run_state,
            )

            # if no more modules to run, break
            if not next_module_keys:
                run_state.result_outputs[module_key] = output_dict    
                break
        
        response = output_dict["response"].message.content
        self._update_memory(pipeline_memory, msg, response)
        # Return the final response
        return response

    def get_pipeline(self):
        """
        Returns the constructed query pipeline.
        
        :return: Configured QueryPipeline object.
        """
        return self.pipeline


class ResponseWithChatHistory(CustomQueryComponent):
    llm: Databricks = Field(..., description="Llama 70B endpoint")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )
    def _validate_component_inputs(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # NOTE: this is OPTIONAL but we show you where to do validation as an example
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        # NOTE: These are required inputs. If you have optional inputs please override
        # `optional_input_keys_dict`
        return {"chat_history", "nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response"}

    def _prepare_context(
        self,
        chat_history: List[ChatMessage],
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )
        user_message = ChatMessage(role="user", content=formatted_context)

        chat_history.append(user_message)

        if self.system_prompt is not None:
            chat_history = [
                ChatMessage(role="system", content=self.system_prompt)
            ] + chat_history

        return chat_history

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            chat_history, nodes, query_str
        )

        response = self.llm.chat(prepared_context)

        return {"response": response}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the component asynchronously."""
        # NOTE: Optional, but async LLM calls are easy to implement
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            chat_history, nodes, query_str
        )

        response = await self.llm.achat(prepared_context)

        return {"response": response}

