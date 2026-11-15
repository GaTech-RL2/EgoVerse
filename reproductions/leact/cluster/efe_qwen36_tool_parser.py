"""Let vLLM 0.19 Qwen3Coder read Responses function parameter schemas."""
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


@ToolParserManager.register_module('qwen3_coder_responses')
class Qwen3CoderResponsesToolParser(Qwen3CoderToolParser):
    def _get_arguments_config(self, func_name, tools):
        # Retain upstream behavior for the nested Chat Completions schema.
        found = super()._get_arguments_config(func_name, tools)
        if found:
            return found
        for tool in tools or []:
            if isinstance(tool, dict):
                record = tool
            elif hasattr(tool, 'model_dump'):
                record = tool.model_dump()
            else:
                record = vars(tool)
            if record.get('type') != 'function' or record.get('name') != func_name:
                continue
            parameters = record.get('parameters')
            if isinstance(parameters, dict):
                return parameters.get('properties', parameters)
        return {}
