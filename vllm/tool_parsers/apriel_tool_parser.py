# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Optional, Union

import partial_json_parser
import regex as re

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall, DeltaMessage, DeltaToolCall,
    ExtractedToolCallInformation, FunctionCall, ToolCall)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

logger = init_logger(__name__)


class AprielToolParser(ToolParser):

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # --- State Management for Streaming ---
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []
        # NEW: State flag to enforce content/tool call separation
        self.in_tool_call_phase: bool = False

        # --- Markers and Patterns ---
        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"
        self.answer_tool_calls_pattern = re.compile(
            r"<tool_calls>([\s\S]*?)</tool_calls>", re.DOTALL)

    # --- RESTORED Non-Streaming Methods ---

    def preprocess_model_output(
            self, model_output: str) -> tuple[Optional[str], Optional[str]]:
        def is_valid_json(s: str) -> bool:
            try:
                json.loads(s)
                return True
            except json.JSONDecodeError:
                return False

        # Find all tool calls
        for match in self.answer_tool_calls_pattern.finditer(model_output):
            start, end = match.span()
            tool_calls_content = match.group(1).strip()

            # Check if tool call is inside a final response region
            if "[BEGIN FINAL RESPONSE]" in model_output and "[END FINAL RESPONSE]" in model_output:
                response_regions = [
                    (m.start(), m.end()) for m in re.finditer(
                        r"\[BEGIN FINAL RESPONSE\](.*?)\[END FINAL RESPONSE\]",
                        model_output, flags=re.DOTALL
                    )
                ]
                in_response = any(start > t_start and end < t_end for t_start, t_end in response_regions)
                if not in_response:
                    continue

            # Validate JSON and return
            if is_valid_json(tool_calls_content):
                content = model_output[:start]
                return None, tool_calls_content

        # Fallback if no valid tool call found
        return model_output, None

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model output. (User's original logic)
        """
        try:
            content, potential_tool_calls = self.preprocess_model_output(
                model_output)

            if not potential_tool_calls:
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=content)

            tool_calls_data = json.loads(potential_tool_calls)
            if not isinstance(tool_calls_data, list):
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=content or model_output,
                )

            tool_calls: list[ToolCall] = []
            for call in tool_calls_data:
                if (not isinstance(call, dict) or "name" not in call
                        or "arguments" not in call):
                    continue

                tool_calls.append(ToolCall(
                    id=f"call_{random_uuid()}",
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=(json.dumps(call["arguments"]) if isinstance(
                            call["arguments"], dict) else call["arguments"]),
                    ),
                ))

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content and content.strip() else None,
            )

        except Exception:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Extracts tool calls for streaming, ensuring a clean separation
        between the content and tool_calls phases.
        """
        # If we have already transitioned to parsing tool calls, do not generate any more content.
        if self.in_tool_call_phase:
            return self._parse_streaming_tool_calls(current_text)

        # Check if the tool call marker has appeared for the first time.
        marker_pos = current_text.find(self.tool_calls_start_token)
        if marker_pos == -1:
            # We are still in the content phase.
            return DeltaMessage(content=delta_text)
        else:
            # This is the transition point from content to tool calls.
            self.in_tool_call_phase = True

            # Send the final chunk of content that appeared before the marker.
            content_before_marker = current_text[:marker_pos]
            final_content_chunk = content_before_marker[len(previous_text):]

            if final_content_chunk:
                return DeltaMessage(content=final_content_chunk)

            # If the marker appeared right after previous_text, there's no new content to send.
            # We can immediately proceed to parse the tools from the current chunk.
            return self._parse_streaming_tool_calls(current_text)

    def _parse_streaming_tool_calls(
            self, current_text: str) -> Union[DeltaMessage, None]:
        """Helper function to parse tool calls once we are in the tool call phase."""
        try:
            parsable_text = current_text.split(self.tool_calls_start_token, 1)[1]
            if self.tool_calls_end_token in parsable_text:
                parsable_text = parsable_text.split(self.tool_calls_end_token, 1)[0]
        except IndexError:
            return None

        try:
            tool_call_arr: list[dict] = partial_json_parser.loads(parsable_text)
            delta: Optional[DeltaMessage] = None

            if len(tool_call_arr) > self.current_tool_id + 1:
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            if self.current_tool_id < 0 or self.current_tool_id >= len(tool_call_arr):
                return None

            current_tool_call = tool_call_arr[self.current_tool_id]

            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            id=f"call_{random_uuid()}",
                            type="function",
                            function=DeltaFunctionCall(name=function_name))
                    ])
                    self.current_tool_name_sent = True
            else:
                cur_arguments = current_tool_call.get("arguments")
                if cur_arguments is not None:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    streamed_so_far = self.streamed_args_for_tool[self.current_tool_id]

                    if cur_args_json.startswith(streamed_so_far):
                        argument_diff = cur_args_json[len(streamed_so_far):]
                        if argument_diff:
                            delta = DeltaMessage(tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff))
                            ])
                            self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta

        except partial_json_parser.exceptions.MalformedJSON:
            return None
        except Exception:
            logger.exception("Error handling streaming tool call.")
            return None
