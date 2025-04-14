"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
import subprocess
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path, PosixPath
from typing import cast, get_args
import io

import httpx
import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from streamlit.delta_generator import DeltaGenerator

from computer_use_demo.loop import (
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult, ToolVersion

PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


@dataclass(kw_only=True, frozen=True)
class ModelConfig:
    tool_version: ToolVersion
    max_output_tokens: int
    default_output_tokens: int
    has_thinking: bool = False


SONNET_3_5_NEW = ModelConfig(
    tool_version="computer_use_20241022",
    max_output_tokens=1024 * 8,
    default_output_tokens=1024 * 4,
)

SONNET_3_7 = ModelConfig(
    tool_version="computer_use_20250124",
    max_output_tokens=128_000,
    default_output_tokens=1024 * 16,
    has_thinking=True,
)

MODEL_TO_MODEL_CONF: dict[str, ModelConfig] = {
    "claude-3-7-sonnet-20250219": SONNET_3_7,
}

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button in red */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"
INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"

# --- Logging Configuration ---
LOG_BASE_DIR = Path("./conversation_logs")


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"  # for logging context


def setup_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv(
            "ANTHROPIC_API_KEY", ""
        )
    if "provider" not in st.session_state:
        st.session_state.provider = (
            os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
        )
    if "provider_radio" not in st.session_state:
        st.session_state.provider_radio = st.session_state.provider
    if "model" not in st.session_state:
        _reset_model()
    if "auth_validated" not in st.session_state:
        st.session_state.auth_validated = False
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 3
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = load_from_storage("system_prompt") or ""
    if "hide_images" not in st.session_state:
        st.session_state.hide_images = False
    if "token_efficient_tools_beta" not in st.session_state:
        st.session_state.token_efficient_tools_beta = False
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False

    if "log_dir" not in st.session_state:
        now = datetime.now()
        session_timestamp = now.strftime("%Y%m%d_%H%M%S")
        st.session_state.log_dir = LOG_BASE_DIR / session_timestamp
        st.session_state.log_image_dir = st.session_state.log_dir / "images"
        st.session_state.log_file_path = st.session_state.log_dir / "conversation.log"
        st.session_state.log_image_counter = 0

        try:
            st.session_state.log_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.log_image_dir.mkdir(exist_ok=True)
            # Initialize log file with session info
            with open(st.session_state.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"Session started: {now.isoformat()}\n")
                f.write(f"Log Directory: {st.session_state.log_dir}\n")
                f.write("-" * 20 + "\n\n")
        except OSError as e:
            st.error(f"Failed to create log directory: {e}")
            # Fallback or disable logging? For now, just error out.
            st.session_state.log_dir = None  # Indicate logging failed


def _log_message(sender: Sender | str, text: str):
    """Appends a message to the session's log file."""
    if not st.session_state.get("log_file_path"):
        return  # Logging disabled or failed to initialize
    try:
        timestamp = datetime.now().isoformat()
        with open(st.session_state.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{sender.upper()}]:\n{text}\n\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")  # Log to console as fallback


def _log_image(base64_image: str) -> str | None:
    """Saves a base64 encoded image to the session's image log directory."""
    if not st.session_state.get("log_image_dir"):
        return None  # Logging disabled or failed to initialize

    try:
        image_data = base64.b64decode(base64_image)
        st.session_state.log_image_counter += 1
        image_filename = f"screenshot_{st.session_state.log_image_counter:04d}.png"
        image_path = st.session_state.log_image_dir / image_filename
        with open(image_path, "wb") as f:
            f.write(image_data)
        # Return relative path for logging
        return str(Path("images") / image_filename)
    except (base64.binascii.Error, OSError, Exception) as e:
        error_msg = f"Error saving image: {e}"
        print(error_msg)
        _log_message(Sender.SYSTEM, f"[ERROR] Failed to save screenshot: {e}")
        return None


def _reset_model():
    st.session_state.model = PROVIDER_TO_DEFAULT_MODEL_NAME[
        cast(APIProvider, st.session_state.provider)
    ]
    _reset_model_conf()


def _reset_model_conf():
    model_conf = (
        SONNET_3_7
        if "3-7" in st.session_state.model
        else MODEL_TO_MODEL_CONF.get(st.session_state.model, SONNET_3_5_NEW)
    )
    st.session_state.tool_version = model_conf.tool_version
    st.session_state.has_thinking = model_conf.has_thinking
    st.session_state.output_tokens = model_conf.default_output_tokens
    st.session_state.max_output_tokens = model_conf.max_output_tokens
    st.session_state.thinking_budget = int(model_conf.default_output_tokens / 2)


async def main():
    """Render loop for streamlit"""
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Claude Computer Use Demo")

    if not os.getenv("HIDE_WARNING", False):
        st.warning(WARNING_TEXT)

    with st.sidebar:

        def _reset_api_provider():
            if st.session_state.provider_radio != st.session_state.provider:
                _reset_model()
                st.session_state.provider = st.session_state.provider_radio
                st.session_state.auth_validated = False

        provider_options = [option.value for option in APIProvider]
        st.radio(
            "API Provider",
            options=provider_options,
            key="provider_radio",
            format_func=lambda x: x.title(),
            on_change=_reset_api_provider,
        )

        st.text_input("Model", key="model", on_change=_reset_model_conf)

        if st.session_state.provider == APIProvider.ANTHROPIC:
            st.text_input(
                "Anthropic API Key",
                type="password",
                key="api_key",
                on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
            )

        st.number_input(
            "Only send N most recent images",
            min_value=0,
            key="only_n_most_recent_images",
            help="To decrease the total tokens sent, remove older screenshots from the conversation",
        )
        st.text_area(
            "Custom System Prompt Suffix",
            key="custom_system_prompt",
            help="Additional instructions to append to the system prompt. see computer_use_demo/loop.py for the base system prompt.",
            on_change=lambda: save_to_storage(
                "system_prompt", st.session_state.custom_system_prompt
            ),
        )
        st.checkbox("Hide screenshots", key="hide_images")
        st.checkbox(
            "Enable token-efficient tools beta", key="token_efficient_tools_beta"
        )
        versions = get_args(ToolVersion)
        st.radio(
            "Tool Versions",
            key="tool_versions",
            options=versions,
            index=versions.index(st.session_state.tool_version),
        )

        st.number_input("Max Output Tokens", key="output_tokens", step=1)

        st.checkbox(
            "Thinking Enabled", key="thinking", value=st.session_state.has_thinking
        )  # Sync with model conf
        st.number_input(
            "Thinking Budget",
            key="thinking_budget",
            max_value=st.session_state.max_output_tokens,
            step=1,
            disabled=not st.session_state.thinking,
        )

        if st.button("Reset", type="primary"):
            with st.spinner("Resetting..."):
                log_dir_before_clear = st.session_state.get(
                    "log_dir"
                )  # Keep log dir info
                st.session_state.clear()
                setup_state()
                # Optionally log reset event
                if log_dir_before_clear:
                    _log_message(Sender.SYSTEM, "Session Reset Initiated.")
                # Start new log session
                setup_state()  # Re-initialize logging state

                subprocess.run("pkill Xvfb; pkill tint2", shell=True)  # noqa: ASYNC221
                await asyncio.sleep(1)
                subprocess.run("./start_all.sh", shell=True)  # noqa: ASYNC221

    if not st.session_state.auth_validated:
        if auth_error := validate_auth(
            st.session_state.provider, st.session_state.api_key
        ):
            st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")
            return
        else:
            st.session_state.auth_validated = True

    chat, http_logs = st.tabs(["Chat", "HTTP Exchange Logs"])
    new_message = st.chat_input(
        "Type a message to send to Claude to control the computer..."
    )

    with chat:
        # render past chats
        for message in st.session_state.messages:
            if isinstance(message["content"], str):
                _render_message(message["role"], message["content"], render_only=True)
            elif isinstance(message["content"], list):
                for block in message["content"]:
                    # the tool result we send back to the Anthropic API isn't sufficient to render all details,
                    # so we store the tool use responses
                    if isinstance(block, dict) and block["type"] == "tool_result":
                        _render_message(
                            Sender.TOOL,
                            st.session_state.tools[block["tool_use_id"]],
                            render_only=True,
                        )
                    else:
                        _render_message(
                            message["role"],
                            cast(BetaContentBlockParam | ToolResult, block),
                            render_only=True,
                        )

        # render past http exchanges
        for identity, (request, response) in st.session_state.responses.items():
            _render_api_response(request, response, identity, http_logs)

        # render past chats
        if new_message:
            # Log before adding to state, in case of interruption issues
            _log_message(Sender.USER, new_message)

            # Prepare message block for state and API
            user_message_content = [
                *maybe_add_interruption_blocks(),  # Handle interruptions first
                BetaTextBlockParam(type="text", text=new_message),
            ]
            st.session_state.messages.append(
                {
                    "role": Sender.USER,
                    "content": user_message_content,
                }
            )
            # Render the new user message (without re-logging)
            _render_message(Sender.USER, new_message, render_only=True)

        try:
            most_recent_message = st.session_state["messages"][-1]
        except IndexError:
            return

        if most_recent_message["role"] is not Sender.USER:
            # we don't have a user message to respond to, exit early
            return

        with track_sampling_loop():
            # run the agent sampling loop with the newest message
            st.session_state.messages = await sampling_loop(
                system_prompt_suffix=st.session_state.custom_system_prompt,
                model=st.session_state.model,
                provider=st.session_state.provider,
                messages=st.session_state.messages,
                output_callback=partial(_render_message, Sender.BOT, render_only=False),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=st.session_state.tools
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=http_logs,
                    response_state=st.session_state.responses,
                ),
                api_key=st.session_state.api_key,
                only_n_most_recent_images=st.session_state.only_n_most_recent_images,
                tool_version=st.session_state.tool_version,
                max_tokens=st.session_state.output_tokens,
                thinking_budget=st.session_state.thinking_budget
                if st.session_state.thinking
                else None,
                token_efficient_tools_beta=st.session_state.token_efficient_tools_beta,
            )


def maybe_add_interruption_blocks():
    if not st.session_state.in_sampling_loop:
        return []
    _log_message(Sender.SYSTEM, "[INTERRUPTION DETECTED]")
    # If this function is called while we're in the sampling loop, we can assume that the previous sampling loop was interrupted
    # and we should annotate the conversation with additional context for the model and heal any incomplete tool use calls
    result = []
    last_message = st.session_state.messages[-1]
    if isinstance(last_message.get("content"), list):
        previous_tool_use_ids = [
            block["id"]
            for block in last_message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        for tool_use_id in previous_tool_use_ids:
            # Log the specific tool interruption
            _log_message(
                Sender.SYSTEM,
                f"Interrupting tool use: {tool_use_id} - {INTERRUPT_TOOL_ERROR}",
            )
            st.session_state.tools[tool_use_id] = ToolResult(error=INTERRUPT_TOOL_ERROR)
            result.append(
                BetaToolResultBlockParam(
                    tool_use_id=tool_use_id,
                    type="tool_result",
                    content=INTERRUPT_TOOL_ERROR,
                    is_error=True,
                )
            )
    # Log the interruption text added for the user
    _log_message(Sender.SYSTEM, f"Adding user interruption text: {INTERRUPT_TEXT}")

    result.append(BetaTextBlockParam(type="text", text=INTERRUPT_TEXT))
    return result


@contextmanager
def track_sampling_loop():
    st.session_state.in_sampling_loop = True
    yield
    st.session_state.in_sampling_loop = False


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key in the sidebar to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    """
    Handle an API response by storing it to state and rendering it.
    """
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _log_message(Sender.SYSTEM, f"[API ERROR] {error.__class__.__name__}: {error}")
        _render_error(error)
    _render_api_response(request, response, response_id, tab)


def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state, logging, and rendering it."""
    tool_state[tool_id] = tool_output

    # Log tool results before rendering
    log_parts = []
    if hasattr(tool_output, "name") and tool_output.name:  # Log tool name if available
        log_parts.append(f"Tool Executed: {tool_output.name}")
    if tool_output.output:
        log_parts.append(f"Output:\n{tool_output.output}")
    if tool_output.error:
        log_parts.append(f"Error: {tool_output.error}")
    if tool_output.base64_image:
        saved_image_path = _log_image(tool_output.base64_image)
        if saved_image_path:
            log_parts.append(f"Screenshot saved: {saved_image_path}")
        else:
            log_parts.append("[Failed to save screenshot]")

    if log_parts:
        _log_message(Sender.TOOL, "\n".join(log_parts))

    # Render the message (without logging again)
    _render_message(Sender.TOOL, tool_output, render_only=True)


def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    """Render an API response to a streamlit tab"""
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception):
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_message(
    sender: Sender,
    message: str | BetaContentBlockParam | ToolResult,
    render_only: bool = False,  # Add flag to prevent double logging
):
    """Convert input/output to a streamlit message and log if not render_only."""
    is_tool_result = not isinstance(message, str | dict)
    log_content = []  # Collect parts to log for this message

    # --- Determine content for UI and potential logging ---
    ui_content_parts = []
    base64_image_to_render = None

    if is_tool_result:
        message = cast(ToolResult, message)
        if message.output:
            if message.__class__.__name__ == "CLIResult":
                ui_content_parts.append(("code", message.output))
                if not render_only:
                    log_content.append(f"Output (CLI):\n{message.output}")
            else:
                ui_content_parts.append(("markdown", message.output))
                if not render_only:
                    log_content.append(f"Output:\n{message.output}")
        if message.error:
            ui_content_parts.append(("error", message.error))
            if not render_only:
                log_content.append(f"Error: {message.error}")
        if message.base64_image and not st.session_state.hide_images:
            base64_image_to_render = message.base64_image
            # Image logging handled by _tool_output_callback

    elif isinstance(message, dict):
        if message["type"] == "text":
            text_content = message["text"]
            ui_content_parts.append(("write", text_content))
            if not render_only:
                log_content.append(text_content)
        elif message["type"] == "thinking":
            thinking_content = message.get("thinking", "")
            full_thinking_text = f"[Thinking]\n\n{thinking_content}"
            ui_content_parts.append(("markdown", full_thinking_text))
            if not render_only:
                log_content.append(full_thinking_text)
        elif message["type"] == "tool_use":
            tool_use_text = f"Tool Use: {message['name']}\nInput: {message['input']}"
            ui_content_parts.append(("code", tool_use_text))
            if not render_only:
                log_content.append(tool_use_text)
        else:
            # only expected return types are text and tool_use
            err_msg = f"Unexpected response type {message['type']}"
            ui_content_parts.append(("error", err_msg))
            if not render_only:
                _log_message(Sender.SYSTEM, f"[ERROR] {err_msg}")

    else:  # Plain string message (likely user input already logged, or simple bot response)
        ui_content_parts.append(("markdown", message))
        if not render_only:
            log_content.append(message)

    # --- Log collected content if needed ---
    if not render_only and log_content:
        full_log_text = "\n".join(log_content)
        if full_log_text.strip():  # Avoid logging empty messages
            _log_message(sender, full_log_text)

    # --- Render to Streamlit UI ---
    # Skip rendering if content is empty AND it's a tool result potentially hidden
    should_render_container = bool(
        ui_content_parts
        or (base64_image_to_render and not st.session_state.hide_images)
    )
    if not should_render_container and is_tool_result:
        return  # Don't render empty tool messages (e.g., hidden screenshots)

    if not ui_content_parts and not base64_image_to_render:
        return  # Don't render completely empty messages

    with st.chat_message(sender):
        for type, content in ui_content_parts:
            if type == "markdown":
                st.markdown(content)
            elif type == "code":
                st.code(content)
            elif type == "error":
                st.error(content)
            elif type == "write":
                st.write(content)

        if base64_image_to_render and not st.session_state.hide_images:
            try:
                st.image(base64.b64decode(base64_image_to_render))
            except Exception as e:
                st.error(f"Failed to render image: {e}")


if __name__ == "__main__":
    LOG_BASE_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())
