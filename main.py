
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types

# Persistent model handle — avoids costly init/destroy per call
_cached_model = None

def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = cactus_init(functiongemma_path)
    return _cached_model


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def _build_gemini_tools(tools):
    """Convert tool dicts to Gemini FunctionDeclaration list."""
    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API with iterative decomposition."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    estimated_actions = max(1, len(re.split(r'\band\b|,', user_text)))

    gemini_tools = _build_gemini_tools(tools)
    contents = [
        "Call ALL relevant tools for every action the user requests. "
        "If the user asks for multiple things, return multiple function calls.\n\n"
        + " ".join(m["content"] for m in messages if m["role"] == "user")
    ]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    # Iterative decomposition: if Gemini returned fewer calls than expected,
    # re-run with the remaining tools to pick up missed actions.
    if len(function_calls) < estimated_actions and estimated_actions > 1:
        got_names = {c["name"] for c in function_calls}
        remaining = [t for t in tools if t["name"] not in got_names]
        if remaining:
            remaining_gemini = _build_gemini_tools(remaining)
            resp2 = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=remaining_gemini),
            )
            for candidate in resp2.candidates:
                for part in candidate.content.parts:
                    if part.function_call and part.function_call.name not in got_names:
                        function_calls.append({
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args),
                        })
                        got_names.add(part.function_call.name)

    total_time_ms = (time.time() - start_time) * 1000

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _clean_string_arg(v):
    """Strip trailing punctuation that models often add."""
    if isinstance(v, str):
        return v.strip(".,;!?'\"`()[] ")
    return v


def _coerce_args(call, tool_by_name):
    """Cast argument types to match tool schema and clean string values."""
    name = call.get("name", "")
    args = call.get("arguments", {})
    tool_def = tool_by_name.get(name)
    if not tool_def:
        return call
    props = tool_def.get("parameters", {}).get("properties", {})
    coerced = {}
    for k, v in args.items():
        if k in props:
            expected = props[k].get("type", "string")
            if expected == "integer" and not isinstance(v, int):
                try:
                    coerced[k] = int(float(str(v)))
                except (ValueError, TypeError):
                    coerced[k] = v
            elif expected == "number" and not isinstance(v, (int, float)):
                try:
                    coerced[k] = float(str(v))
                except (ValueError, TypeError):
                    coerced[k] = v
            elif expected == "string":
                coerced[k] = _clean_string_arg(v)
            else:
                coerced[k] = v
        else:
            coerced[k] = v
    return {"name": name, "arguments": coerced}


def _run_local(model, msgs, cactus_tools, retries=2):
    """Call cactus_complete, retrying when cloud_handoff=true or calls are empty."""
    last_raw = {"function_calls": [], "confidence": 0}
    for attempt in range(retries + 1):
        if attempt > 0:
            cactus_reset(model)
        raw_str = cactus_complete(
            model,
            msgs,
            tools=cactus_tools,
            force_tools=True,
            max_tokens=256,
            confidence_threshold=0.001,
            tool_rag_top_k=0,
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            continue
        last_raw = raw
        handoff = raw.get("cloud_handoff", False)
        calls = raw.get("function_calls", [])
        conf = raw.get("confidence", 0)
        if calls and not handoff:
            return calls, conf, raw
    return last_raw.get("function_calls", []), last_raw.get("confidence", 0), last_raw


def _extract_numbers(text):
    """Pull all integers from a string (e.g. '10 AM' → {10}, '7:30' → {7, 30})."""
    return set(int(n) for n in re.findall(r'\b\d+\b', text))


def _is_well_formed(call, tool_by_name, user_text=""):
    """Check a call has all required params with plausible values.

    Validation layers:
      - All required params present and non-empty
      - Integer params: value must appear in user text (or be 0)
      - String params (multi-word): at least one 3+ char word must appear in user text
    """
    name = call.get("name")
    args = call.get("arguments", {})
    tool = tool_by_name.get(name)
    if not tool:
        return False
    schema = tool.get("parameters", {})
    required = schema.get("required", [])
    props = schema.get("properties", {})
    text_numbers = _extract_numbers(user_text) | {0}
    user_words = set(re.findall(r'[a-zA-Z]{3,}', user_text.lower()))

    for param in required:
        val = args.get(param)
        if val is None or val == "" or val == []:
            return False
        expected_type = props.get(param, {}).get("type", "string")
        if expected_type == "integer":
            try:
                int_val = int(float(str(val)))
            except (ValueError, TypeError):
                return False
            if int_val not in text_numbers:
                return False
        elif expected_type == "string" and isinstance(val, str):
            val_words = set(re.findall(r'[a-zA-Z]{3,}', val.lower()))
            if len(val_words) >= 2:
                overlap = len(val_words & user_words)
                if overlap / len(val_words) < 0.5:
                    return False
            if re.search(r'\d{4}-\d{2}-\d{2}', val) or re.search(r'\d{2}T\d{2}:', val):
                return False
    return True


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    EchoPath multi-signal hybrid routing.

    Strategy:
      1. Run local FunctionGemma (cached model, KV reset, cloud_handoff disabled)
      2. Validate results: function name + all required args present + correct types
      3. For multi-action: iterative decomposition to pick up missing calls
      4. Stay local ONLY when well-formed calls cover all estimated actions
      5. Escalate to cloud when local output is incomplete or malformed
    """
    model = _get_model()
    cactus_reset(model)

    tool_names = {t["name"] for t in tools}
    tool_by_name = {t["name"]: t for t in tools}
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    sys_msg = {"role": "system", "content": "You are a helpful assistant. Use the provided tools to fulfill the user's request. Always respond with tool calls."}

    start = time.time()

    local_calls, confidence, raw = _run_local(
        model, [sys_msg] + messages, cactus_tools, retries=1
    )
    valid_calls = [c for c in local_calls if c.get("name") in tool_names]

    # Estimate how many distinct actions the user requested
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    estimated_actions = max(1, len(re.split(r'\band\b|,', user_text)))

    # Iterative decomposition for multi-action requests
    if len(valid_calls) < estimated_actions and estimated_actions > 1:
        got_names = {c["name"] for c in valid_calls}
        remaining = [t for t in tools if t["name"] not in got_names]
        if remaining:
            cactus_reset(model)
            remaining_cactus = [{"type": "function", "function": t} for t in remaining]
            extra_calls, _, _ = _run_local(
                model, [sys_msg] + messages, remaining_cactus, retries=1
            )
            for c in extra_calls:
                if c.get("name") in tool_names and c.get("name") not in got_names:
                    valid_calls.append(c)
                    got_names.add(c["name"])

    local_time_ms = (time.time() - start) * 1000

    valid_calls = [_coerce_args(c, tool_by_name) for c in valid_calls]

    # Keep only well-formed calls (all required args present, correct type, plausible values)
    well_formed = [c for c in valid_calls if _is_well_formed(c, tool_by_name, user_text)]

    # Routing: stay local when well-formed calls cover all estimated actions
    use_local = len(well_formed) >= estimated_actions and len(well_formed) > 0


    if use_local:
        return {
            "function_calls": well_formed,
            "total_time_ms": local_time_ms,
            "confidence": confidence,
            "source": "on-device",
        }

    # Cloud fallback: local output was incomplete or malformed
    try:
        cloud = generate_cloud(messages, tools)
        cloud["function_calls"] = [_coerce_args(c, tool_by_name) for c in cloud["function_calls"]]
        cloud["source"] = "cloud (fallback)"
        cloud["local_confidence"] = confidence
        cloud["total_time_ms"] += local_time_ms
        return cloud
    except Exception:
        fallback = well_formed if well_formed else valid_calls if valid_calls else local_calls
        return {
            "function_calls": fallback,
            "total_time_ms": local_time_ms,
            "confidence": confidence,
            "source": "on-device",
        }


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)

