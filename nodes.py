import torch
from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_api.latest import ComfyExtension, io
import comfy.utils
import logging

NUM_FLOW_SOCKETS = 5


def _accum_count(accum):
    """Count items in an accumulation, handling tensors (Image/Mask) and dicts (Latent)."""
    if not isinstance(accum, dict) or "accum" not in accum:
        return 0
    total = 0
    for item in accum["accum"]:
        if isinstance(item, dict):
            total += item["samples"].shape[0]
        else:
            total += item.shape[0]
    return total

class _AccumulateNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_AccumulateNode",
            display_name="Accumulate",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[
                io.AnyType.Input("to_add"),
                io.Accumulation.Input("accumulation", optional=True),
            ],
            outputs=[
                io.Accumulation.Output(),
            ],
        )

    @classmethod
    def execute(cls, to_add, accumulation=None) -> io.NodeOutput:
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        logging.info(f"[_AccumulateNode] accum length now: {len(value)}, to_add shape: {getattr(to_add, 'shape', type(to_add))}")
        return io.NodeOutput({"accum": value})

class WhileLoopOpen(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WhileLoopOpen",
            display_name="While Loop Open",
            category="looping",
            inputs=[
                io.Boolean.Input("condition", default=True),
                *[io.AnyType.Input(f"initial_value{i}", optional=True) for i in range(NUM_FLOW_SOCKETS)],
            ],
            outputs=[
                io.FlowControl.Output("flow_control", display_name="FLOW_CONTROL"),
                *[io.AnyType.Output(f"value{i}") for i in range(NUM_FLOW_SOCKETS)],
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, condition: bool, **kwargs) -> io.NodeOutput:
        values = [kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS)]
        return io.NodeOutput("stub", *values)

class WhileLoopClose(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WhileLoopClose",
            display_name="While Loop Close",
            category="looping",
            inputs=[
                io.FlowControl.Input("flow_control", raw_link=True),
                io.Boolean.Input("condition", force_input=True),
                *[io.AnyType.Input(f"initial_value{i}", optional=True) for i in range(NUM_FLOW_SOCKETS)],
            ],
            outputs=[
                *[io.AnyType.Output(f"value{i}") for i in range(NUM_FLOW_SOCKETS)],
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
            enable_expand=True,
            accept_all_inputs=True,
        )

    @staticmethod
    def _explore_dependencies(node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    WhileLoopClose._explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    @staticmethod
    def _collect_contained(node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                WhileLoopClose._collect_contained(child_id, upstream, contained)

    @classmethod
    def execute(cls, flow_control, condition: bool, **kwargs) -> io.NodeOutput:
        dynprompt = cls.hidden.dynprompt
        unique_id = cls.hidden.unique_id
        values = [kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS)]

        if not condition: # Done with the loop — return current values
            return io.NodeOutput(*values)

        # Build the graph expansion for the next loop iteration
        upstream = {}
        cls._explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        cls._collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # Use "Recurse" for this node's clone to avoid exponential name growth
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(NUM_FLOW_SOCKETS):
            new_open.set_input(f"initial_value{i}", values[i])

        my_clone = graph.lookup_node("Recurse")
        result = tuple(my_clone.out(x) for x in range(NUM_FLOW_SOCKETS))
        return io.NodeOutput(*result, expand=graph.finalize())


class _IntOperations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_IntOperations",
            display_name="Int Operations",
            category="looping/logic",
            is_dev_only=True,
            inputs=[
                io.Int.Input("a", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Int.Input("b", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Combo.Input("operation", options=[
                    "add", "subtract", "multiply", "divide", "modulo", "power",
                    "==", "!=", "<", ">", "<=", ">=",
                ]),
            ],
            outputs=[
                io.Int.Output(),
                io.Boolean.Output(),
            ],
        )

    OPS = {
        "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b, "divide": lambda a, b: a // b if b else 0,
        "modulo": lambda a, b: a % b if b else 0, "power": lambda a, b: a ** b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
    }

    @classmethod
    def execute(cls, a: int, b: int, operation: str) -> io.NodeOutput:
        result = cls.OPS[operation](a, b)
        return io.NodeOutput(int(result), bool(result))

class _AccumulationToImageBatch(io.ComfyNode):
    """Internal helper: concatenates an ACCUMULATION of IMAGE/MASK tensors or LATENT dicts into a single batch."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_AccumulationToImageBatch",
            display_name="Accumulation to Batch",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[io.Accumulation.Input("accumulation")],
            outputs=[io.AnyType.Output("result")],
        )

    @classmethod
    def execute(cls, accumulation) -> io.NodeOutput:
        items = accumulation["accum"]
        if isinstance(items[0], dict):
            # Latent dicts — batch the "samples" tensors
            from comfy_extras.nodes_post_processing import batch_latents
            return io.NodeOutput(batch_latents(items))
        else:
            return io.NodeOutput(torch.cat(items, dim=0))

class TensorForLoopOpen(io.ComfyNode):
    """
    Opens a loop that runs N times and collects outputs.
    Wire:
      - flow_control → TensorForLoopClose
      - Use `previous_value` (last iteration's result, or initial_value on first pass) as input to your generation.
      - Connect your generated output → TensorForLoopClose.processed
    Supports IMAGE, MASK, and LATENT types.
    """
    MATCHTYPE = io.MatchType.Template("data", allowed_types=[io.Image, io.Mask, io.Latent])

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TensorForLoopOpen",
            display_name="Tensor For Loop Open",
            category="looping/accumulation",
            inputs=[
                io.Int.Input("count", default=4, min=1, tooltip="Number of loop iterations."),
                io.MatchType.Input("initial_value", template=cls.MATCHTYPE, optional=True,
                                   tooltip="Optional value to use as `previous_value` on the first iteration."),
            ],
            outputs=[
                io.FlowControl.Output("flow_control"),
                io.AnyType.Output("loop_state", tooltip="Internal — connect to TensorForLoopClose."),
                io.MatchType.Output(cls.MATCHTYPE, id="previous_value",
                                    tooltip="The value from the previous_value iteration (or initial_value on first pass)."),
                io.Int.Output("accumulated_count", tooltip="Number of items collected so far (0 on first iteration)."),
                io.Int.Output("current_iteration", tooltip="Current iteration index (1-based)."),
            ],
            hidden=[io.Hidden.unique_id],
            accept_all_inputs=True,
        )


    @classmethod
    def execute(cls, count: int, initial_value=None, **kwargs) -> io.NodeOutput:
        unique_id = cls.hidden.unique_id
        state     = kwargs.get("initial_value0")  # packed dict or None on first pass
        remaining = state["remaining"] if state is not None else count
        accum     = state["accum"]     if state is not None else None
        previous_value  = state["previous_value"]  if state is not None else initial_value
        # Preserve the original open node id so progress always targets the same node
        open_node_id = state["open_node_id"] if state is not None else unique_id
        accumulated_count = _accum_count(accum)
        current_iteration = count - remaining + 1
        loop_state = {"remaining": remaining, "accum": accum, "previous_value": previous_value, "count": count, "open_node_id": open_node_id}
        return io.NodeOutput("stub", loop_state, previous_value, accumulated_count, current_iteration)


class TensorForLoopClose(io.ComfyNode):
    """
    Closes the loop started by TensorForLoopOpen.
    Connect:
      - flow_control from TensorForLoopOpen
      - processed: the output generated this iteration
    Supports IMAGE, MASK, and LATENT types.
    """
    MATCHTYPE = io.MatchType.Template("data", allowed_types=[io.Image, io.Mask, io.Latent])

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TensorForLoopClose",
            display_name="Tensor For Loop Close",
            category="looping/accumulation",
            inputs=[
                io.FlowControl.Input("flow_control", raw_link=True),
                io.MatchType.Input("processed", template=cls.MATCHTYPE, raw_link=True, tooltip="Output generated this iteration."),
                io.Boolean.Input("accumulate", default=True,
                                 tooltip="When enabled, collects all iterations into a batch. When disabled, only outputs the final iteration's result."),
            ],
            outputs=[
                io.MatchType.Output(cls.MATCHTYPE, id="output", tooltip="Accumulated batch or final iteration result, depending on 'accumulate' setting."),
            ],
            enable_expand=True,
        )


    @classmethod
    def execute(cls, flow_control, processed, accumulate=True) -> io.NodeOutput:
        graph = GraphBuilder()
        open_id = flow_control[0]
        # slot 1 of TensorForLoopOpen = loop_state dict {remaining, accum, previous_value}
        unpack = graph.node("_ImageAccumStateUnpack", loop_state=[open_id, 1])
        # unpack outputs: 0=remaining, 1=accum (Accumulation list), 2=previous_value, 3=accumulated_count, 4=count
        sub  = graph.node("_IntOperations", operation="subtract", a=unpack.out(0), b=1)
        cond = graph.node("_IntOperations", a=sub.out(0), b=0, operation=">")

        if accumulate:
            # _AccumulateNode appends a Python reference — no tensor copy, O(1) per iteration
            accum_node = graph.node("_AccumulateNode", to_add=processed, accumulation=unpack.out(1))
            pack = graph.node("_ImageAccumStatePack",
                remaining=sub.out(0),
                accum=accum_node.out(0),
                previous_value=processed,
                count=unpack.out(4),
                open_node_id=unpack.out(5),
            )
        else:
            # No accumulation — just pass the image through as 'previous_value'
            pack = graph.node("_ImageAccumStatePack",
                remaining=sub.out(0),
                accum=unpack.out(1),
                previous_value=processed,
                count=unpack.out(4),
                open_node_id=unpack.out(5),
            )

        while_close = graph.node(
            "WhileLoopClose",
            flow_control=flow_control,
            condition=cond.out(1),
            initial_value0=pack.out(0),
        )

        if accumulate:
            # Concatenate all collected images into one batch tensor
            final_unpack = graph.node("_ImageAccumStateUnpack", loop_state=while_close.out(0))
            final_batch = graph.node("_AccumulationToImageBatch", accumulation=final_unpack.out(1))
            result = final_batch.out(0)
        else:
            # Just return the last iteration's image
            final_unpack = graph.node("_ImageAccumStateUnpack", loop_state=while_close.out(0))
            result = final_unpack.out(2)  # out(2) = previous_value

        return io.NodeOutput(result, expand=graph.finalize())


class _ImageAccumStatePack(io.ComfyNode):
    """Internal helper: packs state into a single dict for TensorForLoopOpen's initial_value0."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_ImageAccumStatePack",
            display_name="Image Accum State Pack",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[
                io.AnyType.Input("remaining"),
                io.Accumulation.Input("accum", optional=True),
                io.AnyType.Input("previous_value"),
                io.AnyType.Input("count"),
                io.AnyType.Input("open_node_id"),
            ],
            outputs=[io.AnyType.Output("loop_state")],
        )

    @classmethod
    def execute(cls, remaining, accum, previous_value, count, open_node_id) -> io.NodeOutput:
        # Update progress on the TensorForLoopOpen node after each iteration completes
        current_iteration = count - remaining
        comfy.utils.ProgressBar(count, node_id=open_node_id).update_absolute(current_iteration)
        return io.NodeOutput({"remaining": remaining, "accum": accum, "previous_value": previous_value, "count": count, "open_node_id": open_node_id})


class _ImageAccumStateUnpack(io.ComfyNode):
    """Internal helper: unpacks loop_state from TensorForLoopOpen."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_ImageAccumStateUnpack",
            display_name="Image Accum State Unpack",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[io.AnyType.Input("loop_state")],
            outputs=[
                io.Int.Output("remaining"),
                io.Accumulation.Output("accumulation"),
                io.AnyType.Output("previous_value"),
                io.Int.Output("accumulated_count"),
                io.Int.Output("count"),
                io.AnyType.Output("open_node_id"),
            ],
        )

    @classmethod
    def execute(cls, loop_state) -> io.NodeOutput:
        remaining = loop_state["remaining"]
        accum = loop_state["accum"]
        previous_value = loop_state["previous_value"]
        count = loop_state.get("count", 0)
        open_node_id = loop_state.get("open_node_id")
        accumulated_count = _accum_count(accum)
        return io.NodeOutput(remaining, accum, previous_value, accumulated_count, count, open_node_id)


class LoopExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WhileLoopOpen,
            WhileLoopClose,
            TensorForLoopOpen,
            TensorForLoopClose,
            _AccumulateNode,
            _IntOperations,
            _AccumulationToImageBatch,
            _ImageAccumStateUnpack,
            _ImageAccumStatePack,
        ]

def comfy_entrypoint():
    return LoopExtension()
