#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BAE inference wrapper using Qwen3VL + LlamaFactory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import Qwen3VLForConditionalGeneration

from .prompts import build_prompt
from .parser import parse_actions6, parse_pixels, validate_actions

# Try to import LlamaFactory
_LLAMAFACTORY_AVAILABLE = False
try:
    from llamafactory.hparams import ModelArguments
    from llamafactory.model import load_tokenizer

    _LLAMAFACTORY_AVAILABLE = True
except ImportError:
    pass


class BAEInference:
    """BAE inference engine using Qwen3VL model."""

    def __init__(
        self,
        model_path: str,
        prompt_type: str = "V3",
        dtype: str = "bf16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 128,
    ):
        """
        Initialize BAE inference.

        Args:
            model_path: Path to Qwen3VL model directory
            prompt_type: "V1", "V2", or "V3"
            dtype: Model dtype ("bf16", "fp16", "fp32", or "auto")
            device_map: Device map for model loading
            trust_remote_code: Whether to trust remote code
            max_new_tokens: Maximum new tokens to generate
        """
        if not _LLAMAFACTORY_AVAILABLE:
            raise RuntimeError(
                "LlamaFactory not available. Install it or add LlamaFactory/src to sys.path"
            )

        self.model_path = model_path
        self.prompt_type = prompt_type.upper()
        self.max_new_tokens = max_new_tokens

        # Load model
        print(f"Loading BAE model from {model_path}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=self._resolve_dtype(dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Load processor via LlamaFactory
        model_args = ModelArguments(
            model_name_or_path=model_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_module = load_tokenizer(model_args)
        self.processor = tokenizer_module["processor"]

        if self.processor is None:
            raise RuntimeError("Failed to load processor from LlamaFactory")

        print("BAE model loaded successfully")

    def _resolve_dtype(self, dtype: str) -> Any:
        """Resolve dtype string to torch dtype."""
        d = str(dtype).lower()
        if d == "auto":
            return "auto"
        if d == "bf16":
            return torch.bfloat16
        if d == "fp16":
            return torch.float16
        if d == "fp32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _build_messages(
        self, prompt_text: str, image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Build messages for Qwen3VL processor."""
        content: List[Dict[str, Any]] = []
        for p in image_paths:
            content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def _get_eos_token_id(self) -> Optional[Union[int, List[int]]]:
        """Get EOS token ID from model or processor."""
        # Try generation_config first
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            gid = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(gid, (list, tuple)):
                gids = [int(x) for x in gid if isinstance(x, int)]
                if gids:
                    return gids
            if isinstance(gid, int):
                return gid

        # Try tokenizer
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            return None

        tid = getattr(tok, "eos_token_id", None)
        if isinstance(tid, (list, tuple)):
            tids = [int(x) for x in tid if isinstance(x, int)]
            if tids:
                return tids
        if isinstance(tid, int):
            return tid

        # Try eos_token string
        eos_tok = getattr(tok, "eos_token", None)
        if isinstance(eos_tok, str):
            try:
                cid = tok.convert_tokens_to_ids(eos_tok)
                if isinstance(cid, int):
                    return cid
            except Exception:
                pass

        # Try Qwen-specific token
        try:
            cid = tok.convert_tokens_to_ids("<|im_end|>")
            if isinstance(cid, int):
                return cid
        except Exception:
            pass

        return None

    def _extract_token_probs(
        self, gen_ids: torch.Tensor, step_scores: Sequence[torch.Tensor]
    ) -> Optional[List[float]]:
        """Extract per-step probability for the selected generated token (batch=1)."""
        if gen_ids.ndim != 2:
            return None

        steps = min(gen_ids.shape[1], len(step_scores))
        if steps <= 0:
            return []

        token_probs: List[float] = []
        for t in range(steps):
            logits_t = step_scores[t]
            probs_t = torch.softmax(logits_t, dim=-1)
            token_t = gen_ids[:, t].unsqueeze(-1)
            picked = probs_t.gather(-1, token_t).squeeze(-1)
            token_probs.append(float(picked[0].item()))

        return token_probs

    @torch.inference_mode()
    def predict(
        self,
        image_paths: List[str],
        instruction: str,
        cur_x: Optional[int] = None,
        cur_y: Optional[int] = None,
        occ_w: Optional[int] = None,
        occ_h: Optional[int] = None,
        occ_meter_per_px: float = 0.05,
        occ_rot_deg: int = 0,
        prev_actions: Optional[str] = None,
        return_token_probs: bool = False,
    ) -> Union[
        Tuple[Optional[List[int]], Optional[List[List[int]]], str, str],
        Tuple[
            Optional[List[int]],
            Optional[List[List[int]]],
            str,
            str,
            Optional[List[float]],
        ],
    ]:
        """
        Predict 6 actions and pixel waypoints for navigation.

        Args:
            image_paths: List of image paths (order depends on prompt_type)
            instruction: Navigation instruction text
            cur_x: Current x pixel coordinate in original OCC pixels (required for V1/V2)
            cur_y: Current y pixel coordinate in original OCC pixels (required for V1/V2)
            occ_w: OCC map width in pixels (required for V1/V2)
            occ_h: OCC map height in pixels (required for V1/V2)
            occ_meter_per_px: Physical OCC scale (meters per pixel), used in prompt only.

        Returns:
            (actions, pixels, raw_text, prompt_text) tuple where:
            - actions: list of 6 ints or None
            - pixels: list of [x,y] waypoints or None
            - raw_text: raw model output string
            - prompt_text: rendered prompt passed to model
            If return_token_probs=True, returns
            (actions, pixels, raw_text, prompt_text, token_probs) where:
            - token_probs: per-generated-token probabilities (batch=1) or None
        """
        # Build prompt
        prompt_text = build_prompt(
            self.prompt_type,
            instruction,
            cur_x=cur_x,
            cur_y=cur_y,
            occ_w=occ_w,
            occ_h=occ_h,
            occ_meter_per_px=occ_meter_per_px,
            occ_rot_deg=occ_rot_deg,
            prev_actions=prev_actions,
        )

        # Build messages
        messages = self._build_messages(prompt_text, image_paths)

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }

        if return_token_probs:
            gen_kwargs["return_dict_in_generate"] = True
            gen_kwargs["output_scores"] = True

        eos_token_id = self._get_eos_token_id()

        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id

        generation_output = self.model.generate(**inputs, **gen_kwargs)

        token_probs: Optional[List[float]] = None
        if return_token_probs:
            output_ids = generation_output.sequences
            token_probs = self._extract_token_probs(
                output_ids[:, len(inputs.input_ids[0]) :], generation_output.scores
            )
        else:
            output_ids = generation_output

        # Decode
        out_trim = output_ids[0][len(inputs.input_ids[0]) :]
        gen_text = self.processor.decode(
            out_trim, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Parse actions and pixels
        actions = parse_actions6(gen_text)

        if actions is not None and not validate_actions(actions):
            print(f"Warning: Invalid actions detected: {actions}")
            actions = None

        pixels = parse_pixels(gen_text)

        if return_token_probs:
            return actions, pixels, gen_text, prompt_text, token_probs

        return actions, pixels, gen_text, prompt_text
