import json

from pathlib import Path
from GN_Bench.tasks.nav.dagger_oracle import DaggerNavigationOracleMixin
from bae_agent_base import BAEAgentBase
from bae.constants import (
    ACTION_STOP,
    NUM_ACTIONS,
    PIXEL_TO_METER,
)


class BAEAgent(DaggerNavigationOracleMixin, BAEAgentBase):
    def __init__(
        self,
        model_path,
        result_path,
        prompt_type,
        action_num=1,
        dtype="bf16",
    ):
        super().__init__(
            model_path=model_path,
            result_path=result_path,
            prompt_type=prompt_type,
            action_num=action_num,
            dtype=dtype,
        )
        self.pending_pixel_list = None
        self.terminated_by_invalid = False
        self.llm_call_count = 0
        self.sample_path = None
        self.meta_path = None
        self.position_dump_counts = None
        self.goal_zone_exit_triggered = False
        self.goal_zone_exit_post_budget = None

    def reset(self, episode_ref, sim=None):
        self.reset_episode_state(episode_ref)
        self.pending_pixel_list = []
        self.terminated_by_invalid = False
        self.llm_call_count = 0
        self.position_dump_counts = {}
        self.goal_zone_exit_triggered = False
        self.goal_zone_exit_post_budget = None

        self.sample_path = Path(self.result_path) / self.episode_id / "sample.json"
        self.meta_path = Path(self.result_path) / self.episode_id / "meta.json"

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(sim.meta, f, ensure_ascii=False, indent=2)

        print("BAE Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        goal_position = observations.get("goal_position")
        instruction = observations.get("instruction")["text"]
        curr_dtg = info.get("distance_to_goal")

        rgb = observations.get("rgb")
        skip_supervision = not sim.config.COLLIDABLE and self._is_agent_in_obstacle(sim)

        if self.terminated_by_invalid:
            return self._return_action(ACTION_STOP)

        if (
            self.goal_zone_exit_post_budget is not None
            and self.goal_zone_exit_post_budget <= 0
        ):
            return self._return_action(ACTION_STOP)

        paths, occ_h, occ_w = self.save_observation_images(sim, rgb)
        image_paths = self.build_prompt_image_paths(paths)
        cur_x, cur_y = self.get_current_pixel(sim)
        prev_actions_xml = self.build_prev_actions_xml()

        # Always dump oracle planning visualizations for debugging,
        # regardless of whether later correction uses Strategy A or B.
        if skip_supervision:
            oracle_cache = None
        else:
            oracle_cache = self._save_oracle_debug_images(sim, goal_position)

        # Call inference
        try:
            agent_pose = self.get_agent_pose(sim)

            actions, pixels, raw_text, prompt_text, token_probs = (
                self.inference.predict(
                    image_paths=image_paths,
                    instruction=instruction,
                    cur_x=cur_x,
                    cur_y=cur_y,
                    occ_w=occ_w,
                    occ_h=occ_h,
                    occ_meter_per_px=PIXEL_TO_METER,
                    occ_rot_deg=0,
                    prev_actions=prev_actions_xml,
                    return_token_probs=True,
                )
            )

            if skip_supervision:
                self.llm_call_count += 1
                self._log_skip_supervision_step(
                    instruction=instruction,
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    cur_x=cur_x,
                    cur_y=cur_y,
                    occ_h=occ_h,
                    occ_w=occ_w,
                    goal_position=goal_position,
                    agent_pose=agent_pose,
                    raw_text=raw_text,
                    actions=actions,
                    pixels=pixels,
                    token_probs=token_probs,
                )

                if actions and len(actions) == NUM_ACTIONS:
                    exec_action = int(actions[0])
                    return self._return_action(exec_action)

                return self._return_action(ACTION_STOP)

            valid = True
            valid_reasons = []

            # vlnce action validity
            act_valid, act_reason = self._rollout_actions_valid(sim, actions)
            if not act_valid:
                valid = False
                valid_reasons.append(act_reason)
            action_hit_step_idx = self._parse_rollout_hit_step(act_reason)

            # Goal zone analysis on actions
            exit_by_action, exit_action_reason, action_goal_info = (
                self._goal_zone_analysis_on_actions(sim, actions, goal_position)
            )
            if exit_by_action:
                valid = False
                valid_reasons.append(exit_action_reason)

            # Pixel path validity
            pix_valid = None
            pix_reason = "skip_for_v3"
            pixel_hit_step_idx = None
            if self.prompt_type in {"V1", "V2"}:
                pix_valid, pix_reason = self._pixel_path_valid(sim, pixels)
                pixel_hit_step_idx = self._parse_pixel_hit_step(pix_reason)
                if not pix_valid:
                    valid = False
                    valid_reasons.append(pix_reason)

            # Goal zone analysis on pixels
            exit_by_pixel, exit_pixel_reason, pixel_goal_info = (
                self._goal_zone_analysis_on_pixels(sim, pixels, goal_position)
            )
            if exit_by_pixel:
                valid = False
                valid_reasons.append(exit_pixel_reason)

            if exit_by_action or exit_by_pixel:
                if not self.goal_zone_exit_triggered:
                    self.goal_zone_exit_triggered = True
                    self.goal_zone_exit_post_budget = 10

            stop_in_actions = (
                actions is not None
                and len(actions) == NUM_ACTIONS
                and int(actions[0]) == ACTION_STOP
            )
            if stop_in_actions and curr_dtg > 1.0:
                valid = False
                valid_reasons.append("stop_outside_goal")

            gt_true_pair = None
            gt_build_reason = None

            context_payload = {
                "instruction": instruction,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "cur_pixel": [cur_x, cur_y],
                "occ_shape": [occ_h, occ_w],
                "action_goal_info": action_goal_info,
                "pixel_goal_info": pixel_goal_info,
            }

            is_overshoot = any(r.startswith("goal_zone_exit_") for r in valid_reasons)

            # Strategy A: Re-plan path using A* + MPC
            gt_output, meta, reason = self._handle_standard_correction(
                sim, goal_position, oracle_cache
            )
            if is_overshoot:
                # Strategy B: Stop exactly at entry point
                gt_output, meta, reason = self._handle_goal_zone_exit(
                    sim, actions, pixels, context_payload
                )

            # Assemble final GT pair if correction was successful
            if gt_output:
                gt_build_reason = reason
                gt_true_pair = {
                    "input": self._construct_gt_input(context_payload),
                    "output": gt_output,
                    **meta,
                }

            self.llm_call_count += 1
            self._log_inference_step(
                instruction,
                prompt_text,
                image_paths,
                cur_x,
                cur_y,
                occ_h,
                occ_w,
                goal_position,
                agent_pose,
                raw_text,
                actions,
                pixels,
                token_probs,
                valid,
                act_valid,
                act_reason,
                action_hit_step_idx,
                stop_in_actions,
                curr_dtg,
                exit_by_action,
                exit_action_reason,
                pix_valid,
                pix_reason,
                pixel_hit_step_idx,
                exit_by_pixel,
                exit_pixel_reason,
                valid_reasons,
                gt_true_pair,
                gt_build_reason,
            )

            action_hard_fail = False
            if not act_valid:
                # Future collision in rollout (step > 0) should not stop now.
                if action_hit_step_idx is None or action_hit_step_idx <= 0:
                    action_hard_fail = True

            if stop_in_actions and curr_dtg > 1.0:
                action_hard_fail = True

            if actions and len(actions) == NUM_ACTIONS and not action_hard_fail:
                exec_action = int(actions[0])

                if stop_in_actions:
                    return self._return_action(ACTION_STOP)

                if action_hit_step_idx is not None and action_hit_step_idx > 0:
                    print(
                        "Warning: Future wall hit at step "
                        f"{action_hit_step_idx}, continuing."
                    )
                if any(self._is_pixel_wall_reason(r) for r in valid_reasons):
                    print("Warning: Pixel path hits wall, continuing.")

                print(f"BAE predicted actions: {actions}")

                if pixels:
                    print(f"BAE predicted pixels: {len(pixels)} waypoints")

                print(f"Raw output: {raw_text}")
                return self._return_action(exec_action)
            else:
                print(
                    f"Terminating episode due to invalid output. Reasons: {valid_reasons}"
                )
                if not sim.config.COLLIDABLE:
                    exec_action = int(actions[0])
                    print(f"Raw output: {raw_text}")
                    return self._return_action(exec_action)

                self.terminated_by_invalid = True
                return self._return_action(ACTION_STOP)

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            self.llm_call_count += 1
            return self._return_action(ACTION_STOP)

    def _return_action(self, action: int):
        """Return an action while honoring the post-goal-zone stop budget."""
        out_action = int(action)

        if (
            self.goal_zone_exit_post_budget is not None
            and self.goal_zone_exit_post_budget <= 0
        ):
            out_action = ACTION_STOP

        if (
            self.goal_zone_exit_post_budget is not None
            and self.goal_zone_exit_post_budget > 0
        ):
            self.goal_zone_exit_post_budget -= 1

        self.history_action_list.append(out_action)
        return {"action": out_action}
