from bae.constants import ACTION_STOP, PIXEL_TO_METER
from bae_agent_base import BAEAgentBase


class BAEAgent(BAEAgentBase):
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
        self.pending_action_list = []

    def reset(self, episode_ref, sim=None):
        self.reset_episode_state(episode_ref)
        self.pending_action_list = []
        print("BAE Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        instruction = observations.get("instruction")["text"]
        rgb = observations.get("rgb")

        paths, occ_h, occ_w = self.save_observation_images(sim, rgb)

        if self.pending_action_list:
            return self._pop_pending_action()

        image_paths = self.build_prompt_image_paths(paths)
        cur_x, cur_y = self.get_current_pixel(sim)
        prev_actions_xml = self.build_prev_actions_xml()

        try:
            actions, pixels, raw_text, prompt_text = self.inference.predict(
                image_paths=image_paths,
                instruction=instruction,
                cur_x=cur_x,
                cur_y=cur_y,
                occ_w=occ_w,
                occ_h=occ_h,
                occ_meter_per_px=PIXEL_TO_METER,
                occ_rot_deg=0,
                prev_actions=prev_actions_xml,
            )

            predicted_actions = list(actions) if actions else []
            selected_actions = predicted_actions[: self.action_num]
            if not selected_actions:
                return {"action": ACTION_STOP}

            self.pending_action_list = selected_actions
            action = self.pending_action_list.pop(0)
            action_img = sim.get_occ_map_with_actions(
                predicted_actions, traj_color=(255, 165, 0)
            )
            self.save_image(action_img, "trajectory_vis/action_rollout")
            self.history_action_list.append(action)
            return {"action": action}

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            return {"action": ACTION_STOP}

    def _pop_pending_action(self):
        action = self.pending_action_list.pop(0)
        self.history_action_list.append(action)
        return {"action": action}
