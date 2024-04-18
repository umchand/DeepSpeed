# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch

from deepspeed.utils import logger, groups
from deepspeed.moe.utils import is_moe_param_group, is_moe_param, remap_param_name
from deepspeed.utils.tensor_fragment import map_to_flat_opt_states
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank

import deepspeed.comm as dist

class DeepSpeedOptimizer(object):
    pass


class ZeROOptimizer(DeepSpeedOptimizer):

    def load_hp_checkpoint_state_from_checkpoint_dir(self, lp_groups_name: str, checkpoint_dir: str) -> None:
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")
        optim_state_path = os.path.join(checkpoint_dir, "optimizer_state.pt")
        assert os.path.isfile(
            optim_state_path), f'{optim_state_path} containing optimizer global state is missing! Cannot proceed.'
        optim_sd = torch.load(optim_state_path)

        self._load_global_state(optim_sd)

        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        if self.mpu is None:
            logger.warn("MPU is not provided, setting tp size to 1 in checkpoint loading.")
            tp_world_size = 1
        else:
            tp_world_size = self.mpu.get_slice_parallel_world_size() if hasattr(self.mpu, "get_slice_parallel_world_size") \
                else self.mpu.get_tensor_model_parallel_world_size()


            # ep_world_size = self.mpu.get_expert_parallel_world_size() if hasattr(self.mpu, "get_expert_parallel_world_size") else 1
            # ep_rank = self.mpu.get_expert_parallel_rank() if hasattr(self.mpu, "get_expert_parallel_rank") else 0

        # print(f"load_hp_checkpoint_state_from_checkpoint_dir optim_state_path={optim_state_path} #optimizer.param_groups={len(self.optimizer.param_groups)} #optim_sd['param_groups']={len(optim_sd['param_groups'])} ep_world_size={ep_world_size} ep_rank={ep_rank} tp_world_size={tp_world_size} tp_rank={tp_rank}")

        for i, (param_group,
                loaded_param_group) in enumerate(zip(self.optimizer.param_groups, optim_sd['param_groups'])):
            # We have an assumption that all params in the same param_group have the same keys
            opt_keys = set()
            steps = []

            if is_moe_param_group(param_group):
                group_name = param_group['name']
                ep_size = groups._get_expert_parallel_world_size(group_name)
                ep_rank = groups._get_expert_parallel_rank(group_name)
            else:
                ep_size = 1
                ep_rank = 0

            print(f"load_hp_checkpoint_state_from_checkpoint_dir optim_state_path={optim_state_path} #optimizer.param_groups={len(self.optimizer.param_groups)} #optim_sd['param_groups']={len(optim_sd['param_groups'])} ep_world_size={ep_size} ep_rank={ep_rank} tp_world_size={tp_world_size} tp_rank={tp_rank}")

            lp_groups = getattr(self, lp_groups_name)
            for lp in lp_groups[i]:
                if lp._hp_mapping is not None:
                    param_name = self.param_names[lp]
                    if is_moe_param(lp):
                        expert_index = lp.num_local_experts * ep_rank + lp.local_expert_index
                        print(f"[r{dist.get_rank()}] local_expert_index={lp.local_expert_index} num_local_experts={lp.num_local_experts} param_name0={param_name} param_name1={remap_param_name(self.param_names[lp], expert_index)} ep_world_size={ep_size} ep_rank={ep_rank}")
                        param_name = remap_param_name(self.param_names[lp], expert_index)

                    print(f"load_hp_checkpoint_state_from_checkpoint_dir Loading {self.param_names[lp]} {tp_rank=} {tp_world_size=} is_moe_param(lp)={is_moe_param(lp)}")
                    step = lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, param_name), tp_rank, tp_world_size)
                    for key in lp._hp_mapping.get_optim_state_keys():
                        opt_keys.add(key)
                    steps.append(step)

            hp_param = param_group['params'][0]
            assert all(step == steps[0] for step in steps), f"Steps {steps} are not equal"
            if steps[0] is not None:
                self.optimizer.state[hp_param]['step'] = steps[0]

            map_to_flat_opt_states(hp_param, lp_groups[i], self.optimizer.state, opt_keys)

            for key, value in loaded_param_group.items():
                if key == 'params':
                    continue
                param_group[key] = value
