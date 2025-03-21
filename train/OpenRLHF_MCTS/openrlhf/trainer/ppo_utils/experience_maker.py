import re
import time
import random
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import ray
import json
import requests
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

from vllm import SamplingParams
from .mcts_sampling import search_for_answers, remove_boxed, last_boxed_only_string, is_equiv

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    rewards: Optional[list]


@dataclass
class Outputs:
    prompt_token_ids: list
    token_ids: tuple
    rewards: list


class VLLMGenerator(ABC):
    def __init__(self, vllm_engines):
        self.vllm_engines = vllm_engines
    
    @torch.no_grad()
    def generate_output(self, all_prompts: List[str], sampling_params):
        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            if prompts:
                all_output_refs.append(
                    llm.generate.remote(prompts, sampling_params=sampling_params)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])
        return all_outputs
    
    def generate_next_action(self, traces: List[str], **kwargs):
        sampling_params = SamplingParams(
            n=1,
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", 512),
            logprobs=2,
            stop=['</next_step>', '</clarify>', '</analysis>', '</subquestion>', '</verify>', '</refine>', '</output>', '</direct_answer>']
        )

        outputs = self.generate_output(traces, sampling_params)

        return [o.outputs[0].text.strip() for o in outputs]

    def predict_next_action(self, traces: List[str], **kwargs):
        sampling_params = SamplingParams(
            n=1,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1),
            logprobs=kwargs.get("logprobs", 12),
            stop=['</next_step>', '</clarify>', '</analysis>', '</subquestion>', '</verify>', '</refine>', '</output>', '</direct_answer>']
        )
        outputs = self.generate_output(traces, sampling_params)

        all_topk_actions = []

        for output in outputs:
            o = output.outputs[0].logprobs[0]
            # topk_tokens = []

            all_actions = ["clarify", "analysis", "next_step", "direct_answer", "subquestion", "verify", "refine", "output"]
            topk_actions = []
            for token_id, logprob_obj in o.items():
                action = logprob_obj.decoded_token.strip()
                if action in "clarify" and "clarify" in all_actions:
                    topk_actions.append("clarify")
                    all_actions.remove("clarify")
                elif action in "analysis" and "analysis" in all_actions:
                    topk_actions.append("analysis")
                    all_actions.remove("analysis")
                elif action in "next_step" and "next_step" in all_actions:
                    topk_actions.append("next_step")
                    all_actions.remove("next_step")
                elif action in "direct_answer" and "direct_answer" in all_actions:
                    topk_actions.append("direct_answer")
                    all_actions.remove("direct_answer")
                elif action in "subquestion" and "subquestion" in all_actions:
                    topk_actions.append("subquestion")
                    all_actions.remove("subquestion")
                elif action in "verify" and "verify" in all_actions:
                    topk_actions.append("verify")
                    all_actions.remove("verify")
                elif action in "refine" and "refine" in all_actions:
                    topk_actions.append("refine")
                    all_actions.remove("refine")
                elif action in "output" and "output" in all_actions:
                    topk_actions.append("output")
                    all_actions.remove("output")

            if topk_actions == []:
                topk_actions = all_actions

            all_topk_actions.append(topk_actions)
        
        return all_topk_actions


    def generate_output_withid(self, all_prompt_token_ids: list, sampling_params):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])
        return all_outputs

class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_solutions: Union[str, List[str]], all_ids: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        experiences = []
        for samples in tqdm(
            self.generate_samples(all_prompts, all_solutions, all_ids, **generate_kwargs),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
                use_prm=args.use_prm,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            elif self.advantage_estimator == "grpo":
                pass
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.info["reward"] = experience.info["avg_reward"]
            del experience.info["avg_reward"]
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_solutions: List[str], all_ids: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.vllm_generator = VLLMGenerator(vllm_engines)

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], all_solutions: Union[str, List[str]], all_ids: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, all_solutions, all_ids,  **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_solutions: List[str], all_ids: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_solutions, all_ids, **generate_kwargs)
        
        if self.strategy.args.sampling_method == "mcts":
            # return self._generate_naive_mcts(all_prompts, all_solutions, all_ids, **generate_kwargs)
            return self._generate_mcts(all_prompts, all_solutions, all_ids, **generate_kwargs)
        elif self.strategy.args.sampling_method == "normal":
            return self._generate_vllm(all_prompts, all_solutions, all_ids, **generate_kwargs)
        elif self.strategy.args.sampling_method == "beam":
            return self._generate_beam(all_prompts, all_solutions, all_ids, **generate_kwargs)
        else:
            raise NotImplementedError
        

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        step_rewards = samples.rewards

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        
        # sequences_list = []
        # offset = 0
        # tokens_list = sequences_cpu.tolist()[0]
        # for length in packed_seq_lens:
        #     sequences_list.append(tokens_list[offset : offset + length])
        #     offset += length
        # queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
        # step_rewards = []
        # for q in queries:
        #     import random
        #     step_rewards.append(random.choice([0.6, 0.7, 0.8, 0.9, 0.3, 0.5]))
        def find_all_step_positions(lst):
            # pattern = ['>\n', '<']
            pattern = [397, 27]  # qwen2.5 and llama3.1 are same!
            positions = []
            for i in range(len(lst) - 1):
                if lst[i:i+2] == pattern:
                    positions.append(i)
            return positions
        
        all_rewards = []
        avg_rewards = []
        if self.strategy.args.use_prm:
            start_idx = 0
            for i in range(len(num_actions)):
                rewards = torch.zeros(num_actions[i], device=device, dtype=torch.float16)
                answer_offset = packed_seq_lens[i] - num_actions[i]
                answer_seq = sequences[0][start_idx+answer_offset:packed_seq_lens[i]+start_idx]
                start_idx += packed_seq_lens[i]
                step_reward_pos = find_all_step_positions(answer_seq.tolist())
                for pos, step_reward in zip(step_reward_pos, step_rewards[i]):
                    rewards[pos] = step_reward
                if len(step_rewards[i]) == len(step_reward_pos) + 1:
                    rewards[-1] = step_rewards[i][-1]
                else:
                    rewards[-1] = -0.2
                    # if num_actions[i] == self.strategy.args.generate_max_len:
                    #     rewards[-1] = -0.5
                    # else:
                    #     try:
                    #         rewards[-1] = step_rewards[i][-1]
                    #     except:
                    #         rewards[-1] = -1
                    # breakpoint()
                all_rewards.append(rewards)

                outcome_reward = sum(step_rewards[i])/len(step_rewards[i])
                avg_rewards.append(outcome_reward)
            avg_rewards = torch.tensor(avg_rewards).to(device)
        else:
            for i in range(len(num_actions)):
                outcome_reward = sum(step_rewards[i])/len(step_rewards[i])
                all_rewards.append(outcome_reward)
            all_rewards = torch.tensor(all_rewards).to(device)
            avg_rewards = all_rewards
                

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start

        base_action_log_probs, value = ref_values[0], ref_values[1]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        # rewards = [r.to(device) for r in rewards]
        # r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        # all_rewards = torch.tensor(step_rewards).to(device)
        r = all_rewards

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "avg_reward": avg_rewards,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def extract_question(self, prompt: str):
        pattern = r"<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            question = match.group(1).strip()
            new_prompt = question
            return new_prompt
        else:
            return prompt


    def _generate_vllm(self, all_prompts: List[str], all_solutions: List[str], all_ids: List[str], **kwargs) -> List[Samples]:
        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_solutions = sum([[solution] * args.n_samples_per_prompt for solution in all_solutions], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        all_outputs = self.vllm_generator.generate_output_withid(all_prompt_token_ids, sampling_params)
        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            solutions = all_solutions[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                rewards = []
                for i, (output, solution) in enumerate(zip(outputs, solutions)):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    # output_match = re.search(r"<output>(.*?)</output>", output.outputs[0].text, re.DOTALL)
                    # if output_match:
                    #     model_answer = output_match.group(1)
                    # else:
                    #     model_answer = ""
                    model_answer = output.outputs[0].text
                    if is_equiv(remove_boxed(last_boxed_only_string(model_answer)), remove_boxed(last_boxed_only_string(solution))):
                        rewards.append([random.choice([0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95, 0.98])])
                    #elif "\\boxed" in output.outputs[0].text:
                        #rewards.append([random.choice([0.3, 0.35, 0.32, 0.38, 0.4, 0.42, 0.28, 0.45])])
                    else:
                        rewards.append([random.choice([0.2, 0.15, 0.22, 0.12, 0.06, 0.18, 0.1, 0.25])])

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        rewards=rewards
                    )
                )
        return samples_list
    
    def _search_for_answers(self, prompt: str, gt_answer: str, question_id: str):
        args = self.strategy.args
        outputs = []
        rollout_outputs, step_rewards = search_for_answers(prompt, args.n_samples_per_prompt, question_id.replace('/', '_'), gt_answer, self.vllm_generator, args.save_tree, args.search_depth)

        for out, rew in zip(rollout_outputs, step_rewards):
            outputs.append(
                Outputs(prompt_token_ids=self.tokenizer.encode(prompt, add_special_tokens=False), token_ids=tuple(self.tokenizer.encode(out, add_special_tokens=False)+[128009]), rewards=rew)
            )
        return outputs

    def _generate_mcts(self, all_prompts: List[str], all_solutions: List[str], all_ids: List[str], **kwargs) -> List[Samples]:
        args = self.strategy.args
        all_outputs = []
        ## 并发搜索
        # with ThreadPoolExecutor(max_workers=16) as executor:
        #     results = executor.map(self._search_for_answers, all_prompts, all_solutions, all_ids)

        # all_outputs = sum(list(results), [])
        for prompt, sol, id in zip(all_prompts, all_solutions, all_ids):
            rollout_outputs, step_rewards = search_for_answers(prompt, args.n_samples_per_prompt, id.replace('/', '_'), sol, self.vllm_generator, args.save_tree, args.search_depth)

            for out, rew in zip(rollout_outputs, step_rewards):
                all_outputs.append(
                    Outputs(prompt_token_ids=self.tokenizer.encode(prompt, add_special_tokens=False), token_ids=tuple(self.tokenizer.encode(out, add_special_tokens=False)+[151645]), rewards=rew)
                )
            # breakpoint()

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                raise NotImplementedError
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                rewards = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    rewards.append(output.rewards)

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        rewards=rewards
                    )
                )
        return samples_list

    def _generate_naive_mcts(self, all_prompts: List[str], all_solutions: List[str], all_ids: List[str], **kwargs) -> List[Samples]:
        args = self.strategy.args
        prefix_sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # breakpoint()
        all_layer1_prompts = []
        layer1_begin_actions = ["<clarify>\n", "<analysis>\n", "<subquestion>\n", "<next_step>\n"]
        layer1_end_actions = ["\n</clarify>\n", "\n</analysis>\n", "\n</subquestion>\n", "\n</next_step>\n"]
        all_prefix_prompt = []
        for prompt in all_prompts:
            all_layer1_prompts += [prompt + a for a in layer1_begin_actions]
        all_layer1_outputs = self.vllm_generator.generate_next_action(all_layer1_prompts)

        for i, (layer1_prompt, layer1_output) in enumerate(zip(all_layer1_prompts, all_layer1_outputs)):
            all_layer1_outputs[i] = layer1_prompt + layer1_output + layer1_end_actions[i%4]
        
        
        layer2_end_actions = ["\n</analysis>\n", "\n</subquestion>\n", "\n</next_step>\n", "\n</direct_answer>\n", "\n</subquestion>\n", "\n</next_step>\n", "\n</direct_answer>\n", "\n</next_step>\n", "\n</direct_answer>\n", "\n</next_step>\n", "\n</direct_answer>\n", "\n</subquestion>\n"]
        all_layer2_prompts = []
        for i, prompt in enumerate(all_layer1_outputs):
            if i%4 == 0:
                all_layer2_prompts += [prompt + a for a in ["<analysis>\n", "<subquestion>\n", "<next_step>\n", "<direct_answer>\n"]]
            elif i%4 == 1:
                all_layer2_prompts += [prompt + a for a in ["<subquestion>\n", "<next_step>\n", "<direct_answer>\n"]]
            elif i%4 == 2:
                all_layer2_prompts += [prompt + a for a in ["<next_step>\n", "<direct_answer>\n"]]
            elif i%4 == 3:
                all_layer2_prompts += [prompt + a for a in ["<next_step>\n", "<direct_answer>\n", "<subquestion>\n"]]
        all_layer2_outputs = self.vllm_generator.generate_next_action(all_layer2_prompts)

        for i, (layer2_prompt, layer2_output) in enumerate(zip(all_layer2_prompts, all_layer2_outputs)):
            all_layer2_outputs[i] = layer2_prompt + layer2_output + layer2_end_actions[i%12]
        
        for i, prompt in enumerate(all_layer2_outputs):
            if i%12 == 3:
                all_prefix_prompt += [prompt + "<output>\n"]
                all_prefix_prompt += [prompt + "<verify>\n"]
            elif i%12 == 7:
                all_prefix_prompt += [prompt + "<next_step>\n"]
                all_prefix_prompt += [prompt + "<subquestion>\n"]
            elif i%12 == 9:
                all_prefix_prompt += [prompt + "<next_step>\n"]
                all_prefix_prompt += [prompt + "<subquestion>\n"]
            elif i%12 == 10:
                all_prefix_prompt += [prompt + "<output>\n"]
                all_prefix_prompt += [prompt + "<verify>\n"]
            else:
                all_prefix_prompt += [prompt]
        
        _all_outputs = self.vllm_generator.generate_output(all_prefix_prompt, prefix_sampling_params)
        all_solutions = sum([[solution] * args.n_samples_per_prompt for solution in all_solutions], [])
        # print(_all_outputs)

        all_outputs = []
        for output, sol in zip(_all_outputs, all_solutions):
            position = self._seperate_prompt(output.prompt_token_ids)
            prompt_token_ids, rest_token_ids = output.prompt_token_ids[:position], output.prompt_token_ids[position:]
            output_token_ids = tuple(rest_token_ids) + output.outputs[0].token_ids
            model_answer = output.prompt + output.outputs[0].text
            rew = self.get_reward(model_answer, sol)
            with open(f"{args.save_tree}/generate.jsonl", 'a+', encoding="utf-8") as f:
                result = json.dumps({
                    "output": model_answer,
                    "reward": rew,
                }, ensure_ascii=False)
                f.write(result + '\n')
            all_outputs.append(
                Outputs(prompt_token_ids=prompt_token_ids, token_ids=output_token_ids, rewards=rew)
            )

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                raise NotImplementedError
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                rewards = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    rewards.append(output.rewards)

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        rewards=rewards
                    )
                )
        return samples_list

    def _seperate_prompt(self, token_ids):
        if "qwen" in self.strategy.args.pretrain:
            sublist = [151644, 77091, 198]
        elif "llama" in self.strategy.args.pretrain:
            sublist = [78191, 128007, 271]
        else:
            raise NotImplementedError
        
        n = len(token_ids)
        m = len(sublist)

        for i in range(n - m + 1):
            if token_ids[i:i + m] == sublist:
                return i+m
        return None

    def has_chinese(self, s):
        return bool(re.search(r'[\u4e00-\u9fff]', s))

    def get_reward(self, input, solution):
        if self.strategy.args.use_prm:
            if "qwen" in self.strategy.args.pretrain:
                question_pattern = r"<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)(?=\Z)"
            elif "llama" in self.strategy.args.pretrain:
                question_pattern = r"<\|eot_id\|><\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|><|start_header_id|>assistant<|end_header_id|>\n\n(.*?)(?=\Z)"
            else:
                raise NotImplementedError
            match = re.search(question_pattern, input, re.DOTALL)
            # print("Input:")
            # print(input, match)
            question, raw_solution = match.group(1), match.group(2)
            solutions = ""
            pattern = r'<(analysis|subquestion|ost|subanswer|clarify|direct_answer|output|verify|refine)>(.*?)</\1>'
            matches = re.findall(pattern, raw_solution, re.DOTALL)
            if not matches:
                solutions = "Step 1: " + raw_solution.strip() + " ки"
            # 创建一个字典来保存不同标签的内容
            i = 1
            last_step = solutions
            for tag, content in matches:
                clean_content = content.strip()
                last_step = clean_content
                solutions += f"Step {i}: {clean_content} ки\n"
                i += 1
            solutions = solutions.strip()
            input_for_prm = f"{question} {solutions}"

            url = self.strategy.args.remote_rm_url[0]
            data = {"input": input_for_prm}
            try:
                resp = requests.post(url, json=data)
                rewards = resp.json()["score"]
            except Exception as e:
                print(e)
                print(f"Exception input:\n{input_for_prm}")
                print(f"Response:\n{resp.text}")
                if i > 1:
                    i -= 1
                rewards = [0.2] * i
            
            if is_equiv(remove_boxed(last_boxed_only_string(last_step)), remove_boxed(last_boxed_only_string(solution))) and not self.has_chinese(input):
                is_correct = 0.6 + 0.4*rewards[-1]
            else:
                is_correct = 0.4*rewards[-1]
            
            # for i, r in enumerate(rewards):
            #     if r >=0.75:
            #         rewards[i] = 0.2
            #     elif r <0.4:
            #         rewards[i] = -0.1
            #     else:
            #         rewards[i] = 0

            rewards[-1] = is_correct
        else:
            # output_match = re.search(r"<output>(.*?)</output>", input, re.DOTALL)
            # if output_match:
            #     model_answer = output_match.group(1)
            # else:
            #     model_answer = ""
            model_answer = input
            if is_equiv(remove_boxed(last_boxed_only_string(model_answer)), remove_boxed(last_boxed_only_string(solution))) and not self.has_chinese(input):
                rewards = [1.0]
            else:
                rewards = [0.0]
        return rewards[-1]


    def _generate_beam(self, all_prompts: List[str], all_solutions: List[str], all_ids: List[str], **kwargs) -> List[Samples]:
        pass

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
