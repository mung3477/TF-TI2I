from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.utils.torch_utils import randn_tensor

import PIL.Image

from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

# reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3_img2img.py

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class TI2I_StableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def __init__(
        self,
        transformer,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        text_encoder_3,
        tokenizer_3,
    ):
        super().__init__(
        transformer,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        text_encoder_3,
        tokenizer_3,
        )
        self.scheduler_config = scheduler.config.copy()

    # # For img2img
    # def prepare_image_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
    #     if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
    #         raise ValueError(
    #             f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
    #         )

    #     image = image.to(device=device, dtype=dtype)

    #     batch_size = batch_size * num_images_per_prompt
    #     if image.shape[1] == self.vae.config.latent_channels:
    #         init_latents = image

    #     else:
    #         if isinstance(generator, list) and len(generator) != batch_size:
    #             raise ValueError(
    #                 f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
    #                 f" size of {batch_size}. Make sure the batch size matches the length of the generators."
    #             )

    #         elif isinstance(generator, list):
    #             init_latents = [
    #                 retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
    #                 for i in range(batch_size)
    #             ]
    #             init_latents = torch.cat(init_latents, dim=0)
    #         else:
    #             init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

    #         init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    #     if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
    #         # expand init_latents for batch_size
    #         additional_image_per_prompt = batch_size // init_latents.shape[0]
    #         init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    #     elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
    #         raise ValueError(
    #             f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
    #         )
    #     else:
    #         init_latents = torch.cat([init_latents], dim=0)

    #     shape = init_latents.shape
    #     noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    #     # get latents
    #     init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)
    #     latents = init_latents.to(device=device, dtype=dtype)

    #     return latents

    # For img2img
    def get_inverse_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :][::-1]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(num_inference_steps-1)

        return timesteps, num_inference_steps - t_start

    # For img2img
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start


    def pred_x0(self, model_output, timestep, sample):

        sample = sample.to(torch.float32)

        sigma_cur = self.scheduler.sigmas[self.scheduler.step_index]
        sigma_0 = self.scheduler.sigmas[-1]
        pred0_sample = sample + (sigma_0 - sigma_cur) * model_output

        return pred0_sample

    # For img2img
    def get_inverse_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        trunc_steps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        timesteps = torch.flip(trunc_steps, dims=[0])
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(num_inference_steps-1)

        return timesteps, num_inference_steps - t_start


    def inverse_step(
        self,
        scheduler,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)
            scheduler._step_index+=1
            print("Init inverse step index",scheduler.step_index)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = scheduler.sigmas[scheduler.step_index]
        sigma_pre = scheduler.sigmas[scheduler.step_index - 1]
        print("sigma cur", sigma.item())
        print("sigma pre", sigma_pre.item())

        # print("inv stepidx",scheduler.step_index)
        # print("inv sigma",sigma.item())
        # print("inv sigma_pre",sigma_pre.item())


        prev_sample = sample + (sigma_pre - sigma) * model_output
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        scheduler._step_index -= 1

        if not return_dict:
            return (prev_sample,)

        return prev_sample

    def __len__(self):
        return self.config.num_train_timesteps


    @torch.no_grad()
    def inverse_a_forward_ab(
        self,
        inv_prompt = "",
        prompt_a = None,
        prompt_b = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image= None,
        strength: float = 1,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        batch_size = 1

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=inv_prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        self.scheduler = FlowMatchEulerDiscreteScheduler(**self.scheduler_config)
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        input_num_inference_steps = num_inference_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_inverse_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        print("inverse timesteps", timesteps)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_image_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        print("Inversing with strength: {:.2f}".format(strength))
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(device)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.inverse_step(self.scheduler, noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()


        # Start dual branch forward

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        if debug:
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []


        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=[prompt_a, prompt_b],
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        print("prompt embed shape", prompt_embeds.shape)

        print("first latent using inversed latents, randomly initialized second latent")

        latents_a = latents.clone()
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )
        latents = torch.cat([latents, latents_a], dim=0)
        print("latents shape", latents.shape)

        print("Denoising...")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if debug:

                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if debug:
            print("return this")
            return image, {"pred_0": pred_0_list}

        if not return_dict:
            return (image,)

        return image

    @torch.no_grad()
    def inverse_forward(
        self,
        inv_prompt = "",
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image= None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):

        if debug:
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=inv_prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        self.scheduler = FlowMatchEulerDiscreteScheduler(**self.scheduler_config)
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        input_num_inference_steps = num_inference_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_inverse_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_image_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        # print("Inversion timesteps", timesteps )
        # print("Inversion sigmas", self.scheduler.sigmas)
        # print("Inversion idx", self.scheduler.step_index)


        timesteps=torch.tensor([0]+timesteps.tolist()).to(device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.inverse_step(self.scheduler, noise_pred, t, latents, return_dict=False)[0]

                if debug:
                    print("noise_pred", noise_pred.shape)
                    print("latent", latents.shape)
                    pred_0 = self.pred_x0(noise_pred, t, latents)
                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()




        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)



        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # print("Forward timesteps", timesteps )
        # print("Forward sigmas", self.scheduler.sigmas)
        # print("Forward idx", self.scheduler.step_index)
        print("Denoising...")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if debug:
                    pred_0 = self.pred_x0(noise_pred, t, latents)

                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if debug:
            print("return this")
            return image, {"pred_0": pred_0_list}

        if not return_dict:
            return (image,)

        return image


    @torch.no_grad()
    def inverse_forward_attn_control(
        self,
        attn_control_inv,
        attn_control_de,
        inv_prompt = "",
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image= None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):
        attn_control_inv(self.transformer)
        if debug:
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=inv_prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        self.scheduler = FlowMatchEulerDiscreteScheduler(**self.scheduler_config)
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        input_num_inference_steps = num_inference_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_inverse_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_image_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )
        print("LATENTS shape", latents.shape)

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        # print("Inversion timesteps", timesteps )
        # print("Inversion sigmas", self.scheduler.sigmas)
        # print("Inversion idx", self.scheduler.step_index)


        timesteps=torch.tensor([0]+timesteps.tolist()).to(device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.inverse_step(self.scheduler, noise_pred, t, latents, return_dict=False)[0]

                if debug:
                    print("noise_pred", noise_pred.shape)
                    print("latent", latents.shape)
                    pred_0 = self.pred_x0(noise_pred, t, latents)
                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()



        attn_control_de(self.transformer)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)



        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # print("Forward timesteps", timesteps )
        # print("Forward sigmas", self.scheduler.sigmas)
        # print("Forward idx", self.scheduler.step_index)
        print("Denoising...")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if debug:
                    pred_0 = self.pred_x0(noise_pred, t, latents)

                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if debug:
            print("return this")
            return image, {"pred_0": pred_0_list}

        if not return_dict:
            return (image,)

        return image
    @torch.no_grad()
    def inverse_only(
        self,
        inv_prompt = "",
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image= None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):

        if debug:
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=inv_prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        self.scheduler = FlowMatchEulerDiscreteScheduler(**self.scheduler_config)
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        input_num_inference_steps = num_inference_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_inverse_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_image_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )
        print("LATENTS shape", latents.shape)

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        # print("Inversion timesteps", timesteps )
        # print("Inversion sigmas", self.scheduler.sigmas)
        # print("Inversion idx", self.scheduler.step_index)


        timesteps=torch.tensor([0]+timesteps.tolist()).to(device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.inverse_step(self.scheduler, noise_pred, t, latents, return_dict=False)[0]

                if debug:
                    print("noise_pred", noise_pred.shape)
                    print("latent", latents.shape)
                    pred_0 = self.pred_x0(noise_pred, t, latents)
                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()
        return latents


    def denoise_2o(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
        num_inference_steps: int,
        mu,
        inverse: bool = False,
        second_order: bool = True,
        callback_on_step_end_tensor_inputs=None,
        callback_on_step_end = None,
        num_warmup_steps = 0,
        debug=False,

    ):


        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps + 1,
            self.device,
            mu=mu,
        )
        if inverse:
            sigmas = torch.flip(sigmas, [0])
            timesteps = torch.flip(timesteps, [0])
        dtype = latents.dtype
        if debug:
            pred_0_list=[]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps
                print("t_curr", t_curr.item())
                print("sigma cur", sigma_curr.item())
                print("sigma pre", sigma_prev.item())

                print(f"Denoising step {i}", t_curr,t_prev)
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t_curr.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if debug:
                    sigma_0 = sigmas[-1]
                    pred_0 = latents + (sigma_0 - sigma_curr) * noise_pred
                    pred_0=latents
                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type="pil")
                    pred_0_list.append(image)

                if (i == len(timesteps) - 2) or not second_order:
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                    latents = latents.to(dtype)
                else:
                    mid_sample = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
                    mid_sample = mid_sample.to(dtype)

                    timestep_mid = (t_curr + (t_prev-t_curr)/2).expand(latents.shape[0])
                    latent_model_input = torch.cat([mid_sample] * 2) if self.do_classifier_free_guidance else latents
                    mid_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_mid,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = mid_noise_pred.chunk(2)
                        mid_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    first_order = (mid_noise_pred - noise_pred) / ((sigma_prev - sigma_curr) / 2)
                    latents = (
                        latents
                        + (sigma_prev - sigma_curr) * noise_pred
                        + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order
                    )
                    latents = latents.to(dtype)

                # if latents.dtype != latents_dtype:
                #     if torch.backends.mps.is_available():
                #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                #         latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t_curr, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(sigmas) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if debug:
            return latents, {"pred_0": pred_0_list}
        return latents, None


    # def denoise_2o(
    #     self,
    #     latents: torch.Tensor,
    #     pooled_prompt_embeds: torch.Tensor,
    #     prompt_embeds: torch.Tensor,
    #     sigmas: torch.Tensor,
    #     timesteps: torch.Tensor,
    #     num_inference_steps: int,
    #     inverse: bool = False,
    #     second_order: bool = True,
    #     callback_on_step_end_tensor_inputs=None,
    #     callback_on_step_end = None,
    #     num_warmup_steps = 0,
    #     debug=False
    # ):
    #     if inverse:
    #         sigmas = torch.flip(sigmas, [0])

    #     dtype = latents.dtype
    #     if debug:
    #         pred_0_list=[]
    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         for i, (sigma_curr, sigma_prev) in enumerate(zip(sigmas[:-1], sigmas[1:])):
    #             t_curr = (sigma_curr * self.scheduler.config.num_train_timesteps)
    #             t_prev = (sigma_prev * self.scheduler.config.num_train_timesteps)
    #             print("t_curr", t_curr.item())
    #             print("sigma cur", sigma_curr.item())
    #             print("sigma pre", sigma_prev.item())

    #             print(f"Denoising step {i}", t_curr,t_prev)
    #             if self.interrupt:
    #                 continue

    #             # expand the latents if we are doing classifier free guidance
    #             latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
    #             # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    #             timestep = t_curr.expand(latent_model_input.shape[0])

    #             noise_pred = self.transformer(
    #                 hidden_states=latent_model_input,
    #                 timestep=timestep,
    #                 encoder_hidden_states=prompt_embeds,
    #                 pooled_projections=pooled_prompt_embeds,
    #                 joint_attention_kwargs=self.joint_attention_kwargs,
    #                 return_dict=False,
    #             )[0]

    #             # perform guidance
    #             if self.do_classifier_free_guidance:
    #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #                 noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

    #             if debug:
    #                 sigma_0 = sigmas[-1]
    #                 pred_0 = latents + (sigma_0 - sigma_curr) * noise_pred
    #                 pred_0=latents
    #                 pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
    #                 image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
    #                 image = self.image_processor.postprocess(image, output_type="pil")
    #                 pred_0_list.append(image)

    #             if (i == len(sigmas) - 1) or not second_order:
    #                 latents = latents + (sigma_prev - sigma_curr) * noise_pred
    #                 latents = latents.to(dtype)
    #             else:
    #                 mid_sample = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
    #                 mid_sample = mid_sample.to(dtype)

    #                 timestep_mid = (t_curr + (t_prev-t_curr)/2).expand(latents.shape[0])
    #                 mid_noise_pred = self.transformer(
    #                     hidden_states=latent_model_input,
    #                     timestep=timestep_mid,
    #                     encoder_hidden_states=prompt_embeds,
    #                     pooled_projections=pooled_prompt_embeds,
    #                     joint_attention_kwargs=self.joint_attention_kwargs,
    #                     return_dict=False,
    #                 )[0]

    #                 if self.do_classifier_free_guidance:
    #                     noise_pred_uncond, noise_pred_text = mid_noise_pred.chunk(2)
    #                     mid_noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

    #                 first_order = (mid_noise_pred - noise_pred) / ((sigma_prev - sigma_curr) / 2)
    #                 latents = (
    #                     latents
    #                     + (sigma_prev - sigma_curr) * noise_pred
    #                     + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order
    #                 )
    #                 latents = latents.to(dtype)

    #             # if latents.dtype != latents_dtype:
    #             #     if torch.backends.mps.is_available():
    #             #         # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
    #             #         latents = latents.to(latents_dtype)

    #             if callback_on_step_end is not None:
    #                 callback_kwargs = {}
    #                 for k in callback_on_step_end_tensor_inputs:
    #                     callback_kwargs[k] = locals()[k]
    #                 callback_outputs = callback_on_step_end(self, i, t_curr, callback_kwargs)

    #                 latents = callback_outputs.pop("latents", latents)
    #                 prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
    #                 negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
    #                 negative_pooled_prompt_embeds = callback_outputs.pop(
    #                     "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
    #                 )

    #             # call the callback, if provided
    #             if i == len(sigmas) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()

    #             if XLA_AVAILABLE:
    #                 xm.mark_step()

    #     if debug:
    #         return latents, {"pred_0": pred_0_list}
    #     return latents, None

    @torch.no_grad()
    def inverse_forward_2o(
        self,
        inv_prompt = "",
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image= None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=inv_prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        self.scheduler = FlowMatchEulerDiscreteScheduler(**self.scheduler_config)
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        input_num_inference_steps = num_inference_steps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, input_num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_inverse_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_image_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        # print("Inversion timesteps", timesteps )
        # print("Inversion sigmas", self.scheduler.sigmas)
        # print("Inversion idx", self.scheduler.step_index)

        print("Inversing with strength: {:.2f}".format(strength))
        latents, debug_dict = self.denoise_2o(inverse=True, second_order=True, debug=debug,
                                  timesteps=timesteps,
                                  latents=latents,
                                  mu=mu,
                                  sigmas=self.scheduler.sigmas,
                                  pooled_prompt_embeds=pooled_prompt_embeds,
                                  prompt_embeds=prompt_embeds,
                                  num_inference_steps=num_inference_steps,
                                  callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,num_warmup_steps=num_warmup_steps)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        latents, debug_dict = self.denoise_2o(inverse=False, second_order=True, debug=debug,
                                  latents=latents,
                                  timesteps=timesteps,
                                  mu=mu,
                                  sigmas = self.scheduler.sigmas,
                                  pooled_prompt_embeds=pooled_prompt_embeds,
                                  prompt_embeds=prompt_embeds,
                                  num_inference_steps=num_inference_steps,
                                  callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,num_warmup_steps=num_warmup_steps)

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        return image, debug_dict

        if not return_dict:
            return (image,)

        return image


    # For img2img
    def prepare_image_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)
        latents = init_latents.to(device=device, dtype=dtype)

        return latents

    # For img2img
    def prepare_image_latents_nonoise(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        latents = init_latents.to(device=device, dtype=dtype)
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents


    # For img2img
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start


    @torch.no_grad()
    def img2img_ab(
        self,
        image= None,
        prompt_a = None,
        prompt_b = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        prompt=[prompt_a, prompt_b]
        batch_size = 1

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        latent_image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # 5. Prepare latent variables

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        image_latents = self.prepare_image_latents(
            latent_image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )



        latents = torch.cat([latents, image_latents], dim=0)

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)


        if debug:
            print("time steps", timesteps)
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []
            xt_list=[]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                image_latents = self.prepare_image_latents(
                    latent_image,
                    t.unsqueeze(0),
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )

                latents = torch.cat([latents[:1], image_latents], dim=0)

                if debug:
                    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                    image = self.vae.decode(latents, return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    xt_list.append(image)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if debug:
                    pred_0 = self.pred_x0(noise_pred, t, latents)

                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if debug:
            print("return this")
            return image, {"pred_0": pred_0_list, "xt": xt_list}

        if not return_dict:
            return (image,)

        return image



    @torch.no_grad()
    def img2img_multi(
        self,
        images: list = [],
        prompt_a = None,
        prompt_b: list = [],
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        prompt=[prompt_a] + prompt_b
        batch_size = 1

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # 5. Prepare latent variables

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        for sub_image in images:
            sub_image = self.image_processor.preprocess(sub_image, height=height, width=width)
            image_latents = self.prepare_image_latents(
                sub_image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )
            latents = torch.cat([latents, image_latents], dim=0)


        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)


        if debug:
            print("time steps", timesteps)
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []
            xt_list=[]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latents = latents[:1]


                for sub_image in images:
                    # 3. Preprocess image
                    sub_image = self.image_processor.preprocess(sub_image, height=height, width=width)
                    image_latents = self.prepare_image_latents(
                        sub_image,
                        t.unsqueeze(0),
                        batch_size,
                        num_images_per_prompt,
                        prompt_embeds.dtype,
                        device,
                        generator,
                    )
                    latents = torch.cat([latents, image_latents], dim=0)

                if debug:
                    xt_latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

                    image = self.vae.decode(xt_latents, return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    xt_list.append(image)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if debug:
                    pred_0 = self.pred_x0(noise_pred, t, latents)

                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if debug:
            print("return this")
            return image, {"pred_0": pred_0_list,"xt": xt_list}

        if not return_dict:
            return (image,)

        return image


    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        image= None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        debug=False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image, height=height, width=width)

        # 4. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            print("This case use dynamic shifting")
            image_seq_len = (int(height) // self.vae_scale_factor // self.transformer.config.patch_size) * (
                int(width) // self.vae_scale_factor // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_image_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)


        if debug:
            print("time steps", timesteps)
            print("sigmas", self.scheduler.sigmas)
            print("DEBUG MODE ON")
            pred_0_list = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if debug:
                    pred_0 = self.pred_x0(noise_pred, t, latents)

                    pred_0 = (pred_0/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    image = self.vae.decode(pred_0.to(self.vae.dtype), return_dict=False)[0]
                    image = self.image_processor.postprocess(image, output_type=output_type)
                    pred_0_list.append(image)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if debug:
            print("return this")
            return image, {"pred_0": pred_0_list}

        if not return_dict:
            return (image,)

        return image
