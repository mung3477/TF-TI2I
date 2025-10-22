from typing import Optional

import torch.nn.functional as F
import torchvision.transforms as T
import math
import torch

from src.attention.visualization import ScalableAttentionProcessor, SavedAttnMapType


def compute_attn_weight(query, key, value=None, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None) -> torch.Tensor:
    device = query.device  # 確保所有張量在同一設備上
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)  # 確保 attn_bias 在同一設備上
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)  # 確保 temp_mask 在同一設備上
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_mask = attn_mask.to(device)  # 確保 attn_mask 在同一設備上
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_bias = attn_bias.to(attn_weight.dtype)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight


def process_attn_map_otsu(data, blur_func=T.GaussianBlur(kernel_size=5, sigma=1)):
    data= data.view(1, 1, 64, 64)
    # smooth
    dtype=data.dtype
    data = blur_func(data).flatten().float()
    # normalize
    min_val, max_val = data.min(), data.max()
    # edge_case
    if min_val==max_val:
        print("! edge_case in process attn otsu")
        return torch.ones_like(data).to(dtype)
    data = (data - min_val) / (max_val - min_val+1e-9)
    min_val, max_val = data.min(), data.max()

    bins = 256  # 直方圖 bin 數
    bin_width = (data.max() - data.min()) / bins

    # 計算直方圖（僅用 torch.histc）
    hist = torch.histc(data, bins=bins, min=min_val.item(), max=max_val.item())

    # 建立 bin 中心點
    bin_centers = torch.linspace(min_val, max_val, bins, device=data.device)

    # 總像素數
    total_pixels = data.numel()

    # 計算累積直方圖
    weight_background = torch.cumsum(hist, dim=0)  # 背景權重
    weight_foreground = total_pixels - weight_background  # 前景權重

    # 避免除以 0
    valid_mask = (weight_background > 0) & (weight_foreground > 0)

    # 計算累積均值
    sum_total = torch.sum(bin_centers * hist)  # 總強度
    sum_background = torch.cumsum(bin_centers * hist, dim=0)  # 背景累積強度
    mean_background = sum_background / weight_background.clamp(min=1)  # 背景均值
    mean_foreground = (sum_total - sum_background) / weight_foreground.clamp(min=1)  # 前景均值

    # 計算類間方差
    between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

    # 只考慮有效值，找最大類間方差對應的 threshold
    between_class_variance[~valid_mask] = 0
    best_threshold_idx = torch.argmax(between_class_variance)  # 找最大變異數的 index
    best_threshold = bin_centers[best_threshold_idx]  # 找到對應的 threshold 值

    binary_data = (data > best_threshold).float()
    return binary_data.to(dtype)

def refer_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None,  attn_reweight=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    # torch.Size([1, 38, 4429, 4762])
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if attn_reweight is not None:
        attn_weight[:, :, 4096:, :4096] *= attn_reweight[0] # ref2self
        attn_weight[:, :, 4096:, 4096:4429] *= attn_reweight[1] # ref2ref

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

def WTA_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False,
        wta_parameter={"wta_weight":[],
                       "cross2ref":False}) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight torch.Size([1, 38, 4429, 4762])

    ##############################################################
    # mean accross all the batch and heads
    mean_atn_weight = torch.mean(attn_weight, dim=[0,1])
    refer_score=[]

    wta_weight = wta_parameter["wta_weight"]
    wta_shift = wta_parameter["wta_shift"]
    # mean of contextual attention weights (L2Ctx); noise latent consists of 4096 tokens
    abs_global_contextual_mean = torch.abs(torch.mean(mean_atn_weight[:,4096:]))
    if "wta_cross" in wta_parameter and wta_parameter["wta_cross"]:
        for ref_idx in range(len(wta_weight)):
            score_shift = wta_shift[ref_idx]
            total_score = torch.mean(mean_atn_weight[:,4096+333*ref_idx:4429+333*ref_idx], dim=-1)*wta_weight[ref_idx]+score_shift*abs_global_contextual_mean
            # print("total_score",total_score.shape)
            refer_score.append(total_score)
        refer_score = torch.stack(refer_score, dim=1)
        refers_argmax = torch.argmax(refer_score, dim=1)

        wta_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        # keep winner and set others to -inf
        for ref_idx in range(len(wta_weight)):
            rows_with_ref_idx = torch.nonzero(refers_argmax == ref_idx).squeeze()  # (N,)
            # refer_before and after to be -inf

            wta_bias[rows_with_ref_idx,4096:4096+333*ref_idx] = float("-inf")
            wta_bias[rows_with_ref_idx,4429+333*ref_idx:] = float("-inf")
        if wta_parameter["cross2ref"]==False:
            wta_bias[4096:,4429:] = float("-inf")

    else:
        # we first measure the attention score of references to a given vision token
        for ref_idx in range(len(wta_weight)):
            score_shift = wta_shift[ref_idx]
            # 4096(noise) + 333(text prompt) + 333(ref1) + 333(ref2) + ...
            total_score = torch.mean(mean_atn_weight[:,4429+333*ref_idx:4762+333*ref_idx], dim=-1)*wta_weight[ref_idx]+score_shift*abs_global_contextual_mean
            refer_score.append(total_score)
        refer_score = torch.stack(refer_score, dim=1)
        refers_argmax = torch.argmax(refer_score, dim=1)

        # (4096(noise) + 333(text prompt), 4096(noise) + 333(text prompt) + 333(ref1) + 333(ref2) + ...)
        wta_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        # and selectively keep only the contextual tokens with highest attention score to minimize distribution disturbances.
        # keep winner and set others to -inf;
        for ref_idx in range(len(wta_weight)):
            # rows that should refer to ref_idx context
            rows_with_ref_idx = torch.nonzero(refers_argmax == ref_idx).squeeze()  # (N,)
            # set before and after ref_idx th context to be -inf
            wta_bias[rows_with_ref_idx,4429:4429+333*ref_idx] = float("-inf")
            wta_bias[rows_with_ref_idx,4762+333*ref_idx:] = float("-inf")
        if wta_parameter["cross2ref"]==False:
            wta_bias[4096:,4429:] = float("-inf")

    if "global_shift_idxs" in wta_parameter:
        global_shift_idxs = wta_parameter["global_shift_idxs"]
        global_shift_weights = wta_parameter["global_shift_weights"]
        for shift_idx, shift_weight in zip(global_shift_idxs, global_shift_weights):
            attn_weight[:,:,:,4429+333*ref_idx:4762+333*ref_idx] += shift_weight*abs_global_contextual_mean

    attn_weight += attn_bias
    attn_weight += wta_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    ####################################################################################
    return attn_weight @ value

class TI2I_JointAttnProcessor2_0_multi(ScalableAttentionProcessor):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer=-1, contextual_replace=True, replace_start=0, replace_end=-1,operator="concat",
                wta_control_signal={},
                ref_control_signal={
                    "on":False,
                    "ref_idxs":[],
                    "control_type":"main_context",
                },
        ###########################
                height=64,
                width=64,
                save_attn_maps: Optional[SavedAttnMapType] = None
        ):

        super().__init__(height=height, width=width, save_attn_maps=save_attn_maps)
        ###########################

        self.step=-1
        self.layer=layer
        self.wta_control_signal=wta_control_signal

        self.ref_control_signal=ref_control_signal
        self.contextual_replace = contextual_replace
        self.replace_start = replace_start
        self.replace_end = replace_end
        self.operator = operator


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask = None,

        ############################################################
        height: int = None,
        timestep: Optional[torch.Tensor] = None,
        ############################################################

        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        self.step+=1
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            if self.contextual_replace:
                bs = len(encoder_hidden_states_query_proj)//2

                if self.operator == "mean":
                    encoder_hidden_states_query_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_query_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
                    encoder_hidden_states_key_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_key_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
                    encoder_hidden_states_value_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_value_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
                elif self.operator == "head_wise":
                    qs=[]
                    ks=[]
                    vs=[]
                    for i in range(bs):
                        q_split = torch.chunk(encoder_hidden_states_query_proj[bs+i], bs, dim=0)
                        k_split = torch.chunk(encoder_hidden_states_key_proj[bs+i], bs, dim=0)
                        v_split = torch.chunk(encoder_hidden_states_value_proj[bs+i], bs, dim=0)

                        qs.append(q_split[i])
                        ks.append(k_split[i])
                        vs.append(v_split[i])
                    encoder_hidden_states_query_proj[bs] = torch.cat(qs, dim=0)
                    encoder_hidden_states_key_proj[bs] = torch.cat(ks, dim=0)
                    encoder_hidden_states_value_proj[bs] = torch.cat(vs, dim=0)
                elif self.operator == "concat":
                    # Side branch

                    # Concat text embeddings of positive text prompts
                    # (4, 38(heads), 333, 64(head_dim)) => (1, 38, 666, 64)
                    main_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    main_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)

                    # Concat queries from latents originally used for generation
                    # (1, 38, 4096, 64) + (1, 38, 333, 64) => (1, 38, 4429, 64)
                    main_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)

                    # Concat keys and values from image latent originally used for generation and all the positive text embeddings, including the one corresponds to the reference image
                    main_key = torch.cat([key[bs:bs+1], main_encoder_hidden_states_key_proj], dim=2)
                    main_value = torch.cat([value[bs:bs+1], main_encoder_hidden_states_value_proj], dim=2)

                    if not self.wta_control_signal["on"]:
                        main_hidden_states = F.scaled_dot_product_attention(main_query, main_key, main_value, dropout_p=0.0, is_causal=False)
                    else:
                        self.wta_parameter = self.wta_control_signal["hyper_parameter"]
                        main_hidden_states = WTA_scaled_dot_product_attention(main_query, main_key, main_value, dropout_p=0.0, is_causal=False, wta_parameter=self.wta_parameter)

                    if self.ref_control_signal['on']:
                        # If define control layer but not use it
                        if "control_layers" in self.ref_control_signal and self.layer not in self.ref_control_signal["control_layers"]:
                            pass
                        else:
                            ref_control_idxs = self.ref_control_signal['ref_idxs']
                            ref_control_prompts = self.ref_control_signal['ref_prompts']
                            ref_control_hyper_parameter=self.ref_control_signal["hyper_parameter"]
                            ref_control_hidden_states=[]
                            for instance_idx, (ref_ref_idx, ref_control_prompt) in enumerate(zip(ref_control_idxs, ref_control_prompts)):
                                # concat reference latent with the text embedding of the reference prompt
                                batch_ref_idx = bs + ref_ref_idx + 1
                                ref_query = torch.cat([query[batch_ref_idx : batch_ref_idx + 1],
                                                    encoder_hidden_states_query_proj[batch_ref_idx : batch_ref_idx + 1]], dim=2)
                                ref_key = torch.cat([key[batch_ref_idx : batch_ref_idx + 1],
                                                    encoder_hidden_states_key_proj[batch_ref_idx : batch_ref_idx + 1]], dim=2)
                                ref_value = torch.cat([value[batch_ref_idx : batch_ref_idx + 1],
                                                    encoder_hidden_states_value_proj[batch_ref_idx : batch_ref_idx + 1]], dim=2)
                                # I2Ctx
                                # (1, 38(head), 4429(image + text), 4429((image + text))) -> (1, 38, 4096, 333(text prompt))
                                attn_map = compute_attn_weight(ref_query, ref_key)[:,:,:4096,4096:]

                                # Mean for each image token
                                mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

                                if "base_attn_weight" in ref_control_hyper_parameter:
                                    base_attn_weight = ref_control_hyper_parameter["base_attn_weight"]
                                else:
                                    base_attn_weight = 0
                                if "mask_attn_reweight" in ref_control_hyper_parameter:
                                    mask_attn_reweight = ref_control_hyper_parameter["mask_attn_reweight"]
                                else:
                                    mask_attn_reweight = 1

                                if "process_func" in ref_control_hyper_parameter:
                                    process_func = ref_control_hyper_parameter["process_func"]
                                else:
                                    process_func = process_attn_map_otsu

                                r2s_attn_weight = process_func(mean_attn_map*mask_attn_reweight+base_attn_weight)
                                ref_hidden_states, attn_weight = refer_scaled_dot_product_attention(ref_query, ref_key, ref_value, attn_reweight=[r2s_attn_weight,1])
                                ref_control_hidden_states.append(ref_hidden_states)

                                ######### Save attention maps ###########
                                if hasattr(self, "save_attn_maps") and encoder_hidden_states is not None:
                                    text_length = encoder_hidden_states_query_proj.shape[2]
                                    self.store_attn_map(attn_weight, text_length, height, timestep, ref_control_prompt)
                                #########################################


                            if "debug" in self.ref_control_signal and self.ref_control_signal["debug"]:
                                # attention of reference branch toward main branch contextual token
                                if "ref_control" not in self.debug_dict:
                                    self.debug_dict["ref_control"] = {}

                                if self.ref_control_signal["control_type"]=="main_context":
                                    for ref_idx in ref_control_idxs:

                                        attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
                                                                            torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs:bs+1]], dim=2))[:,:,:4096,4096:]
                                        s2c_attn_weight=torch.mean(attn_map, dim=[1,3])[0]


                                        debug_key=f"{self.step}_{self.layer}_{ref_idx}"
                                        self.debug_dict["ref_control"][debug_key]=s2c_attn_weight.cpu()
                                        raw_map, bin_map = process_attn_map_otsu_debug(s2c_attn_weight)
                                        self.debug_dict["ref_control"]["raw_"+debug_key]=raw_map.cpu()
                                        self.debug_dict["ref_control"]["bin_"+debug_key]=bin_map.cpu()
                                else:
                                    for ref_idx in ref_control_idxs:
                                        attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
                                                                            torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs+ref_idx+1:bs+ref_idx+2]], dim=2))[:,:,:4096,4096:]
                                        s2c_attn_weight=torch.mean(attn_map, dim=[1,3])[0]


                                        debug_key=f"{self.step}_{self.layer}_{ref_idx}"
                                        self.debug_dict["ref_control"][debug_key]=s2c_attn_weight.cpu()
                                        raw_map, bin_map = process_attn_map_otsu_debug(s2c_attn_weight)
                                        self.debug_dict["ref_control"]["raw_"+debug_key]=raw_map.cpu()
                                        self.debug_dict["ref_control"]["bin_"+debug_key]=bin_map.cpu()

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        if self.contextual_replace and self.operator == "concat":
            # Replace hidden states of the main generation section
            hidden_states[bs:bs+1]=main_hidden_states
            if self.ref_control_signal["on"]:
                # If define control layer but not use it
                if "control_layers" in self.ref_control_signal and self.layer not in self.ref_control_signal["control_layers"]:
                    pass
                else:
                    ref_idxs = self.ref_control_signal["ref_idxs"]
                    for instance_idx, ref_idx in enumerate(ref_idxs):
                        hidden_states[bs+ref_idx+1:bs+ref_idx+1+1]=ref_control_hidden_states[instance_idx]
        # print("hidden_states",hidden_states.shape)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
