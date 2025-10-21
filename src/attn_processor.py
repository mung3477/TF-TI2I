import torch.nn.functional as F
import torchvision.transforms as T
import math
import torch

def process_attn_map(data):
    data= data.view(1, 1, 64, 64)
    # smooth
    gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=1)
    data = gaussian_blur(data).flatten()
    # normalize
    data = (data - data.min()) / (data.max() - data.min())
    return data

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


def texture_control_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, attn_reweight=None) -> torch.Tensor:
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
    # print("attn_weight",attn_weight.shape)

    if attn_reweight is not None:
        attn_weight[:,:,:4096,:4096] *= attn_reweight[0] # self2self
        attn_weight[:,:,:4096,4096:4429] *= attn_reweight[1] # self2cross
        attn_weight[:,:,:4096,4429:] *= attn_reweight[2] # self2ref

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Texture_Control_JointAttnProcessor2_0_multi:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer_idx=-1, contextual_replace=True, replace_start=0, replace_end=-1, operator="concat", save_attn=False,
                  tex_control=False, tex_control_signal={}, debug=False,):
        self.layer_idx = layer_idx
        self.contextual_replace = contextual_replace
        self.replace_start = replace_start
        self.replace_end = replace_end
        self.operator = operator
        self.save_attn = save_attn
        self.tex_control = tex_control
        self.tex_control_signal = tex_control_signal
        self.debug = debug
        if save_attn:
            self.self2self_list=[]
            self.self2cross_list=[]
            self.self2ref_list=[]
        if self.debug:
            self.debug_dict={}
            self.debug_dict["tex_control_attn_map"]=[]


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
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

                    side_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    side_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    side_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)
                    side_key = torch.cat([key[bs:bs+1], side_encoder_hidden_states_key_proj], dim=2)
                    side_value = torch.cat([value[bs:bs+1], side_encoder_hidden_states_value_proj], dim=2)
                    # print("side_query",side_query.shape)
                    # print("side_key",side_key.shape)
                    # print("side_value",side_value.shape)

                    if self.save_attn:
                        side_attn_weight = compute_attn_weight(side_query, side_key, side_value)

                        head_mean = torch.mean(side_attn_weight, dim=1)


                        self2self = torch.mean(head_mean[0, :, :4096], dim=-1)
                        self2cross = torch.mean(head_mean[0, :, 4096:4429], dim=-1)
                        self2ref = torch.mean(head_mean[0, :, 4429:], dim=-1)

                        self.self2self_list.append(self2self)
                        self.self2cross_list.append(self2cross)
                        self.self2ref_list.append(self2ref)

                    if self.tex_control:
                        tex_control_hyper_parameter=self.tex_control_signal["hyper_parameter"]

                        side_attn_weight = compute_attn_weight(side_query, side_key, side_value)
                        head_mean = torch.mean(side_attn_weight, dim=1)
                        self2cross = torch.mean(head_mean[0, :4096, 4096:4429], dim=-1)
                        self2ref = torch.mean(head_mean[0, :4096, 4429:], dim=-1)
                        mask_attn_reweight = tex_control_hyper_parameter["mask_attn_reweight"]

                        if "process_func" in tex_control_hyper_parameter:
                            process_func = tex_control_hyper_parameter["process_func"]
                        else:
                            process_func = process_attn_map

                        if tex_control_hyper_parameter["attn_map_source"]=="cross":
                            mean_attn_map = self2cross
                        elif tex_control_hyper_parameter["attn_map_source"]=="ref":
                            mean_attn_map = self2ref
                            s2r_attn_weight = process_func(mean_attn_map)*mask_attn_reweight
                        elif tex_control_hyper_parameter["attn_map_source"]=="crossXref":
                            mean_attn_map_sc = torch.abs(self2cross)
                            mean_attn_map_sr = torch.abs(self2ref)
                            mean_attn_map = mean_attn_map_sr * mean_attn_map_sc
                        else:
                            mean_attn_map_sc = self2cross
                            mean_attn_map_sr = self2ref
                            mean_attn_map = mean_attn_map_sr - mean_attn_map_sc

                        if self.debug:
                            self.debug_dict["tex_control_attn_map"].append(mean_attn_map)
                        s2r_attn_weight = (process_func(mean_attn_map)*mask_attn_reweight).view(-1,1)
                        if "clamp" in tex_control_hyper_parameter:
                            clamp_value = tex_control_hyper_parameter["clamp"]
                            s2r_attn_weight = torch.clamp(s2r_attn_weight, min=clamp_value[0], max=clamp_value[1])

                        side_hidden_states = texture_control_scaled_dot_product_attention(side_query, side_key, side_value, dropout_p=0.0, is_causal=False, attn_reweight=[1,1,s2r_attn_weight])

                    else:
                        side_hidden_states = F.scaled_dot_product_attention(side_query, side_key, side_value, dropout_p=0.0, is_causal=False)


                    # print("side_hidden_states",side_hidden_states.shape)

                # print("encoder_hidden",encoder_hidden_states_query_proj[2].shape)
                # print("bs",bs)
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        if self.contextual_replace and self.operator == "concat":
            hidden_states[bs:bs+1]=side_hidden_states

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


def obj_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
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


    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight torch.Size([1, 38, 4429, 4762])
    # print("attn_weight",attn_weight.shape)

    if attn_reweight is not None:

        attn_weight[:, :, 4096:, :4096] *= attn_reweight[0] # ref2self
        attn_weight[:, :, 4096:, 4096:4429] *= attn_reweight[1] # ref2ref

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Object_Control_JointAttnProcessor2_0_multi:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer_idx=-1, contextual_replace=True, replace_start=0, replace_end=-1, operator="concat", save_attn=False,
                  obj_control=False, obj_control_signal=[], debug=False,):
        self.layer_idx = layer_idx
        self.contextual_replace = contextual_replace
        self.replace_start = replace_start
        self.replace_end = replace_end
        self.operator = operator
        self.save_attn = save_attn
        self.obj_control = obj_control
        self.obj_control_signal = obj_control_signal
        self.debug = debug
        if save_attn:
            self.self2self_list=[]
            self.self2cross_list=[]
            self.self2ref_list=[]
        if self.debug:
            self.debug_dict={}
            self.debug_dict["obj_control_attn_map"]=[]


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
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
                    # key for image token, encoder_hidden.. for textual token
                    side_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    side_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    side_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)
                    side_key = torch.cat([key[bs:bs+1], side_encoder_hidden_states_key_proj], dim=2)
                    side_value = torch.cat([value[bs:bs+1], side_encoder_hidden_states_value_proj], dim=2)

                    if self.save_attn:
                        side_attn_weight = compute_attn_weight(side_query, side_key, side_value)
                        # print("side_attn_weight",side_attn_weight.shape)

                        head_mean = torch.mean(side_attn_weight, dim=1)
                        # print("head_mean", head_mean.shape)

                        self2self = torch.mean(head_mean[0, :, :4096], dim=-1)
                        self2cross = torch.mean(head_mean[0, :, 4096:4429], dim=-1)
                        self2ref = torch.mean(head_mean[0, :, 4429:], dim=-1)

                        self.self2self_list.append(self2self)
                        self.self2cross_list.append(self2cross)
                        self.self2ref_list.append(self2ref)

                    if self.obj_control:
                        # Current design for object control on contextual token is based on restricting
                        # the information of reference image flowing into contextual token, updating the contextual token
                        # with mask restriction that generated based on textual attention
                        obj_ref_idxs = self.obj_control_signal["ref_idxs"]
                        obj_ref_prompts = self.obj_control_signal["ref_prompts"]
                        obj_control_hyper_parameter=self.obj_control_signal["hyper_parameter"]

                        obj_control_hidden_states=[]
                        for instance_idx, obj_ref_idx in enumerate(obj_ref_idxs):
                            obj_query = torch.cat([query[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1],
                                                  encoder_hidden_states_query_proj[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1]], dim=2)
                            obj_key = torch.cat([key[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1],
                                                  encoder_hidden_states_key_proj[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1]], dim=2)
                            obj_value = torch.cat([value[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1],
                                                  encoder_hidden_states_value_proj[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1]], dim=2)
                            attn_map = compute_attn_weight(obj_query, obj_key)[:,:,:4096,4096:]
                            mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

                            base_attn_weight = obj_control_hyper_parameter["base_attn_weight"]
                            mask_attn_reweight = obj_control_hyper_parameter["mask_attn_reweight"]
                            if "process_func" in obj_control_hyper_parameter:
                                process_func = obj_control_hyper_parameter["process_func"]
                            else:
                                process_func = process_attn_map

                            r2s_attn_weight = process_func(mean_attn_map)*mask_attn_reweight

                            r2s_attn_weight+=base_attn_weight
                            r2s_attn_weight = torch.clamp(r2s_attn_weight, 0, 1)


                            # if "obj_control_layer_list" in obj_control_hyper_parameter:
                            #     obj_control_layer = obj_control_hyper_parameter["obj_control_layer_list"]
                            #     base_attn_weight = obj_control_hyper_parameter["base_attn_weight"]
                            #     if self.layer_idx in obj_control_layer:
                            #         process_func = obj_control_hyper_parameter["process_func"]
                            #         r2s_attn_weight = process_func(mean_attn_map)
                            #     else:
                            #         r2s_attn_weight = torch.ones_like(mean_attn_map)
                            #     r2s_attn_weight+=base_attn_weight
                            #     r2s_attn_weight = torch.clamp(r2s_attn_weight, 0, 1)

                            ref_hidden_states = obj_scaled_dot_product_attention(obj_query, obj_key, obj_value, attn_reweight=[r2s_attn_weight,1])
                            obj_control_hidden_states.append(ref_hidden_states)
                            if self.debug:
                                self.debug_dict["obj_control_attn_map"].append(mean_attn_map.cpu())

                    side_hidden_states = F.scaled_dot_product_attention(side_query, side_key, side_value, dropout_p=0.0, is_causal=False)

                    # print("side_hidden_states",side_hidden_states.shape)

                # print("encoder_hidden",encoder_hidden_states_query_proj[2].shape)
                # print("bs",bs)
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)


        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        if self.contextual_replace and self.operator == "concat":
            hidden_states[bs:bs+1]=side_hidden_states
        if self.obj_control:
            obj_ref_idxs = self.obj_control_signal["ref_idxs"]
            for instance_idx, obj_ref_idx in enumerate(obj_ref_idxs):
                hidden_states[bs+obj_ref_idx+1:bs+obj_ref_idx+1+1]=obj_control_hidden_states[instance_idx]

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


    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight torch.Size([1, 38, 4429, 4762])
    # print("attn_weight",attn_weight.shape)

    if attn_reweight is not None:

        attn_weight[:, :, 4096:, :4096] *= attn_reweight[0] # ref2self
        attn_weight[:, :, 4096:, 4096:4429] *= attn_reweight[1] # ref2ref

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def main_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False,
        attn_reweight_dict={"self":1,
                            "cross":1,
                            "self2refers_idx":[],
                            "self2refers_weight":[],
                            "cross2refers_idx":[],
                            "cross2refers_weight":[],
                            "global_reweight_idx":[],
                            "global_reweight_weight":[]}) -> torch.Tensor:
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
    # print("attn_weight",attn_weight.shape)

    if attn_reweight_dict is not None:
        attn_weight[:,:,:4096,:4096] *= attn_reweight_dict["self"] # self2self
        attn_weight[:,:,:4096,4096:4429] *= attn_reweight_dict["cross"] # self2cross
        for refer_idx, refer_weight  in zip(attn_reweight_dict["self2refers_idx"],attn_reweight_dict["self2refers_weight"]):
            attn_weight[:, :, :4096, 4429+refer_idx*333:4762+refer_idx*333] *= refer_weight

        for refer_idx, refer_weight  in zip(attn_reweight_dict["cross2refers_idx"],attn_reweight_dict["cross2refers_weight"]):
            attn_weight[:, :, 4096:, 4429+refer_idx*333:4762+refer_idx*333] *= refer_weight

        for refer_idx, refer_weight  in zip(attn_reweight_dict["global_reweight_idx"],attn_reweight_dict["global_reweight_weight"]):
            attn_weight[:, :, 4096:, 4429+refer_idx*333:4762+refer_idx*333] *= refer_weight

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class task_sepcific_JointAttnProcessor2_0_multi:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer=-1, close_contextual_replace=False, replace_start=0, replace_end=-1, operator="concat",
                 global_control_signal={"close_contextual_replace":True,
                                        "self_attn_reweight":1,
                                        "cross_attn_reweight":1,
                                    },
                 obj_control_signal={"on":False,
                                     "hyper_parameter":{}},
                 tex_control_signal={"on":False,
                                     "hyper_parameter":{}},
                 act_control_signal={"on":False,
                                     "hyper_parameter":{}},
                 bg_control_signal={"on":False,
                                     "hyper_parameter":{}
                                     }
                 ):
        self.global_control_signal = global_control_signal
        self.step=-1
        self.layer=layer
        self.replace_start = replace_start
        self.replace_end = replace_end
        self.operator = operator

        # obj_control control r2s by mask IN REFER Branch
        # focus on retrieving object-related features
        self.obj_control_signal = obj_control_signal
        # tex_control control s2r by mask IN MAIN Branch
        # focus on applying texture-related features in object mask
        self.tex_control_signal = tex_control_signal
        # act_control control r2s by mask IN REFER Branch
        # similar to obj_control, but focus on blurred, strong features
        self.act_control_signal = act_control_signal
        # bg_control control s2r by non-mask IN MAIN Branch
        # similar to tex_control, but focus on blurred, strong features
        self.bg_control_signal = bg_control_signal

        self.debug_dict={}
        # IF NO LAYER ASSIGNMENT OR STEP ASSIGNMENT, activate contextual control

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask = None,
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
                    # Conditional Main Branch (bs:bs+1), Conditional Refer Branches (bs+1:bs*2)

                    main_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    main_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    main_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)
                    main_key = torch.cat([key[bs:bs+1], main_encoder_hidden_states_key_proj], dim=2)
                    main_value = torch.cat([value[bs:bs+1], main_encoder_hidden_states_value_proj], dim=2)



                    if self.obj_control_signal["on"]:
                        # obj_control control r2s by mask IN REFER Branch
                        # focus on retrieving object-related features
                        obj_ref_idxs = self.obj_control_signal["ref_idxs"]
                        obj_control_hyper_parameter=self.obj_control_signal["hyper_parameter"]
                        obj_control_hidden_states=[]
                        for obj_ref_idx in obj_ref_idxs:
                            # obtaion attention map for this object refer branch only
                            obj_query = torch.cat([query[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1],
                                                  encoder_hidden_states_query_proj[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1]], dim=2)
                            obj_key = torch.cat([key[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1],
                                                  encoder_hidden_states_key_proj[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1]], dim=2)
                            obj_value = torch.cat([value[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1],
                                                  encoder_hidden_states_value_proj[bs+obj_ref_idx+1 : bs+obj_ref_idx+1+1]], dim=2)
                            # Compute attention weight of visual token toward textual token
                            attn_map = compute_attn_weight(obj_query, obj_key)[:,:,:4096,4096:]
                            # Average across different heads
                            mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

                            if "base_attn_weight" in obj_control_hyper_parameter:
                                base_attn_weight = obj_control_hyper_parameter["base_attn_weight"]
                            else:
                                base_attn_weight = 0
                            if "mask_attn_reweight" in obj_control_hyper_parameter:
                                mask_attn_reweight = obj_control_hyper_parameter["mask_attn_reweight"]
                            else:
                                mask_attn_reweight = 1
                            if "process_func" in obj_control_hyper_parameter:
                                process_func = obj_control_hyper_parameter["process_func"]
                            else:
                                process_func = process_attn_map

                            if "control_step_list" in obj_control_hyper_parameter:
                                control_step_list = obj_control_hyper_parameter["control_step_list"]
                            else:
                                control_step_list = None
                            if "control_layer_list" in obj_control_hyper_parameter:
                                control_layer_list = obj_control_hyper_parameter["control_layer_list"]
                            else:
                                control_layer_list = None
                            step_match = control_step_list is None or self.step in control_step_list
                            layer_match = control_layer_list is None or self.layer in control_layer_list

                            if step_match and layer_match:
                                # match when not defined or defined and inside
                                r2s_attn_weight = process_func(mean_attn_map)*mask_attn_reweight
                                r2s_attn_weight+=base_attn_weight

                                if "clamp" in obj_control_hyper_parameter:
                                    clamp_value = obj_control_hyper_parameter["clamp"]
                                    r2s_attn_weight = torch.clamp(r2s_attn_weight, min=clamp_value[0], max=clamp_value[1])
                            else:
                                # Neglect the attention weight
                                r2s_attn_weight = torch.ones_like(mean_attn_map)

                            ref_hidden_states = refer_scaled_dot_product_attention(obj_query, obj_key, obj_value, attn_reweight=[r2s_attn_weight,1])
                            obj_control_hidden_states.append(ref_hidden_states)


                    if self.tex_control_signal["on"]:
                        # tex_control control s2r by mask IN MAIN Branch
                        # focus on applying texture-related features in object mask

                        # Compute attention of main branch toward object tokens (in texture and background references
                        tex_control_hyper_parameter=self.tex_control_signal["hyper_parameter"]
                        tex_ref_idxs = self.tex_control_signal["ref_idxs"]
                        tex_s2r_attn_list=[]
                        for tex_ref_idx in tex_ref_idxs:
                            # obtaion attention map for this object refer branch only
                            main_and_tex_key = torch.cat([key[bs:bs+1],
                                                        encoder_hidden_states_key_proj[bs+tex_ref_idx+1 : bs+tex_ref_idx+1+1]], dim=2)
                            # main_and_tex_value = torch.cat([value[bs:bs+1],
                            #                             encoder_hidden_states_value_proj[bs+tex_ref_idx+1 : bs+tex_ref_idx+1+1]], dim=2)
                            # Compute attention weight of visual token toward textual token
                            attn_map = compute_attn_weight(query[bs:bs+1], main_and_tex_key )[:,:,:4096,4096:]
                            # Average across different heads
                            mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

                            if "base_attn_weight" in tex_control_hyper_parameter:
                                base_attn_weight = tex_control_hyper_parameter["base_attn_weight"]
                            else:
                                base_attn_weight = 0
                            if "mask_attn_reweight" in tex_control_hyper_parameter:
                                mask_attn_reweight = tex_control_hyper_parameter["mask_attn_reweight"]
                            else:
                                mask_attn_reweight = 1.2
                            if "process_func" in tex_control_hyper_parameter:
                                process_func = tex_control_hyper_parameter["process_func"]
                            else:
                                process_func = process_attn_map

                            s2r_attn_weight = process_func(mean_attn_map)*mask_attn_reweight

                            if "control_step_list" in tex_control_hyper_parameter:
                                control_step_list = tex_control_hyper_parameter["control_step_list"]
                            else:
                                control_step_list = None
                            if "control_layer_list" in tex_control_hyper_parameter:
                                control_layer_list = tex_control_hyper_parameter["control_layer_list"]
                            else:
                                control_layer_list = None
                            step_match = control_step_list is None or self.step in control_step_list
                            layer_match = control_layer_list is None or self.layer in control_layer_list

                            if step_match and layer_match:
                                # match when not defined or defined and inside
                                s2r_attn_weight = process_func(mean_attn_map)*mask_attn_reweight
                                s2r_attn_weight+=base_attn_weight

                                if "clamp" in tex_control_hyper_parameter:
                                    clamp_value = tex_control_hyper_parameter["clamp"]
                                    s2r_attn_weight = torch.clamp(s2r_attn_weight, min=clamp_value[0], max=clamp_value[1])
                            else:
                                # Neglect the attention weight
                                s2r_attn_weight = torch.ones_like(mean_attn_map)
                            # reshape with additional dimension for attention manipulation
                            if "debug" in self.tex_control_signal:
                                if "tex_control" not in self.debug_dict:
                                    self.debug_dict["tex_control"] = {}
                                debug_key=f"{self.step}_{self.layer}_{tex_ref_idx}"
                                self.debug_dict["tex_control"][debug_key]=s2r_attn_weight.cpu()

                            s2r_attn_weight = s2r_attn_weight.view(-1,1)
                            tex_s2r_attn_list.append(s2r_attn_weight)


                    if self.act_control_signal["on"]:
                        # act_control control r2s by mask IN REFER Branch
                        # similar to obj_control, but focus on blurred, strong features
                        act_ref_idxs = self.act_control_signal["ref_idxs"]
                        act_control_hyper_parameter=self.act_control_signal["hyper_parameter"]

                        if "control_step_list" in act_control_hyper_parameter:
                                control_step_list = act_control_hyper_parameter["control_step_list"]
                        else:
                            control_step_list = None
                        if "control_layer_list" in act_control_hyper_parameter:
                            control_layer_list = act_control_hyper_parameter["control_layer_list"]
                        else:
                            control_layer_list = None
                        step_match = control_step_list is None or self.step in control_step_list
                        layer_match = control_layer_list is None or self.layer in control_layer_list


                        act_control_hidden_states=[]
                        for act_ref_idx in act_ref_idxs:
                            act_query = torch.cat([query[bs+act_ref_idx+1 : bs+act_ref_idx+1+1],
                                                encoder_hidden_states_query_proj[bs+act_ref_idx+1 : bs+act_ref_idx+1+1]], dim=2)
                            act_key = torch.cat([key[bs+act_ref_idx+1 : bs+act_ref_idx+1+1],
                                                encoder_hidden_states_key_proj[bs+act_ref_idx+1 : bs+act_ref_idx+1+1]], dim=2)
                            act_value = torch.cat([value[bs+act_ref_idx+1 : bs+act_ref_idx+1+1],
                                                encoder_hidden_states_value_proj[bs+act_ref_idx+1 : bs+act_ref_idx+1+1]], dim=2)
                            if step_match and layer_match:
                                # Compute attention weight of visual token toward textual token
                                attn_map = compute_attn_weight(act_query, act_key)[:,:,:4096,4096:]
                                # Average across different heads
                                mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

                                if "base_attn_weight" in act_control_hyper_parameter:
                                    base_attn_weight = act_control_hyper_parameter["base_attn_weight"]
                                else:
                                    base_attn_weight = 0
                                if "mask_attn_reweight" in act_control_hyper_parameter:
                                    mask_attn_reweight = act_control_hyper_parameter["mask_attn_reweight"]
                                else:
                                    mask_attn_reweight = 1
                                if "process_func" in act_control_hyper_parameter:
                                    process_func = act_control_hyper_parameter["process_func"]
                                else:
                                    process_func = process_attn_map

                                # match when not defined or defined and inside
                                r2s_attn_weight = process_func(mean_attn_map)*mask_attn_reweight
                                r2s_attn_weight+=base_attn_weight


                                if "clamp" in act_control_hyper_parameter:
                                    clamp_value = act_control_hyper_parameter["clamp"]
                                    r2s_attn_weight = torch.clamp(r2s_attn_weight, min=clamp_value[0], max=clamp_value[1])
                            else:
                                # Neglect the attention weight
                                r2s_attn_weight = 1

                            ref_hidden_states = refer_scaled_dot_product_attention(act_query, act_key, act_value, attn_reweight=[r2s_attn_weight,1])
                            act_control_hidden_states.append(ref_hidden_states)


                    if self.bg_control_signal["on"]:
                        # bg_control control s2r by non-mask IN MAIN Branch
                        # similar to tex_control, but focus on blurred, strong features
                        bg_control_hyper_parameter=self.bg_control_signal["hyper_parameter"]
                        bg_ref_idxs = self.bg_control_signal["ref_idxs"]
                        bg_s2r_attn_list=[]
                        for bg_ref_idx in bg_ref_idxs:
                            # obtaion attention map for this object refer branch only
                            main_and_bg_key = torch.cat([key[bs:bs+1],
                                                        encoder_hidden_states_key_proj[bs+bg_ref_idx+1 : bs+bg_ref_idx+1+1]], dim=2)
                            # main_and_bg_value = torch.cat([value[bs:bs+1],
                            #                             encoder_hidden_states_value_proj[bs+bg_ref_idx+1 : bs+bg_ref_idx+1+1]], dim=2)
                            # Compute attention weight of visual token toward textual token
                            attn_map = compute_attn_weight(query[bs:bs+1], main_and_bg_key )[:,:,:4096,4096:]
                            # Average across different heads
                            mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

                            if "base_attn_weight" in bg_control_hyper_parameter:
                                base_attn_weight = bg_control_hyper_parameter["base_attn_weight"]
                            else:
                                base_attn_weight = 0
                            if "mask_attn_reweight" in bg_control_hyper_parameter:
                                mask_attn_reweight = bg_control_hyper_parameter["mask_attn_reweight"]
                            else:
                                mask_attn_reweight = 1.2
                            if "process_func" in bg_control_hyper_parameter:
                                process_func = bg_control_hyper_parameter["process_func"]
                            else:
                                process_func = process_attn_map

                            # Focusing on non-object area
                            s2r_attn_weight = (1-process_func(mean_attn_map))*mask_attn_reweight

                            if "control_step_list" in bg_control_hyper_parameter:
                                control_step_list = bg_control_hyper_parameter["control_step_list"]
                            else:
                                control_step_list = None
                            if "control_layer_list" in bg_control_hyper_parameter:
                                control_layer_list = bg_control_hyper_parameter["control_layer_list"]
                            else:
                                control_layer_list = None
                            step_match = control_step_list is None or self.step in control_step_list
                            layer_match = control_layer_list is None or self.layer in control_layer_list

                            if step_match and layer_match:
                                # match when not defined or defined and inside
                                s2r_attn_weight = process_func(mean_attn_map)*mask_attn_reweight
                                s2r_attn_weight+=base_attn_weight

                                if "clamp" in bg_control_hyper_parameter:
                                    clamp_value = bg_control_hyper_parameter["clamp"]
                                    s2r_attn_weight = torch.clamp(s2r_attn_weight, min=clamp_value[0], max=clamp_value[1])
                            else:
                                # Neglect the attention weight
                                s2r_attn_weight = torch.ones_like(mean_attn_map)
                            # reshape with additional dimension for attention manipulation
                            if "debug" in self.bg_control_signal:
                                if "bg_control" not in self.debug_dict:
                                    self.debug_dict["bg_control"] = {}
                                debug_key=f"{self.step}_{self.layer}_{bg_ref_idx}"
                                self.debug_dict["bg_control"][debug_key]=s2r_attn_weight.cpu()

                            s2r_attn_weight = s2r_attn_weight.view(-1,1)
                            bg_s2r_attn_list.append(s2r_attn_weight)


                    main2ref_mask=torch.ones([main_query.shape[2], main_key.shape[2]]).to(torch.bool).to(main_query.device)

                    if self.global_control_signal["close_contextual_replace"]:
                        # if close contextual replace, we block inactivate reference to be utilzed by main branch
                        # otherwise main branch utilizing reference branch as default ()
                        if self.obj_control_signal["on"]:
                            inactivate=False
                            if "control_step_list" in obj_control_hyper_parameter and self.step not in obj_control_hyper_parameter["control_step_list"]:
                                inactivate=True
                            if "control_layer_list" in obj_control_hyper_parameter and self.layer not in obj_control_hyper_parameter["control_layer_list"]:
                                inactivate=True
                            if inactivate:
                                for obj_ref_idx in obj_ref_idxs:
                                    main2ref_mask[:4096, 4429+obj_ref_idx*333:4762+obj_ref_idx*333] = False

                        if self.tex_control_signal["on"]:
                            inactivate=False
                            if "control_step_list" in tex_control_hyper_parameter and self.step not in tex_control_hyper_parameter["control_step_list"]:
                                inactivate=True
                            if "control_layer_list" in tex_control_hyper_parameter and self.layer not in tex_control_hyper_parameter["control_layer_list"]:
                                inactivate=True
                            if inactivate:
                                for tex_ref_idx in tex_ref_idxs:
                                    main2ref_mask[:4096, 4429+tex_ref_idx*333:4762+tex_ref_idx*333] = False

                        if self.act_control_signal["on"]:
                            inactivate=False
                            if "control_step_list" in act_control_hyper_parameter and self.step not in act_control_hyper_parameter["control_step_list"]:
                                inactivate=True
                            if "control_layer_list" in act_control_hyper_parameter and self.layer not in act_control_hyper_parameter["control_layer_list"]:
                                inactivate=True
                            if inactivate:
                                for act_ref_idx in act_ref_idxs:
                                    main2ref_mask[:4096, 4429+act_ref_idx*333:4762+act_ref_idx*333] = False

                        if self.bg_control_signal["on"]:
                            inactivate=False
                            if "control_step_list" in bg_control_hyper_parameter and self.step not in bg_control_hyper_parameter["control_step_list"]:
                                inactivate=True
                            if "control_layer_list" in bg_control_hyper_parameter and self.layer not in bg_control_hyper_parameter["control_layer_list"]:
                                inactivate=True
                            if inactivate:
                                for bg_ref_idx in bg_ref_idxs:
                                    main2ref_mask[:4096, 4429+bg_ref_idx*333:4762+bg_ref_idx*333] = False
                    else:
                        pass

                    task_reweight_idx_list=[]
                    task_reweigh_list=[]
                    if self.tex_control_signal["on"]:
                        task_reweight_idx_list+=tex_ref_idxs
                        task_reweigh_list+=tex_s2r_attn_list
                    if self.bg_control_signal["on"]:
                        task_reweight_idx_list+=bg_ref_idxs
                        task_reweigh_list+=bg_s2r_attn_list
                    # self.debug_dict["main2ref_mask"]=main2ref_mask.cpu()
                    global_reweight_idx=[]
                    global_reweight_weight=[]
                    for signal in [self.obj_control_signal, self.tex_control_signal, self.act_control_signal, self.bg_control_signal]:
                        if "main_branch_reweight" in signal["hyper_parameter"]:
                            global_reweight_idx += signal["ref_idxs"]
                            global_reweight_weight += signal["hyper_parameter"]["main_branch_reweight"]


                    main_hidden_states = main_scaled_dot_product_attention(main_query, main_key, main_value, attn_mask=main2ref_mask, dropout_p=0.0,
                                                        is_causal=False, enable_gqa=False,
                                                        attn_reweight_dict={"self":self.global_control_signal["self_attn_reweight"],
                                                                            "cross":self.global_control_signal["cross_attn_reweight"],
                                                                            "self2refers_idx":task_reweight_idx_list,
                                                                            "self2refers_weight":task_reweigh_list,
                                                                            "cross2refers_idx":[],
                                                                            "cross2refers_weight":[],
                                                                            "global_reweight_idx":global_reweight_idx,
                                                                            "global_reweight_weight":global_reweight_weight})

                    # if self.tex_control_signal["on"] or self.bg_control_signal["on"]:
                    #     task_reweight_idx_list=[]
                    #     task_reweigh_list=[]
                    #     if self.tex_control_signal["on"]:
                    #         task_reweight_idx_list+=tex_ref_idxs
                    #         task_reweigh_list+=tex_s2r_attn_list
                    #     if self.bg_control_signal["on"]:
                    #         task_reweight_idx_list+=bg_ref_idxs
                    #         task_reweigh_list+=bg_s2r_attn_list
                    #     # self.debug_dict["main2ref_mask"]=main2ref_mask.cpu()
                    #     global_reweight_idx=[]
                    #     global_reweight_weight=[]
                    #     for signal in [self.obj_control_signal, self.tex_control_signal, self.act_control_signal, self.bg_control_signal]:
                    #         if "main_branch_reweight" in signal["hyper_parameter"]:
                    #             global_reweight_idx += signal["ref_idxs"]
                    #             global_reweight_weight += signal["hyper_parameter"]["main_branch_reweight"]


                    #     main_hidden_states = main_scaled_dot_product_attention(main_query, main_key, main_value, attn_mask=main2ref_mask, dropout_p=0.0,
                    #                                         is_causal=False, enable_gqa=False,
                    #                                         attn_reweight_dict={"self":self.global_control_signal["self_attn_reweight"],
                    #                                                             "cross":self.global_control_signal["cross_attn_reweight"],
                    #                                                             "self2refers_idx":task_reweight_idx_list,
                    #                                                             "self2refers_weight":task_reweigh_list,
                    #                                                             "cross2refers_idx":[],
                    #                                                             "cross2refers_weight":[],
                    #                                                             "global_reweight_idx":global_reweight_idx,
                    #                                                             "global_reweight_weight":global_reweight_weight})
                    # else:
                    #     # Pytorch backend support faster than customized computation method
                    #     # if don't need to manipulate attention texture and background
                    #     # utilizing original attention computation
                    #     main_hidden_states = F.scaled_dot_product_attention(main_query, main_key, main_value, dropout_p=0.0,
                    #                                                         attn_mask=main2ref_mask, is_causal=False)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        if self.contextual_replace and self.operator == "concat":
            hidden_states[bs:bs+1]=main_hidden_states
        if self.obj_control_signal["on"]:
            obj_ref_idxs = self.obj_control_signal["ref_idxs"]
            for instance_idx, obj_ref_idx in enumerate(obj_ref_idxs):
                hidden_states[bs+obj_ref_idx+1:bs+obj_ref_idx+1+1]=obj_control_hidden_states[instance_idx]
        if self.act_control_signal["on"]:
            act_ref_idxs = self.act_control_signal["ref_idxs"]
            for instance_idx, act_ref_idx in enumerate(act_ref_idxs):
                hidden_states[bs+act_ref_idx+1:bs+act_ref_idx+1+1]=act_control_hidden_states[instance_idx]

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



class Customized_JointAttnProcessor2_0_multi:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, contextual_replace=True, replace_start=0, replace_end=-1,operator="mean"):
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
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
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

                    side_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    side_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)
                    side_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)
                    side_key = torch.cat([key[bs:bs+1], side_encoder_hidden_states_key_proj], dim=2)
                    side_value = torch.cat([value[bs:bs+1], side_encoder_hidden_states_value_proj], dim=2)
                    # print("side_query",side_query.shape)
                    # print("side_key",side_key.shape)
                    # print("side_value",side_value.shape)
                    side_hidden_states = F.scaled_dot_product_attention(side_query, side_key, side_value, dropout_p=0.0, is_causal=False)
                    # print("side_hidden_states",side_hidden_states.shape)

                # print("encoder_hidden",encoder_hidden_states_query_proj[2].shape)
                # print("bs",bs)
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        if self.contextual_replace and self.operator == "concat":
            hidden_states[bs:bs+1]=side_hidden_states
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
import torch.nn.functional as F
import math
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


def process_attn_map_otsu_debug(data, blur_func=T.GaussianBlur(kernel_size=5, sigma=1)):
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
    return data.to(dtype), binary_data.to(dtype)



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

def WTA_scaled_dot_product_attention_debug(query, key, value, attn_mask=None, dropout_p=0.0,
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
    # print("attn_weight",attn_weight.shape)
    mean_atn_weight = torch.mean(attn_weight, dim=[0,1])
    refer_score=[]

    wta_weight = wta_parameter["wta_weight"]
    wta_shift = wta_parameter["wta_shift"]
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
        for ref_idx in range(len(wta_weight)):
            score_shift = wta_shift[ref_idx]
            total_score = torch.mean(mean_atn_weight[:,4429+333*ref_idx:4762+333*ref_idx], dim=-1)*wta_weight[ref_idx]+score_shift*abs_global_contextual_mean
            # print("total_score",total_score.shape)
            refer_score.append(total_score)
        refer_score = torch.stack(refer_score, dim=1)
        refers_argmax = torch.argmax(refer_score, dim=1)

        wta_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        # keep winner and set others to -inf
        for ref_idx in range(len(wta_weight)):
            rows_with_ref_idx = torch.nonzero(refers_argmax == ref_idx).squeeze()  # (N,)
            # refer_before and after to be -inf

            wta_bias[rows_with_ref_idx,4429:4429+333*ref_idx] = float("-inf")
            wta_bias[rows_with_ref_idx,4762+333*ref_idx:] = float("-inf")
        if wta_parameter["cross2ref"]==False:
            wta_bias[4096:,4429:] = float("-inf")
    return refers_argmax[:4096], refer_score



# class WTA_JointAttnProcessor2_0_multi:
#     """Attention processor used typically in processing the SD3-like self-attention projections."""

#     def __init__(self, layer=-1, contextual_replace=True, replace_start=0, replace_end=-1,operator="concat",
#                  wta_control_signal={},
#                  ref_control_signal={"on":False,
#                                     "ref_idxs":[],
#                                     "control_tyoe":"main_context",
#                                      },
#                  schedule_control_signal={"on":False,
#                                     "ref_idxs_schedule":[],
#                                      }):
#         self.step=-1
#         self.layer=layer

#         self.contextual_replace = contextual_replace

#         self.wta_control_signal=wta_control_signal
#         self.wta_parameter = wta_control_signal["hyper_parameter"]
#         self.ref_control_signal=ref_control_signal
#         self.schedule_control_signal=schedule_control_signal


#         self.replace_start = replace_start
#         self.replace_end = replace_end
#         self.operator = operator
#         self.debug_dict={}


#     def __call__(
#         self,
#         attn,
#         hidden_states: torch.FloatTensor,
#         encoder_hidden_states: torch.FloatTensor = None,
#         attention_mask = None,
#         *args,
#         **kwargs,
#     ) -> torch.FloatTensor:
#         self.step+=1
#         residual = hidden_states
#         batch_size = hidden_states.shape[0]

#         # `sample` projections.
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(hidden_states)
#         value = attn.to_v(hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # `context` projections.
#         if encoder_hidden_states is not None:
#             encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
#             encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#             encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

#             encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)
#             encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)
#             encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)

#             if attn.norm_added_q is not None:
#                 encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
#             if attn.norm_added_k is not None:
#                 encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

#             if self.contextual_replace:
#                 bs = len(encoder_hidden_states_query_proj)//2

#                 if self.operator == "mean":
#                     encoder_hidden_states_query_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_query_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
#                     encoder_hidden_states_key_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_key_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
#                     encoder_hidden_states_value_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_value_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
#                 elif self.operator == "head_wise":
#                     qs=[]
#                     ks=[]
#                     vs=[]
#                     for i in range(bs):
#                         q_split = torch.chunk(encoder_hidden_states_query_proj[bs+i], bs, dim=0)
#                         k_split = torch.chunk(encoder_hidden_states_key_proj[bs+i], bs, dim=0)
#                         v_split = torch.chunk(encoder_hidden_states_value_proj[bs+i], bs, dim=0)

#                         qs.append(q_split[i])
#                         ks.append(k_split[i])
#                         vs.append(v_split[i])
#                     encoder_hidden_states_query_proj[bs] = torch.cat(qs, dim=0)
#                     encoder_hidden_states_key_proj[bs] = torch.cat(ks, dim=0)
#                     encoder_hidden_states_value_proj[bs] = torch.cat(vs, dim=0)
#                 elif self.operator == "concat":
#                     # Side branch

#                     main_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
#                     main_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)
#                     main_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)
#                     main_key = torch.cat([key[bs:bs+1], main_encoder_hidden_states_key_proj], dim=2)
#                     main_value = torch.cat([value[bs:bs+1], main_encoder_hidden_states_value_proj], dim=2)
#                     # print("main_query",main_query.shape)
#                     # print("main_key",main_key.shape)
#                     # print("main_value",main_value.shape)

#                     # Schedule Contextual Control
#                     if self.schedule_control_signal["on"]:

#                         L, S = main_query.size(-2), main_key.size(-2)
#                         schedule_mask = torch.ones(L, S, dtype=bool, device=main_query.device)
#                         ref_idxs_schedule = self.schedule_control_signal["ref_idxs_schedule"]
#                         ref_len = (S-4096-333)//333
#                         for ref_idx in range(ref_len):
#                             cur_step_ref_idxs = ref_idxs_schedule[self.step]
#                             if ref_idx not in cur_step_ref_idxs:
#                                 schedule_mask[:,4429+333*ref_idx:4762+333*ref_idx] = False

#                     else:
#                         schedule_mask = None

#                     main_hidden_states = WTA_scaled_dot_product_attention(main_query, main_key, main_value, dropout_p=0.0,
#                                                                           attn_mask=schedule_mask,
#                                                                           is_causal=False, wta_parameter=self.wta_parameter)
#                     if "debug" in self.wta_control_signal and self.wta_control_signal["debug"]:
#                         if "wta_control" not in self.debug_dict:
#                             self.debug_dict["wta_control"] = {}
#                         debug_key=f"{self.step}_{self.layer}"
#                         weight, score=WTA_scaled_dot_product_attention_debug(main_query, main_key, main_value, dropout_p=0.0, is_causal=False, wta_parameter=self.wta_parameter)
#                         self.debug_dict["wta_control"][debug_key]=weight.cpu()
#                         debug_key=f"{self.step}_{self.layer}_score"
#                         self.debug_dict["wta_control"][debug_key]=score.cpu()
#                     # print("main_hidden_states",main_hidden_states.shape)

#                     if self.ref_control_signal['on']:
#                         ref_control_idxs = self.ref_control_signal['ref_idxs']
#                         ref_control_hyper_parameter=self.ref_control_signal["hyper_parameter"]
#                         ref_control_hidden_states=[]
#                         for instance_idx, ref_idx in enumerate(ref_control_idxs):


#                             ref_query = torch.cat([query[bs+ref_idx+1 : bs+ref_idx+1+1],
#                                                 encoder_hidden_states_query_proj[bs+ref_idx+1 : bs+ref_idx+1+1]], dim=2)
#                             ref_key = torch.cat([key[bs+ref_idx+1 : bs+ref_idx+1+1],
#                                                 encoder_hidden_states_key_proj[bs+ref_idx+1 : bs+ref_idx+1+1]], dim=2)
#                             ref_value = torch.cat([value[bs+ref_idx+1 : bs+ref_idx+1+1],
#                                                 encoder_hidden_states_value_proj[bs+ref_idx+1 : bs+ref_idx+1+1]], dim=2)

#                             if self.ref_control_signal["control_type"] == "main_prompt":
#                                 attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
#                                                         torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs:bs+1]], dim=2))[:,:,:4096,4096:]
#                             else:
#                                 attn_map = compute_attn_weight(ref_query, ref_key)[:,:,:4096,4096:]

#                             mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

#                             if "base_attn_weight" in ref_control_hyper_parameter:
#                                 base_attn_weight = ref_control_hyper_parameter["base_attn_weight"]
#                             else:
#                                 base_attn_weight = 0
#                             if "mask_attn_reweight" in ref_control_hyper_parameter:
#                                 mask_attn_reweight = ref_control_hyper_parameter["mask_attn_reweight"]
#                             else:
#                                 mask_attn_reweight = 1

#                             if "process_func" in ref_control_hyper_parameter:
#                                 process_func = ref_control_hyper_parameter["process_func"]
#                             else:
#                                 process_func = process_attn_map_otsu

#                             r2s_attn_weight = process_func(mean_attn_map*mask_attn_reweight+base_attn_weight)
#                             ref_hidden_states = refer_scaled_dot_product_attention(ref_query, ref_key, ref_value, attn_reweight=[r2s_attn_weight,1])
#                             ref_control_hidden_states.append(ref_hidden_states)


#                         if "debug" in self.ref_control_signal and self.ref_control_signal["debug"]:
#                             # attention of reference branch toward main branch contextual token
#                             if "ref_control" not in self.debug_dict:
#                                 self.debug_dict["ref_control"] = {}

#                             if self.ref_control_signal["control_type"]=="main_prompt":
#                                 for ref_idx in ref_control_idxs:
#                                     attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
#                                                                     torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs:bs+1]], dim=2))[:,:,:4096,4096:]
#                                     s2c_attn_weight=torch.mean(attn_map, dim=[1,3])[0]


#                                     debug_key=f"{self.step}_{self.layer}_{ref_idx}"
#                                     self.debug_dict["ref_control"][debug_key]=s2c_attn_weight.cpu()
#                                     raw_map, bin_map = process_attn_map_otsu_debug(s2c_attn_weight)
#                                     self.debug_dict["ref_control"]["raw_"+debug_key]=raw_map.cpu()
#                                     self.debug_dict["ref_control"]["bin_"+debug_key]=bin_map.cpu()
#                             else:
#                                 for ref_idx in ref_control_idxs:
#                                     attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
#                                                                         torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs+ref_idx+1:bs+ref_idx+2]], dim=2))[:,:,:4096,4096:]
#                                     s2c_attn_weight=torch.mean(attn_map, dim=[1,3])[0]


#                                     debug_key=f"{self.step}_{self.layer}_{ref_idx}"
#                                     self.debug_dict["ref_control"][debug_key]=s2c_attn_weight.cpu()
#                                     raw_map, bin_map = process_attn_map_otsu_debug(s2c_attn_weight)
#                                     self.debug_dict["ref_control"]["raw_"+debug_key]=raw_map.cpu()
#                                     self.debug_dict["ref_control"]["bin_"+debug_key]=bin_map.cpu()

#                 # print("encoder_hidden",encoder_hidden_states_query_proj[2].shape)
#                 # print("bs",bs)
#             query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
#             key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
#             value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

#         hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#         if self.contextual_replace and self.operator == "concat":
#             hidden_states[bs:bs+1]=main_hidden_states
#         # Update references contextual token with Reference Selctive Masking
#         if self.ref_control_signal["on"]:
#             ref_idxs = self.ref_control_signal["ref_idxs"]
#             for instance_idx, ref_idx in enumerate(ref_idxs):
#                 hidden_states[bs+ref_idx+1:bs+ref_idx+1+1]=ref_control_hidden_states[instance_idx]
#         # print("hidden_states",hidden_states.shape)
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)



#         if encoder_hidden_states is not None:
#             # Split the attention outputs.
#             hidden_states, encoder_hidden_states = (
#                 hidden_states[:, : residual.shape[1]],
#                 hidden_states[:, residual.shape[1] :],
#             )
#             if not attn.context_pre_only:
#                 encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)
#         if encoder_hidden_states is not None:
#             return hidden_states, encoder_hidden_states
#         else:
#             return hidden_states

class TI2I_JointAttnProcessor2_0_multi:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, layer=-1, contextual_replace=True, replace_start=0, replace_end=-1,operator="concat",
                 wta_control_signal={},
                 ref_control_signal={"on":False,
                                    "ref_idxs":[],
                                    "control_tyoe":"main_context",
                                     }):
        self.step=-1
        self.layer=layer
        self.wta_control_signal=wta_control_signal

        self.ref_control_signal=ref_control_signal
        self.contextual_replace = contextual_replace
        self.replace_start = replace_start
        self.replace_end = replace_end
        self.operator = operator
        self.debug_dict={}


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask = None,
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
                        if "debug" in self.wta_control_signal and self.wta_control_signal["debug"]:
                            if "wta_control" not in self.debug_dict:
                                self.debug_dict["wta_control"] = {}
                            debug_key=f"{self.step}_{self.layer}"
                            weight, score=WTA_scaled_dot_product_attention_debug(main_query, main_key, main_value, dropout_p=0.0, is_causal=False, wta_parameter=self.wta_parameter)
                            self.debug_dict["wta_control"][debug_key]=weight.cpu()
                            debug_key=f"{self.step}_{self.layer}_score"
                            self.debug_dict["wta_control"][debug_key]=score.cpu()
                        # print("main_hidden_states",main_hidden_states.shape)

                    if self.ref_control_signal['on']:
                        # If define control layer but not use it
                        if "control_layers" in self.ref_control_signal and self.layer not in self.ref_control_signal["control_layers"]:
                            pass
                        else:
                            ref_control_idxs = self.ref_control_signal['ref_idxs']
                            ref_control_hyper_parameter=self.ref_control_signal["hyper_parameter"]
                            ref_control_hidden_states=[]
                            for instance_idx, ref_ref_idx in enumerate(ref_control_idxs):
                                # concat reference latent with the text embedding of the reference prompt
                                ref_query = torch.cat([query[bs+ref_ref_idx+1 : bs+ref_ref_idx+1+1],
                                                    encoder_hidden_states_query_proj[bs+ref_ref_idx+1 : bs+ref_ref_idx+1+1]], dim=2)
                                ref_key = torch.cat([key[bs+ref_ref_idx+1 : bs+ref_ref_idx+1+1],
                                                    encoder_hidden_states_key_proj[bs+ref_ref_idx+1 : bs+ref_ref_idx+1+1]], dim=2)
                                ref_value = torch.cat([value[bs+ref_ref_idx+1 : bs+ref_ref_idx+1+1],
                                                    encoder_hidden_states_value_proj[bs+ref_ref_idx+1 : bs+ref_ref_idx+1+1]], dim=2)
                                # I2Ctx
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
                                ref_hidden_states = refer_scaled_dot_product_attention(ref_query, ref_key, ref_value, attn_reweight=[r2s_attn_weight,1])
                                ref_control_hidden_states.append(ref_hidden_states)


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

                # print("encoder_hidden",encoder_hidden_states_query_proj[2].shape)
                # print("bs",bs)
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


# class TI2I_JointAttnProcessor2_0_multi:
#     """Attention processor used typically in processing the SD3-like self-attention projections."""

#     def __init__(self, layer=-1, contextual_replace=True, replace_start=0, replace_end=-1,operator="concat",
#                  wta_control_signal={"on":False},
#                  ref_control_signal={"on":False,
#                                     "ref_idxs":[],
#                                     "control_tyoe":"main_context",
#                                      },
#                  schedule_control_signal={"on":False,
#                                     "ref_idxs_schedule":[],
#                                      }):
#         self.step=-1
#         self.layer=layer

#         self.contextual_replace = contextual_replace

#         self.wta_control_signal=wta_control_signal
#         self.wta_parameter = wta_control_signal["hyper_parameter"]
#         self.ref_control_signal=ref_control_signal
#         self.schedule_control_signal=schedule_control_signal


#         self.replace_start = replace_start
#         self.replace_end = replace_end
#         self.operator = operator
#         self.debug_dict={}


#     def __call__(
#         self,
#         attn,
#         hidden_states: torch.FloatTensor,
#         encoder_hidden_states: torch.FloatTensor = None,
#         attention_mask = None,
#         *args,
#         **kwargs,
#     ) -> torch.FloatTensor:
#         self.step+=1
#         residual = hidden_states
#         batch_size = hidden_states.shape[0]

#         # `sample` projections.
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(hidden_states)
#         value = attn.to_v(hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         if attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if attn.norm_k is not None:
#             key = attn.norm_k(key)

#         # `context` projections.
#         if encoder_hidden_states is not None:
#             encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
#             encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#             encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

#             encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)
#             encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)
#             encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
#                 batch_size, -1, attn.heads, head_dim
#             ).transpose(1, 2)

#             if attn.norm_added_q is not None:
#                 encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
#             if attn.norm_added_k is not None:
#                 encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

#             if self.contextual_replace:
#                 bs = len(encoder_hidden_states_query_proj)//2

#                 if self.operator == "mean":
#                     encoder_hidden_states_query_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_query_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
#                     encoder_hidden_states_key_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_key_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
#                     encoder_hidden_states_value_proj[bs, :, self.replace_start:self.replace_end, :] = torch.mean(encoder_hidden_states_value_proj[bs+1:bs*2, :, self.replace_start:self.replace_end, :], dim=0)
#                 elif self.operator == "head_wise":
#                     qs=[]
#                     ks=[]
#                     vs=[]
#                     for i in range(bs):
#                         q_split = torch.chunk(encoder_hidden_states_query_proj[bs+i], bs, dim=0)
#                         k_split = torch.chunk(encoder_hidden_states_key_proj[bs+i], bs, dim=0)
#                         v_split = torch.chunk(encoder_hidden_states_value_proj[bs+i], bs, dim=0)

#                         qs.append(q_split[i])
#                         ks.append(k_split[i])
#                         vs.append(v_split[i])
#                     encoder_hidden_states_query_proj[bs] = torch.cat(qs, dim=0)
#                     encoder_hidden_states_key_proj[bs] = torch.cat(ks, dim=0)
#                     encoder_hidden_states_value_proj[bs] = torch.cat(vs, dim=0)
#                 elif self.operator == "concat":
#                     # Side branch

#                     main_encoder_hidden_states_key_proj = torch.cat(list(encoder_hidden_states_key_proj)[bs:bs*2], dim=1).unsqueeze(0)
#                     main_encoder_hidden_states_value_proj = torch.cat(list(encoder_hidden_states_value_proj)[bs:bs*2], dim=1).unsqueeze(0)
#                     main_query = torch.cat([query[bs:bs+1], encoder_hidden_states_query_proj[bs:bs+1]], dim=2)
#                     main_key = torch.cat([key[bs:bs+1], main_encoder_hidden_states_key_proj], dim=2)
#                     main_value = torch.cat([value[bs:bs+1], main_encoder_hidden_states_value_proj], dim=2)
#                     # print("main_query",main_query.shape)
#                     # print("main_key",main_key.shape)
#                     # print("main_value",main_value.shape)

#                     # Schedule Contextual Control
#                     if self.schedule_control_signal["on"]:

#                         L, S = main_query.size(-2), main_key.size(-2)
#                         schedule_mask = torch.ones(L, S, dtype=bool, device=main_query.device)
#                         ref_idxs_schedule = self.schedule_control_signal["ref_idxs_schedule"]
#                         ref_len = (S-4096-333)//333
#                         for ref_idx in range(ref_len):
#                             cur_step_ref_idxs = ref_idxs_schedule[self.step]
#                             if ref_idx not in cur_step_ref_idxs:
#                                 schedule_mask[:,4429+333*ref_idx:4762+333*ref_idx] = False

#                     else:
#                         schedule_mask = None
#                     if self.wta_control_signal["on"]:
#                         main_hidden_states = WTA_scaled_dot_product_attention(main_query, main_key, main_value, dropout_p=0.0,
#                                                                             attn_mask=schedule_mask,
#                                                                             is_causal=False, wta_parameter=self.wta_parameter)
#                     else:
#                         main_hidden_states = F.scaled_dot_product_attention(main_query, main_key, main_value, attn_mask=schedule_mask,dropout_p=0.0, is_causal=False)

#                     if "debug" in self.wta_control_signal and self.wta_control_signal["debug"]:
#                         if "wta_control" not in self.debug_dict:
#                             self.debug_dict["wta_control"] = {}
#                         debug_key=f"{self.step}_{self.layer}"
#                         weight, score=WTA_scaled_dot_product_attention_debug(main_query, main_key, main_value, dropout_p=0.0, is_causal=False, wta_parameter=self.wta_parameter)
#                         self.debug_dict["wta_control"][debug_key]=weight.cpu()
#                         debug_key=f"{self.step}_{self.layer}_score"
#                         self.debug_dict["wta_control"][debug_key]=score.cpu()
#                     # print("main_hidden_states",main_hidden_states.shape)

#                     if self.ref_control_signal['on']:
#                         ref_control_idxs = self.ref_control_signal['ref_idxs']
#                         ref_control_hyper_parameter=self.ref_control_signal["hyper_parameter"]
#                         ref_control_hidden_states=[]
#                         for instance_idx, ref_idx in enumerate(ref_control_idxs):


#                             ref_query = torch.cat([query[bs+ref_idx+1 : bs+ref_idx+1+1],
#                                                 encoder_hidden_states_query_proj[bs+ref_idx+1 : bs+ref_idx+1+1]], dim=2)
#                             ref_key = torch.cat([key[bs+ref_idx+1 : bs+ref_idx+1+1],
#                                                 encoder_hidden_states_key_proj[bs+ref_idx+1 : bs+ref_idx+1+1]], dim=2)
#                             ref_value = torch.cat([value[bs+ref_idx+1 : bs+ref_idx+1+1],
#                                                 encoder_hidden_states_value_proj[bs+ref_idx+1 : bs+ref_idx+1+1]], dim=2)

#                             if self.ref_control_signal["control_type"] == "main_prompt":
#                                 attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
#                                                         torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs:bs+1]], dim=2))[:,:,:4096,4096:]
#                             else:
#                                 attn_map = compute_attn_weight(ref_query, ref_key)[:,:,:4096,4096:]

#                             mean_attn_map = torch.mean(attn_map, dim=[1,3])[0]

#                             if "base_attn_weight" in ref_control_hyper_parameter:
#                                 base_attn_weight = ref_control_hyper_parameter["base_attn_weight"]
#                             else:
#                                 base_attn_weight = 0
#                             if "mask_attn_reweight" in ref_control_hyper_parameter:
#                                 mask_attn_reweight = ref_control_hyper_parameter["mask_attn_reweight"]
#                             else:
#                                 mask_attn_reweight = 1

#                             if "process_func" in ref_control_hyper_parameter:
#                                 process_func = ref_control_hyper_parameter["process_func"]
#                             else:
#                                 process_func = process_attn_map_otsu

#                             r2s_attn_weight = process_func(mean_attn_map*mask_attn_reweight+base_attn_weight)
#                             ref_hidden_states = refer_scaled_dot_product_attention(ref_query, ref_key, ref_value, attn_reweight=[r2s_attn_weight,1])
#                             ref_control_hidden_states.append(ref_hidden_states)


#                         if "debug" in self.ref_control_signal and self.ref_control_signal["debug"]:
#                             # attention of reference branch toward main branch contextual token
#                             if "ref_control" not in self.debug_dict:
#                                 self.debug_dict["ref_control"] = {}

#                             if self.ref_control_signal["control_type"]=="main_prompt":
#                                 for ref_idx in ref_control_idxs:
#                                     attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
#                                                                     torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs:bs+1]], dim=2))[:,:,:4096,4096:]
#                                     s2c_attn_weight=torch.mean(attn_map, dim=[1,3])[0]


#                                     debug_key=f"{self.step}_{self.layer}_{ref_idx}"
#                                     self.debug_dict["ref_control"][debug_key]=s2c_attn_weight.cpu()
#                                     raw_map, bin_map = process_attn_map_otsu_debug(s2c_attn_weight)
#                                     self.debug_dict["ref_control"]["raw_"+debug_key]=raw_map.cpu()
#                                     self.debug_dict["ref_control"]["bin_"+debug_key]=bin_map.cpu()
#                             else:
#                                 for ref_idx in ref_control_idxs:
#                                     attn_map = compute_attn_weight(query[bs+ref_idx+1:bs+ref_idx+2],
#                                                                         torch.cat([key[bs+ref_idx+1:bs+ref_idx+2], encoder_hidden_states_key_proj[bs+ref_idx+1:bs+ref_idx+2]], dim=2))[:,:,:4096,4096:]
#                                     s2c_attn_weight=torch.mean(attn_map, dim=[1,3])[0]


#                                     debug_key=f"{self.step}_{self.layer}_{ref_idx}"
#                                     self.debug_dict["ref_control"][debug_key]=s2c_attn_weight.cpu()
#                                     raw_map, bin_map = process_attn_map_otsu_debug(s2c_attn_weight)
#                                     self.debug_dict["ref_control"]["raw_"+debug_key]=raw_map.cpu()
#                                     self.debug_dict["ref_control"]["bin_"+debug_key]=bin_map.cpu()

#                 # print("encoder_hidden",encoder_hidden_states_query_proj[2].shape)
#                 # print("bs",bs)
#             query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
#             key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
#             value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

#         hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
#         if self.contextual_replace and self.operator == "concat":
#             hidden_states[bs:bs+1]=main_hidden_states
#         # Update references contextual token with Reference Selctive Masking
#         if self.ref_control_signal["on"]:
#             ref_idxs = self.ref_control_signal["ref_idxs"]
#             for instance_idx, ref_idx in enumerate(ref_idxs):
#                 hidden_states[bs+ref_idx+1:bs+ref_idx+1+1]=ref_control_hidden_states[instance_idx]
#         # print("hidden_states",hidden_states.shape)
#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)



#         if encoder_hidden_states is not None:
#             # Split the attention outputs.
#             hidden_states, encoder_hidden_states = (
#                 hidden_states[:, : residual.shape[1]],
#                 hidden_states[:, residual.shape[1] :],
#             )
#             if not attn.context_pre_only:
#                 encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)
#         if encoder_hidden_states is not None:
#             return hidden_states, encoder_hidden_states
#         else:
#             return hidden_states
