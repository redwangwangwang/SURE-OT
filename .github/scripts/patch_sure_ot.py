#!/usr/bin/env python3

from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one literal match, found {count}"
        )
    return text.replace(old, new, 1)


model_path = Path("models/R2GenGPT.py")
model = model_path.read_text(encoding="utf-8")

model = replace_once(
    model,
    "from evalcap.BiOT import SinkhornOTChangeDetector\n",
    "from evalcap.BiOT import SinkhornOTChangeDetector\n"
    "from evalcap.SURE_OT import SUREOTModule\n",
    "SURE-OT import",
)

model = replace_once(
    model,
    "        self.detector = SinkhornOTChangeDetector(eps=0.1, max_iter=50, thresh_mode=0.05, k_ratio=0.1, global_q=0.9, init_alpha=10.0)\n",
    '''        self.use_sure_ot = getattr(args, "use_sure_ot", True)
        self.detector = None
        if not self.use_sure_ot:
            self.detector = SinkhornOTChangeDetector(
                eps=0.1,
                max_iter=50,
                thresh_mode=0.05,
                k_ratio=0.1,
                global_q=0.9,
                init_alpha=10.0,
            )
''',
    "detector initialization",
)

model = replace_once(
    model,
    "        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)\n",
    '''        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.sure_ot = None
        if self.use_sure_ot:
            self.sure_ot = SUREOTModule(
                dim=self.llama_model.config.hidden_size,
                num_tokens=getattr(args, "sure_ot_num_tokens", 2),
                epsilon=getattr(args, "sure_ot_epsilon", 0.07),
                tau=getattr(args, "sure_ot_tau", 0.7),
                num_iters=getattr(args, "sure_ot_iters", 40),
                spatial_weight=getattr(
                    args, "sure_ot_spatial_weight", 0.05
                ),
                adapter_rank=getattr(args, "sure_ot_adapter_rank", 128),
                prior_strength=getattr(
                    args, "sure_ot_prior_strength", 1.0
                ),
                residual_threshold=getattr(
                    args, "sure_ot_residual_threshold", 0.25
                ),
                balanced=getattr(args, "sure_ot_balanced", False),
                use_role_adapters=getattr(
                    args, "sure_ot_use_role_adapters", True
                ),
            )
''',
    "SURE-OT module initialization",
)

prompt_methods = r'''    def prompt_wrap(
        self,
        img_embeds,
        atts_img,
        context_img_embeds,
        dynamic_prompt=None,
        evolution_tokens=None,
    ):
        """Build either the original text prompt or SURE-OT token prompt."""
        batch_size = img_embeds.shape[0]
        device = img_embeds.device

        p_1 = "Human: <Img>"
        p_4 = "</Img> for reference \nAssistant:"
        p_1_tokens = self.llama_tokenizer(
            p_1, return_tensors="pt", add_special_tokens=False
        ).to(device)
        p_4_tokens = self.llama_tokenizer(
            p_4, return_tensors="pt", add_special_tokens=False
        ).to(device)
        p_1_embeds = self.embed_tokens(p_1_tokens.input_ids).expand(
            batch_size, -1, -1
        )
        p_4_embeds = self.embed_tokens(p_4_tokens.input_ids).expand(
            batch_size, -1, -1
        )

        if evolution_tokens is not None:
            p_2 = (
                "</Img> Generate a comprehensive and detailed diagnosis report "
                "for this chest xray image. Use the learned temporal evolution "
                "evidence below to reason about newly emerged, resolved, "
                "persistent, and uncertain findings: <Evo>"
            )
            p_3 = "</Evo>\nHere is the historical chest xray image: <Img>"
            p_2_tokens = self.llama_tokenizer(
                p_2, return_tensors="pt", add_special_tokens=False
            ).to(device)
            p_3_tokens = self.llama_tokenizer(
                p_3, return_tensors="pt", add_special_tokens=False
            ).to(device)
            p_2_embeds = self.embed_tokens(p_2_tokens.input_ids).expand(
                batch_size, -1, -1
            )
            p_3_embeds = self.embed_tokens(p_3_tokens.input_ids).expand(
                batch_size, -1, -1
            )
            wrapped_img_embeds = torch.cat(
                [
                    p_1_embeds,
                    img_embeds,
                    p_2_embeds,
                    evolution_tokens,
                    p_3_embeds,
                    context_img_embeds,
                    p_4_embeds,
                ],
                dim=1,
            )
        else:
            if dynamic_prompt is None:
                raise ValueError(
                    "dynamic_prompt is required when evolution_tokens is None"
                )
            p_2 = [
                (
                    "</Img> Generate a comprehensive and detailed diagnosis "
                    f"report for this chest xray image. {dynamic_prompt[i]}"
                )
                for i in range(batch_size)
            ]
            p_3 = "\nHere is the historical chest xray image: <Img>"
            p_3_tokens = self.llama_tokenizer(
                p_3, return_tensors="pt", add_special_tokens=False
            ).to(device)
            dynamic_prompt_tokens = self.llama_tokenizer(
                p_2,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
                max_length=60,
            ).to(device)
            p_2_embeds = self.embed_tokens(dynamic_prompt_tokens.input_ids)
            p_3_embeds = self.embed_tokens(p_3_tokens.input_ids).expand(
                batch_size, -1, -1
            )
            wrapped_img_embeds = torch.cat(
                [
                    p_1_embeds,
                    img_embeds,
                    p_2_embeds,
                    p_3_embeds,
                    context_img_embeds,
                    p_4_embeds,
                ],
                dim=1,
            )

        wrapped_atts_img = atts_img[:, :1].expand(
            -1, wrapped_img_embeds.shape[1]
        )
        return wrapped_img_embeds, wrapped_atts_img

    def _prepare_temporal_inputs(
        self,
        current_image_embeds,
        atts_img,
        context_image_embeds,
        compute_swap=False,
    ):
        """Create report-generation inputs for SURE-OT or the baseline."""
        if self.use_sure_ot:
            sure_ot_output = self.sure_ot(
                current_image_embeds,
                context_image_embeds,
                compute_swap=(
                    compute_swap
                    and getattr(self.args, "sure_ot_use_swap", True)
                ),
            )
            wrapped, wrapped_atts = self.prompt_wrap(
                current_image_embeds,
                atts_img,
                context_image_embeds,
                evolution_tokens=sure_ot_output["tokens"],
            )
            return wrapped, wrapped_atts, sure_ot_output

        dynamic_prompt, _, _, _, _ = self.detector.forward(
            current_image_embeds, context_image_embeds
        )
        wrapped, wrapped_atts = self.prompt_wrap(
            current_image_embeds,
            atts_img,
            context_image_embeds,
            dynamic_prompt=dynamic_prompt,
        )
        return wrapped, wrapped_atts, None

'''

model, prompt_count = re.subn(
    r"    def prompt_wrap\(.*?(?=    def forward\(self, samples\):)",
    lambda _match: prompt_methods,
    model,
    count=1,
    flags=re.DOTALL,
)
if prompt_count != 1:
    raise RuntimeError(
        f"prompt_wrap replacement: expected one match, found {prompt_count}"
    )

forward_pattern = re.compile(
    r'''        context_report = samples\["context_input_text"\]\n'''
    r'''        dynamic_prompt,[^\n]+self\.detector\.forward'''
    r'''\(current_image_embeds, context_image_embeds\)\n'''
    r'''(?:[ \t]*\n)?'''
    r'''        current_label = samples\["current_labels"\]\n'''
    r'''(?:[ \t]*\n)?'''
    r'''        current_img_embeds, atts_img = self\.prompt_wrap'''
    r'''\(current_image_embeds, atts_img, context_image_embeds, dynamic_prompt\)\n'''
)
model, forward_count = forward_pattern.subn(
    '''        current_label = samples["current_labels"]
        current_img_embeds, atts_img, sure_ot_output = (
            self._prepare_temporal_inputs(
                current_image_embeds,
                atts_img,
                context_image_embeds,
                compute_swap=self.training,
            )
        )
''',
    model,
    count=1,
)
if forward_count != 1:
    raise RuntimeError(
        f"training forward replacement: expected one match, found {forward_count}"
    )

model = replace_once(
    model,
    '''        loss_kl = F.kl_div(log_p_text, p_img, reduction='batchmean')
        loss = outputs.loss + 1 * (loss_kl + loss_cls_text) + 1 * loss_cls_image
        return {"loss": loss}
''',
    '''        loss_kl = F.kl_div(log_p_text, p_img, reduction='batchmean')
        loss = (
            outputs.loss
            + 1.0 * (loss_kl + loss_cls_text)
            + 1.0 * loss_cls_image
        )

        result = {
            "loss": loss,
            "loss_lm": outputs.loss.detach(),
            "loss_cls_text": loss_cls_text.detach(),
            "loss_cls_image": loss_cls_image.detach(),
            "loss_vl_kl": loss_kl.detach(),
        }
        if sure_ot_output is not None:
            swap_loss = sure_ot_output["swap_loss"]
            transport_loss = sure_ot_output["transport_loss"]
            regularization_loss = sure_ot_output["regularization_loss"]
            loss = (
                loss
                + getattr(self.args, "lambda_sure_ot_swap", 0.1)
                * swap_loss
                + getattr(self.args, "lambda_sure_ot_transport", 0.01)
                * transport_loss
                + getattr(self.args, "lambda_sure_ot_reg", 0.01)
                * regularization_loss
            )
            result.update(
                {
                    "loss": loss,
                    "loss_sure_ot_swap": swap_loss.detach(),
                    "loss_sure_ot_transport": transport_loss.detach(),
                    "loss_sure_ot_reg": regularization_loss.detach(),
                    "sure_ot_new_mass": sure_ot_output[
                        "new_score"
                    ].mean().detach(),
                    "sure_ot_resolved_mass": sure_ot_output[
                        "resolved_score"
                    ].mean().detach(),
                }
            )
        return result
''',
    "training loss integration",
)

validation_pattern = re.compile(
    r'''        dynamic_prompts,[^\n]+self\.detector\.forward'''
    r'''\(current_img_embeds, context_image_embeds\)\n'''
    r'''(?:[ \t]*\n)?'''
    r'''        context_report = samples\["context_input_text"\]\n'''
    r'''(?:[ \t]*\n)?'''
    r'''        current_img_embeds, atts_img = self\.prompt_wrap'''
    r'''\(current_img_embeds, atts_img, context_image_embeds, dynamic_prompts\)\n'''
)
model, validation_count = validation_pattern.subn(
    '''        current_img_embeds, atts_img, _ = (
            self._prepare_temporal_inputs(
                current_img_embeds,
                atts_img,
                context_image_embeds,
                compute_swap=False,
            )
        )
''',
    model,
    count=1,
)
if validation_count != 1:
    raise RuntimeError(
        f"validation integration: expected one match, found {validation_count}"
    )

test_pattern = re.compile(
    r'''        dynamic_prompts,[^\n]+self\.detector\.forward'''
    r'''\(current_image_embeds, context_image_embeds\)\n'''
    r'''(?:[ \t]*\n)?'''
    r'''        context_report = samples\["context_input_text"\]\n'''
    r'''(?:[ \t]*\n)?'''
    r'''        current_img_embeds, atts_img = self\.prompt_wrap'''
    r'''\(current_image_embeds, atts_img, context_image_embeds, dynamic_prompts\)\n'''
)
model, test_count = test_pattern.subn(
    '''        current_img_embeds, atts_img, _ = (
            self._prepare_temporal_inputs(
                current_image_embeds,
                atts_img,
                context_image_embeds,
                compute_swap=False,
            )
        )
''',
    model,
    count=1,
)
if test_count != 1:
    raise RuntimeError(
        f"test integration: expected one match, found {test_count}"
    )

model_path.write_text(model, encoding="utf-8")

config_path = Path("configs/config.py")
config = config_path.read_text(encoding="utf-8")
sure_ot_args = '''
# ========================= SURE-OT Settings ==========================
parser.add_argument('--use_sure_ot', default=True, type=lambda x: (str(x).lower() == 'true'), help='enable SURE-OT; False restores the upstream BiOTPrompt path')
parser.add_argument('--sure_ot_num_tokens', default=2, type=int, help='number of continuous tokens per evolution category')
parser.add_argument('--sure_ot_epsilon', default=0.07, type=float, help='entropic UOT regularization')
parser.add_argument('--sure_ot_tau', default=0.7, type=float, help='UOT marginal relaxation strength')
parser.add_argument('--sure_ot_iters', default=40, type=int, help='number of generalized Sinkhorn iterations')
parser.add_argument('--sure_ot_spatial_weight', default=0.05, type=float, help='patch-coordinate contribution to transport cost')
parser.add_argument('--sure_ot_adapter_rank', default=128, type=int, help='bottleneck rank for temporal role adapters')
parser.add_argument('--sure_ot_prior_strength', default=1.0, type=float, help='strength of residual priors in evolution-token attention')
parser.add_argument('--sure_ot_residual_threshold', default=0.25, type=float, help='threshold used only for diagnostic hard masks')
parser.add_argument('--sure_ot_balanced', default=False, type=lambda x: (str(x).lower() == 'true'), help='balanced-OT ablation; residual marginals should collapse')
parser.add_argument('--sure_ot_use_role_adapters', default=True, type=lambda x: (str(x).lower() == 'true'), help='use direction-aware current/history adapters')
parser.add_argument('--sure_ot_use_swap', default=True, type=lambda x: (str(x).lower() == 'true'), help='enable temporal swap consistency during training')
parser.add_argument('--lambda_sure_ot_swap', default=0.1, type=float, help='weight of swap consistency')
parser.add_argument('--lambda_sure_ot_transport', default=0.01, type=float, help='weight of the UOT objective')
parser.add_argument('--lambda_sure_ot_reg', default=0.01, type=float, help='weight of residual sparsity and spatial regularization')
'''
config, config_count = re.subn(
    r"(parser\.add_argument\('--end_sym'[^\n]*\)\n)",
    lambda match: match.group(1) + sure_ot_args + "\n",
    config,
    count=1,
)
if config_count != 1:
    raise RuntimeError(
        f"config integration: expected one --end_sym line, found {config_count}"
    )
config_path.write_text(config, encoding="utf-8")
