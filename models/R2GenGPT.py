import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel,AutoModelForCausalLM, GPT2TokenizerFast, \
    BertModel
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from evalcap.transformer import Transformer
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import copy
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from evalcap.metrics_clinical import CheXbertMetrics
from torchvision.transforms import functional as TF
from PIL import Image
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from evalcap.BiOT import SinkhornOTChangeDetector

class VisionClassifier(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):  # x: [B, D]
        return self.classifier(x)

class TextClassifier(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):  # x: [B, D]
        return self.classifier(x)

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chexbert_metrics = CheXbertMetrics('./data/chexbert.pth', args.batch_size,torch.device(args.device))
        self.detector = SinkhornOTChangeDetector(eps=0.1, max_iter=50, thresh_mode=0.05, k_ratio=0.1, global_q=0.9, init_alpha=10.0)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()

            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)  # 分词器
        self.llama_tokenizer.pad_token_id = 0

        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            print("Begining to load BiMEdiX")

            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )

        print("Successfully to load BiMEdiX")
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')
        else:
            print("Beginning to load Input_Embedding")

            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.cls_head_visual = VisionClassifier(4096,14 * 4)
        self.cls_head_text = TextClassifier(4096,14 * 4)
        self.criterion_cls = nn.CrossEntropyLoss()

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'),weights_only=False)['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        device = images.device
        with torch.set_grad_enabled(True):
            image_embed = self.visual_encoder(images)['last_hidden_state'].to(device)  # 12*49*1024
            image_embed_pooler = self.visual_encoder(images)['pooler_output'].to(device)

        inputs_llama = self.llama_proj(image_embed)
        inputs_llama_pooler = self.llama_proj(image_embed_pooler)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama, inputs_llama_pooler,atts_llama

    def prompt_wrap(self, img_embeds, atts_img, context_img_embeds, dynamic_prompt):
        batch_size = img_embeds.shape[0]

        p_1 = 'Human: <Img>'
        p_2 = [f'</Img> Generate a comprehensive and detailed diagnosis report for this chest xray image. {dynamic_prompt[i]}' for i in range(batch_size)]
        p_3 = '\nHere is the historical chest xray image: <Img>'
        p_4 = '</Img> for reference \nAssistant:'

        p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_3_tokens = self.llama_tokenizer(p_3, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_4_tokens = self.llama_tokenizer(p_4, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        dynamic_prompt_tokens = self.llama_tokenizer(
            p_2, return_tensors="pt", add_special_tokens=False, padding="max_length",
            truncation=True,
            max_length=60).to(img_embeds.device)

        p_1_embeds = self.embed_tokens(p_1_tokens.input_ids).expand(batch_size, -1, -1)
        p_2_embeds = self.embed_tokens(dynamic_prompt_tokens.input_ids)
        p_3_embeds = self.embed_tokens(p_3_tokens.input_ids).expand(batch_size, -1, -1)
        p_4_embeds = self.embed_tokens(p_4_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_1_embeds, img_embeds, p_2_embeds, p_3_embeds, context_img_embeds, p_4_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, img_embeds_pooler, atts_img = self.encode_img(image)
        current_image_embeds = self.layer_norm(img_embeds)

        context_image = samples["context_image"]
        context_img_embeds, context_img_embeds_pooler, context_atts_img = self.encode_img(context_image)
        context_image_embeds = self.layer_norm(context_img_embeds)

        context_report = samples["context_input_text"]
        dynamic_prompt, new_mask, disappear_mask, sinkhorn_loss_p2n, sinkhorn_loss_n2p = self.detector.forward(current_image_embeds, context_image_embeds)

        current_label = samples["current_labels"]

        current_img_embeds, atts_img = self.prompt_wrap(current_image_embeds, atts_img, context_image_embeds, dynamic_prompt)

        self.llama_tokenizer.padding_side = "right"
        #替换这个部分
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = current_img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, current_img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        text_hidden = outputs.hidden_states[-1][:, -self.hparams.max_length:, :]  # [B, L, D]
        current_text_embed = self.layer_norm(text_hidden.mean(dim=1))  # [B, D]

        cls_preds_text = self.cls_head_text(current_text_embed)
        cls_preds_text = cls_preds_text.view(-1, 4, 14)
        loss_cls_text = self.criterion_cls(cls_preds_text, current_label[:,:14])

        cls_preds_image = self.cls_head_visual(img_embeds_pooler)
        cls_preds_image = cls_preds_image.view(-1, 4, 14)
        loss_cls_image = self.criterion_cls(cls_preds_image, current_label[:, :14])

        cls_preds_text_kl = cls_preds_text.permute(0, 2, 1)
        cls_preds_image_kl = cls_preds_image.permute(0, 2, 1)

        log_p_text = F.log_softmax(cls_preds_text_kl, dim=-1)
        p_img = F.softmax(cls_preds_image_kl.detach(), dim=-1)

        loss_kl = F.kl_div(log_p_text, p_img, reduction='batchmean')
        loss = outputs.loss + 1 * (loss_kl + loss_cls_text) + 1 * loss_cls_image
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'],
                                                                        eval_res['ROUGE_L']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, img_embeds_pooler, atts_img = self.encode_img(image)
        current_img_embeds = self.layer_norm(img_embeds)

        context_image = samples["context_image"]
        context_img_embeds, context_img_embeds_pooler, context_atts_img = self.encode_img(context_image)
        context_image_embeds = self.layer_norm(context_img_embeds)

        dynamic_prompts, new_mask, disappear_mask, sinkhorn_loss_p2n, sinkhorn_loss_n2p = self.detector.forward(current_img_embeds, context_image_embeds)

        context_report = samples["context_input_text"]

        current_img_embeds, atts_img = self.prompt_wrap(current_img_embeds, atts_img, context_image_embeds, dynamic_prompts)

        batch_size = current_img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, current_img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        generated_tokens = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature
        )

        hypo = [self.decode(i) for i in generated_tokens]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text.replace('<s>', '')  # 移除起始标记

        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref1 = {k: [v] for k, v in zip(ids, ref)}
        hypo1 = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref1, hypo=hypo1)
        eval_ce = self.chexbert_metrics.compute(ref, hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)
        self.print(eval_ce)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, img_embeds_pooler, atts_img = self.encode_img(image)
        current_image_embeds = self.layer_norm(img_embeds)

        context_image = samples["context_image"]
        context_img_embeds, context_img_embeds_pooler, context_atts_img = self.encode_img(context_image)
        context_image_embeds = self.layer_norm(context_img_embeds)

        dynamic_prompts, new_mask, disappear_mask, sinkhorn_loss_p2n, sinkhorn_loss_n2p = self.detector.forward(current_image_embeds, context_image_embeds)

        context_report = samples["context_input_text"]

        current_img_embeds, atts_img = self.prompt_wrap(current_image_embeds, atts_img, context_image_embeds, dynamic_prompts)
        batch_size = current_img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, current_img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        generated_tokens = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature
        )

        hypo = [self.decode(i) for i in generated_tokens]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref1 = {k: [v] for k, v in zip(ids, ref)}
        hypo1 = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref1, hypo=hypo1)
        eval_ce = self.chexbert_metrics.compute(ref, hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        self.print(f"Test result of {self.hparams.delta_file}: {eval_ce}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()