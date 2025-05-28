import torch
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper

def calculate_jsd(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(F.log_softmax(p_logits, dim=-1), m, reduction='batchmean')
    kl_qm = F.kl_div(F.log_softmax(q_logits, dim=-1), m, reduction='batchmean')

    return 0.5 * (kl_pm + kl_qm)

def generate(
    model,
    input_ids: torch.Tensor,
    # past_key_values,
    max_new_tokens: int = 300,
    # mature_layer: int = 18,
    mature_layer: int = 6,
    candidate_premature_layers: list = [0,2,4],
    # candidate_premature_layers: list= [0,2,4],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    kl_weight: float = 1.0,  # Scaling factor for divergence
):
    # device = model.device
    # output_ids = input_ids.clone().to(device)
    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids
    logits_processor = LogitsProcessorList()
    premature_layer_dist = {l:0 for l in candidate_premature_layers}
    premature_layers = []

    # Initialize logits processors
    # if temperature != 1.0:
    #     logits_processor.append(TemperatureLogitsWarper(temperature))
    # if top_k > 0:
    #     logits_processor.append(TopKLogitsWarper(top_k))
    # if top_p < 1.0:
    #     logits_processor.append(TopPLogitsWarper(top_p))

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass to get logits from all layers
            # outputs = model(
            #     input_ids=input_ids,
            #     past_key_values=past_key_values,
            #     use_cache=True,
            # )


            logits_dict, outputs = model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=True,
                    # use_cache=True,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

            # logits_dict = outputs.logits_dict

            # if isinstance(outputs, tuple):  # When early_exit_layers is used
            #     logits_dict, outputs = outputs
            # else:  # When early_exit_layers is None
            #     logits_dict = None  # No early exit logits in this case

            # logits_dict = {}
            # for layer in candidate_premature_layers + [mature_layer]:
            #     logits = model.lm_head(outputs.hidden_states[layer])
            #     logits_dict[layer] = logits

            # print(f"mature_layer:{mature_layer}")
            # print(f"candidate_premature_layers:{candidate_premature_layers}")
            # print(f"logits_dict:{logits_dict}")
            # print(f"outputs.logits: {outputs.logits}")
            # if logits_dict is not None:
            #     mature_logits = logits_dict.get(mature_layer, None)
            #     if mature_logits is not None:
            #         print(f"Mature logits shape: {mature_logits.shape}")
            #     else:
            #         print(f"No mature logits found for layer {mature_layer}")
            # else:
            #     print("No logits_dict found!")


            # print(f"logits_dict type: {type(logits_dict)}")
            # print(f"outputs type: {type(outputs)}")
            # # print(f"outputs type: {outputs.shape}")
            # # print(f"outputs: {outputs}")
            # # print(f"logits_dict type: {logits_dict.shape}")
            # print(f"outputs type: {type(outputs)}")
            # print(f"outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'No keys'}")
            # if hasattr(outputs, "logits"):
            #   print(f"Logits shape: {outputs.logits.shape}")

            # print(f"logits_dict type: {type(logits_dict)}")
            # if logits_dict:
            #     for layer, logits in logits_dict.items():
            #         print(f"Layer {layer} logits shape: {logits.shape}")

            # Get mature layer logits
            if logits_dict is None or mature_layer not in logits_dict:
              raise ValueError(f"logits_dict is None or missing key {mature_layer}. Check early_exit_layers.")

            mature_logits = logits_dict[mature_layer][:, -1, :]

            # Calculate JSD for all candidate layers
            jsd_values = []
            for layer in candidate_premature_layers:
                premature_logits = logits_dict[layer][:, -1, :]
                jsd = calculate_jsd(mature_logits, premature_logits)
                jsd_values.append(jsd)

            # Convert to tensor and find max index
            jsd_tensor = torch.tensor(jsd_values)
            max_jsd_idx = torch.argmax(jsd_tensor).item()
            premature_layer = candidate_premature_layers[max_jsd_idx]

            # Update distribution tracking
            premature_layer_dist[premature_layer] += 1
            premature_layers.append(premature_layer)

            # Compute contrastive logits and sample
            contrastive_logits = mature_logits - logits_dict[premature_layer][:, -1, :]
            contrastive_logits = logits_processor(input_ids, contrastive_logits)
            probs = F.softmax(contrastive_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # # Select layer with max JSD
            # max_jsd_idx = torch.argmax(torch.tensor(jsd_values))
            # selected_premature_logits = logits_dict[
            #     candidate_premature_layers[max_jsd_idx]
            # ][:, -1, :]

            # # Compute contrastive logits
            # contrastive_logits = mature_logits - selected_premature_logits

            # # Apply temperature/top-k/p
            # contrastive_logits = logits_processor(input_ids, contrastive_logits)

            # # Sample next token
            # probs = F.softmax(contrastive_logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)

            # premature_layer = candidate_premature_layers[int(jsd_values.argmax().cpu().item())]
            # premature_layer_dist[premature_layer] += 1

            # premature_layers.append(premature_layer)

            # # Get mature layer logits (final layer)
            # mature_logits = outputs.logits[:, -1, :]
            # mature_probs = F.softmax(mature_logits, dim=-1)

            # # Initialize divergence adjustment
            # kl_adjustment = torch.zeros_like(mature_logits)

            # # Compute KL contributions for each premature layer
            # for layer in candidate_premature_layers:
            #     premature_logits = outputs.logits_dict[layer][:, -1, :]
            #     premature_probs = F.softmax(premature_logits, dim=-1)

            #     # KL(Premature || Mature) contribution per token
            #     kl_contribution = premature_probs * (
            #         torch.log(premature_probs + 1e-8) - torch.log(mature_probs + 1e-8)
            #     )
            #     kl_adjustment += kl_contribution

            # # Adjust mature logits with KL divergence
            # adjusted_logits = mature_logits + kl_weight * kl_adjustment
            # adjusted_logits = logits_processor(input_ids, adjusted_logits)

            # # Sample next token
            # probs = F.softmax(adjusted_logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)

            # Update tracking variables
            output_ids = torch.cat([output_ids, next_token], dim=1)
            input_ids = next_token
            past_key_values = outputs.past_key_values

            if next_token.item() == model.config.eos_token_id:
                break

    return output_ids[:, origin_ids.shape[-1]:], premature_layer_dist

if __name__ == "__main__":
    model_name = ""
    hf_token = ""

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
    answer_instruction = ""
    question = ""
    prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    ------------------------------------------------
    {answer_instruction}
    Question:
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    generate_t1 = time()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = generate(model, input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
    generate_t2 = time()

    print(f"generated text: {generated_text}")