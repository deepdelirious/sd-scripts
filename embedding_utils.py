import logging
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

def inject_embedding(model, strategy, tokenizer, placeholder, embed_file, embed_key):
    embed_state_dict = load_file(embed_file)
    if not embed_key in embed_state_dict:
        raise Exception(f"{embed_key} not found in {embed_file}")
    embed = embed_state_dict[embed_key]
    placeholders = [f"{placeholder.replace(' ', '_')}_{i}" for i in range(0, len(embed))]
    tokenizer.add_tokens(placeholders)
    indexes = tokenizer.convert_tokens_to_ids(placeholders)
    if (model.get_input_embeddings().num_embeddings <= len(tokenizer)):
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Expanded model embeddings to : {model.get_input_embeddings().num_embeddings}")
    for e, i in zip(embed, indexes):
        model.get_input_embeddings().weight.data[i] = e
    logger.info(f"Added custom embedding for {placeholder} to {embed_key} as token(s) {indexes}")
    strategy.add_replacement(placeholder, " ".join(placeholders))