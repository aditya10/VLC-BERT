import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

def prepare_mask(key_mask, query_mask):

    len_k = key_mask.size(1)
    len_q = query_mask.size(1)

    padding_mask1 = query_mask.unsqueeze(1).expand(-1, len_k, -1).transpose(1,2)
    padding_mask2 = key_mask.unsqueeze(1).expand(-1, len_q, -1)

    return padding_mask1*padding_mask2

class SimpleFusionLayer(nn.Module):
    def __init__(self, config):
        super(SimpleFusionLayer, self).__init__()

        self.num_heads = config.num_heads
        # this is the dimension of the feature vector for both query and key/value
        self.embed_dim = config.hidden_size
        self.reduce_attention_output = config.reduce_attention_output

        self.attend_ques = config.attend_ques

        if self.reduce_attention_output:
            self.dim_layer = nn.Linear(self.embed_dim, self.embed_dim)

        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.fusion = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0.1, batch_first=True)
        self.fusion.apply(init_weights)

    def forward(self, embeddings, image_fusion=False):
        """
         If specified, a mask of shape (N, S) indicating which elements within key to ignore for the purpose of attention (i.e. treat as "padding").
         embeddings: tensor of shape (N, S, D) where the last item in the sequence is the query, eg. S = 6 when first 5 are commonsense embeddings and last 1 is question
        """
        # note: you can also do batch size as first dimension, by setting batch_first=True.
        # SHAPE: Batch size * Sequence length * embedding dimension

        if image_fusion:
            first = embeddings[:, -2:, :]
        else:   
            first = embeddings[:, -1, :].unsqueeze(1)

        if not self.attend_ques:
            embeddings = embeddings[:,:-1,:]

        # Normalize the input
        first = self.norm(first)
        second = self.norm(embeddings)

        query = first
        key = second
        value = second
        
        attn_output, attn_output_weights = self.fusion(query, key, value)
        
        if self.reduce_attention_output:
            # reduce the attention output to 1024 dimensions
            attn_output = self.dim_layer(attn_output)

        return attn_output, attn_output_weights