import torch
import torch.nn.functional as F
def lalign_kl(x_,y_):
    total_loss = 0.0
    for i, (x, y) in enumerate(zip(x_, y_)):
        B, _ = x.shape
        y_expanded = y.expand(B, -1)
        x_prob = F.softmax(x, dim=1)
        y_prob = F.softmax(y_expanded, dim=1)
        loss = F.kl_div(F.log_softmax(x_prob, dim=1), y_prob, reduction='batchmean')
        total_loss += loss
    return total_loss / len(x_)
def lalign(x, y, alpha=2):
    diff = x - y
    norms = torch.norm(diff, dim=1, p=2)
    powered_norms = norms.pow(alpha)
    loss = powered_norms.mean()
    return loss

def lalign_kl2(alignment_list, prompt_list):
    loss = 0.0
    eps = 1e-8
    for align, prompt in zip(alignment_list, prompt_list):
        B, N1, D = align.shape
        B, N2, D = prompt.shape
        align_dist = align.mean(dim=1)  # [B, D]
        prompt_dist = prompt.mean(dim=1)  # [B, D]
        p = F.softmax(align_dist, dim=-1)
        q = F.softmax(prompt_dist, dim=-1)
        p = p + eps
        p = p / p.sum(dim=-1, keepdim=True)
        loss += F.kl_div(q.clamp_min(1e-8).log(), p, reduction='batchmean')

    return loss / len(alignment_list)

def lalign_maxmin(prompt_list, text_feat):
    loss = 0.0
    B_text, D_text = text_feat.shape

    text_norms = text_feat.norm(dim=-1, keepdim=True)  # [B, 1]

    for prompt in prompt_list:
        B, N, D = prompt.shape
        assert B == B_text and D == D_text, f"Shape mismatch: {prompt.shape} vs {text_feat.shape}"

        prompt_norms = prompt.norm(dim=-1)  # [B, N]

        target_norms = text_norms.expand(B, N)  # [B, N]

        loss += F.mse_loss(prompt_norms, target_norms)

    return loss / len(prompt_list)
# #

# def lalign_maxmin(prompt_list, text_feat):

#     total_loss = 0.0
#     B_text, D_text = text_feat.shape
#
#     text_norm_mean = text_feat.norm(dim=-1).mean()  # [1]
#
#     for prompt in prompt_list:
#         B, N, D = prompt.shape
#         assert B == B_text and D == D_text, f"Shape mismatch: {prompt.shape} vs {text_feat.shape}"
#
#         prompt_norm_mean = prompt.norm(dim=-1).mean()  # [1]
#
#         total_loss += torch.abs(prompt_norm_mean - text_norm_mean)
#
#     return total_loss / len(prompt_list) 
