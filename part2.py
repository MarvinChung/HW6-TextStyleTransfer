import os
import torch
import torch.nn.functional as F
import time 
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
import argparse
from train import batch_preprocess, get_lengths
from utils import tensor2text
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def log(f, *args):
    print(*args)
    print(*args, file=f)

def plot_attn(attention, inp_token, style, out_token, vocab, ax, title=None):

    style_text = '[neg]' if style == 0 else '[pos]'
    inp_text = [style_text] + [vocab.itos[i] for i in inp_token]
    out_text = [vocab.itos[i] for i in out_token]
    
    inp_length = inp_text.index('<eos>')
    out_length = out_text.index('<eos>')

    inp_text = inp_text[:inp_length]
    out_text = out_text[:out_length]
    attention = attention[:out_length, :inp_length]
    attention /= torch.sum(attention, dim=-1, keepdim=True)

    im = ax.imshow(attention, cmap="YlGn")

    ax.set_xticks(np.arange(inp_length))
    ax.set_yticks(np.arange(out_length))

    ax.set_xticklabels(inp_text)
    ax.set_yticklabels(out_text)

    ax.set_title(title)

def find_gradient(model, log_prob, inp_token, style, out_token):
    
    loss = F.nll_loss(log_prob, out_token, reduction='none')

    token_emb_weight = model.embed.token_embed.weight
    style_emb_weight = model.style_embed.weight

    gradient_list = []
    for i in range(len(out_token)):
        model.zero_grad()
        loss[i].backward(retain_graph=True)

        full_tok_gradients = token_emb_weight.grad
        full_style_gradients = style_emb_weight.grad

        tok_gradients = torch.index_select(full_tok_gradients, dim=0, index=inp_token)
        style_gradients = full_style_gradients[style]

        inp_gradients = torch.cat((style_gradients.unsqueeze(0), tok_gradients), 0).detach()
        gradient_list.append(inp_gradients)

    gradient = torch.stack(gradient_list).cpu()
    return gradient

def plot_gradient_norm(gradient, inp_token, style, out_token, vocab, ax, title):
    
    style_text = '[neg]' if style == 0 else '[pos]'
    inp_text = [style_text] + [vocab.itos[i] for i in inp_token]
    out_text = [vocab.itos[i] for i in out_token]
    
    inp_length = inp_text.index('<eos>')
    out_length = out_text.index('<eos>')

    inp_text = inp_text[:inp_length]
    out_text = out_text[:out_length]
    gradient = gradient[:out_length, :inp_length]

    gradient_norm = gradient.norm(dim=-1)
    gradient_norm /= torch.sum(gradient_norm, dim=-1, keepdim=True) # normalize

    ax.imshow(gradient_norm)
    ax.set_xticks(np.arange(inp_length))
    ax.set_yticks(np.arange(out_length))
    ax.set_xticklabels(inp_text)
    ax.set_yticklabels(out_text)
    ax.set_title(title)
    # ax.set(xlabel='input', ylabel='output')

def part2(args):

    ## load model
    #model_prefix = './save/Feb15203331/ckpts/1300'
    model_prefix = os.path.join(args.part2_model_dir, str(args.part2_step))

    args.preload_F = f'{model_prefix}_F.pth'
    args.preload_D = f'{model_prefix}_D.pth'

    ## load data
    train_iters, dev_iters, test_iters, vocab = load_dataset(args)

    ## output dir
    output_dir = 'part2_output'
    os.makedirs(output_dir, exist_ok = True)

    log_f = open(os.path.join(output_dir, 'log.txt'), 'w')

    model_F = StyleTransformer(args, vocab).to(args.device)
    model_D = Discriminator(args, vocab).to(args.device)

    assert os.path.isfile(args.preload_F)
    model_F.load_state_dict(torch.load(args.preload_F))
    assert os.path.isfile(args.preload_D)
    model_D.load_state_dict(torch.load(args.preload_D))
    
    model_F.eval()
    model_D.eval()

    dataset = test_iters
    pos_iter = dataset.pos_iter
    neg_iter = dataset.neg_iter

    pad_idx = vocab.stoi['<pad>'] # 1
    eos_idx = vocab.stoi['<eos>'] # 2
    unk_idx = vocab.stoi['<unk>'] # 0

    # we would use the s_id-th example
    #use sample_id = 29 would get the same results in slide
    sample_id = np.random.randint(args.batch_size)
    log(log_f, f'sample id : {sample_id}')
    batch = next(iter(pos_iter))
    sample_inp_token = batch.text[sample_id]
    sample_inp_length = get_lengths(batch.text, eos_idx)[sample_id]
    sample_raw_style = 1

    ## 2-1 attention
    log(log_f, "***** 2-1: Attention & Gradient norm *****")

    attn_weight = None

    inp_token = sample_inp_token
    inp_tokens = torch.stack((inp_token, inp_token))
    inp_lengths = get_lengths(inp_tokens, eos_idx)
    raw_style = sample_raw_style
    styles = torch.tensor([raw_style, 1-raw_style]).type_as(inp_tokens)

    log_probs = model_F(
        inp_tokens, 
        None,
        inp_lengths,
        styles,
        generate=True,
        differentiable_decode=False,
        temperature=1,
    )

    recon_log_prob = log_probs[0]
    recon_out_token = torch.argmax(recon_log_prob, dim=-1)
    rev_log_prob = log_probs[1]
    rev_out_token = torch.argmax(rev_log_prob, dim=-1)

    inp_text = [vocab.itos[i] for i in inp_token]
    recon_out_text = [vocab.itos[i] for i in recon_out_token]
    rev_out_text = [vocab.itos[i] for i in rev_out_token]

    inp_len = inp_text.index('<eos>')
    recon_out_len = recon_out_text.index('<eos>')
    rev_out_len = rev_out_text.index('<eos>')

    log(log_f, '[inp]:', ' '.join(inp_text[:inp_len]))
    log(log_f, '[rec]:', ' '.join(recon_out_text[:recon_out_len]))
    log(log_f, '[rev]:', ' '.join(rev_out_text[:rev_out_len]))
    
    ### part a: attention
    log(log_f, "part a: attention")
    rev_attn = model_F.get_decode_src_attn_weight()

    if attn_weight == None:
        attn_weight = rev_attn
    else:
        for layer in range(len(rev_attn)):
            attn_weight[layer] = torch.cat([attn_weight[layer], rev_attn[layer]])

    # shape of rev_attn (layer, batch, head, out_len, inp_len+1)

    for i in range(len(styles)):
        style = styles[i]
        title = 'recon' if style == raw_style else 'reverse'
        out_token = recon_out_token if style == raw_style else rev_out_token
        for layer in range(len(attn_weight)):
            ## if you use different number of heads, you have to change here
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            for head in range(4):
                ax = axs[head//2][head%2]
                attn = attn_weight[layer][i][head]
                plot_attn(attn, inp_token, styles[i], out_token, vocab, ax, f'head {head+1}')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            fig.colorbar(axs[0][0].get_images()[0], ax=axs)
            fig.suptitle(f'cross attention @ layer {layer+1} (x: input, y: output)', fontsize=20)
            output_name = f'{output_dir}/{title}_attn_layer{layer+1}.png'
            fig.savefig(output_name)
            log(log_f, f'save cross attention figure at {output_name}')
    
    ### part b: gradient norm
    log(log_f, 'part b: gradient norm')

    recon_gradient = find_gradient(model_F, recon_log_prob, inp_token, styles[0], recon_out_token)
    rev_gradient = find_gradient(model_F, rev_log_prob, inp_token, styles[1], rev_out_token)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Gradient norm (x: input, y: output)",fontsize=20)

    plot_gradient_norm(recon_gradient, inp_token, styles[0], recon_out_token, vocab, ax1, "reconstruction")
    plot_gradient_norm(rev_gradient, inp_token, styles[1], rev_out_token, vocab, ax2, "reverse")

    ## ploting config
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.colorbar(ax1.get_images()[0], ax=(ax1, ax2))
    output_name = f'{output_dir}/gradient_norm.png'
    fig.savefig(output_name)
    log(log_f, f'save gradient norm figure at {output_name}')
    log(log_f, '***** 2-1 end *****')
    log(log_f)


    # 2-2. mask input tokens
    log(log_f, '***** 2-2: mask input *****')
    raw_style = sample_raw_style

    inp_token = sample_inp_token
    inp_length = sample_inp_length
    
    inp_tokens = inp_token.repeat(inp_length, 1) ## mask until '. <eos>' but contain the origin sentence
    for i in range(inp_tokens.shape[0]-1):
        inp_tokens[i+1][i] = unk_idx

    inp_lengths = torch.full_like(inp_tokens[:, 0], inp_length)
    raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
    rev_styles = 1 - raw_styles

    with torch.no_grad():
        rev_log_probs = model_F(
            inp_tokens, 
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=False,
            temperature=1
        )

    # gold_text = tensor2text(vocab, inp_tokens.cpu(), remain_unk=True)
    rev_idx = rev_log_probs.argmax(-1).cpu()
    # rev_output = tensor2text(vocab, rev_idx, remain_unk=True)

    gold_text = []
    rev_output = []

    for i in range(len(rev_idx)):
        inp_text = [vocab.itos[w] for w in inp_tokens[i]]
        rev_text = [vocab.itos[w] for w in rev_idx[i]]

        inp_len = inp_text.index('<eos>')
        rev_len = rev_text.index('<eos>')

        gold_text.append(' '.join(inp_text[:inp_len]))
        rev_output.append(' '.join(rev_text[:rev_len]))

    for i in range(len(gold_text)):
        log(log_f, '-')
        log(log_f, '[ORG]', gold_text[i])
        log(log_f, '[REV]', rev_output[i])

    log(log_f, '***** 2-2 end *****')
    log(log_f)

    ## 2-3. tsne
    log(log_f, "***** 2-3: T-sne *****")
    features = []
    labels = []

    for batch in pos_iter:

        inp_tokens = batch.text
        inp_lengths = get_lengths(inp_tokens, eos_idx)

        _, pos_features = model_D(inp_tokens, inp_lengths, return_features=True)
        features.extend(pos_features.detach().cpu().numpy())
        labels.extend([0 for i in range(pos_features.shape[0])])

        raw_style = 1
        raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
        rev_styles = 1 - raw_styles

        with torch.no_grad():
            rev_log_probs = model_F(
                inp_tokens, 
                None,
                inp_lengths,
                rev_styles,
                generate=True,
                differentiable_decode=False,
                temperature=1
            )

        rev_tokens = rev_log_probs.argmax(-1)
        rev_lengths = get_lengths(rev_tokens, eos_idx)
        _, rev_features = model_D(rev_tokens, inp_lengths, return_features=True)
        features.extend(rev_features.detach().cpu().numpy())
        labels.extend([1 for i in range(rev_features.shape[0])])
        

    for batch in neg_iter:

        inp_tokens = batch.text
        inp_lengths = get_lengths(inp_tokens, eos_idx)

        _, neg_features = model_D(inp_tokens, inp_lengths, return_features=True)
        features.extend(neg_features.detach().cpu().numpy())
        labels.extend([2 for i in range(neg_features.shape[0])])

        raw_style = 0
        raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
        rev_styles = 1 - raw_styles

        with torch.no_grad():
            rev_log_probs = model_F(
                inp_tokens, 
                None,
                inp_lengths,
                rev_styles,
                generate=True,
                differentiable_decode=False,
                temperature=1
            )

        rev_tokens = rev_log_probs.argmax(-1)
        rev_lengths = get_lengths(rev_tokens, eos_idx)
        _, rev_features = model_D(rev_tokens, inp_lengths, return_features=True)
        features.extend(rev_features.detach().cpu().numpy())
        labels.extend([3 for i in range(rev_features.shape[0])])

    labels = np.array(labels)
    colors = ['red', 'blue', 'orange', 'green']
    classes = ['POS', 'POS -> NEG', 'NEG', 'NEG -> POS']
    X_emb = TSNE(n_components=2).fit_transform(features)
    
    fig, ax = plt.subplots()
    for i in range(4):
        idxs = labels == i
        ax.scatter(X_emb[idxs, 0], X_emb[idxs, 1], color=colors[i], label=classes[i], alpha=0.8, edgecolors='none')
    ax.legend()
    ax.set_title('t-sne of four distributions')
    output_name = os.path.join(output_dir, 'tsne.png')
    plt.savefig(output_name)
    log(log_f, f'save T-sne figure at {output_name}')
    log(log_f, "***** 2-3 end *****")
    log(log_f)

    log_f.close()