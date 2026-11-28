"""Cart E scratch: supervised autoregressive gaze conditions hd2 action/DINO FM."""
import torch
from .policy import CampaignPolicy
from .gaze import GazeTokens, Layout, BINS, quantize_gaze, gaze_ce
from .objectives import clean_encoder_view, flow_loss

class GazePolicy(CampaignPolicy):
    def __init__(self, reference, normalizer, robot, gaze_weight=.1):
        # Common modules (including fresh DINO) initialize exactly like paired hd2 B.
        super().__init__(reference, 'B', normalizer, robot)
        self.arm = 'E'
        self.gaze = GazeTokens(self.policy.embed_dim)
        self.gaze_weight = gaze_weight
        self.num_heads = self.policy.trunk['trunk'].blocks[0].attn.num_heads
        # Ordinary policy views keep the identical hd2 augmentation recipe.
        # Only the separate clean prefix is unaugmented, so its UV labels remain
        # in the original sensor frame (scoped clean_encoder_view in encode).

    def encode(self, batch):
        m = self.policy
        m.device = next(self.parameters()).device
        m.train_mode = self.training
        data = {k:v for k,v in batch.items() if k.startswith('front_pcd_') or k.startswith('state_')}
        _, features = m.stem_process('rby1', m.preprocess_states('rby1', data))
        context = torch.cat([features[k] for k in m.shared_keys + m.modalities['rby1']], dim=1)
        clean = batch['clean_cloud'].reshape(len(context), 1, 1, -1)
        with clean_encoder_view(m.encoders['front_pcd_1']):
            _, clean_tokens = m.get_visual_embeds('shared', {'front_pcd_1':clean}, 'front_pcd_1')
        prefix = torch.cat((clean_tokens, self.gaze.task.expand(len(context), -1, -1)), dim=1)
        return prefix, context

    def run_trunk(self, encoded, labels, valid, condition=True):
        prefix, context = encoded
        m = self.policy
        queries = m.action_tokens.expand(len(prefix), -1, -1)
        layout = Layout(prefix.shape[1], context.shape[1], queries.shape[1])
        tokens = torch.cat((prefix, self.gaze(labels), context, queries), dim=1)
        tokens = tokens + m.get_position_embedding(tokens, m.embed_dim)
        out, _ = m.trunk['trunk'](tokens, attn_mask=layout.mask(valid, self.num_heads, condition))
        return out[:, layout.action], self.gaze.logits(out, layout)

    def predicted_features(self, encoded, condition=True):
        prefix, _ = encoded
        labels = torch.zeros((len(prefix), 2), dtype=torch.long, device=prefix.device)
        valid = torch.ones(len(prefix), dtype=torch.bool, device=prefix.device)
        _, logits_x = self.run_trunk(encoded, labels, valid, condition)
        labels[:, 0] = logits_x[:, 0].argmax(-1)
        _, logits_y = self.run_trunk(encoded, labels, valid, condition)
        labels[:, 1] = logits_y[:, 1].argmax(-1)
        features, _ = self.run_trunk(encoded, labels, valid, condition)
        return features, labels

    def forward(self, batch):
        encoded = self.encode(batch)
        valid = batch['gaze_valid'].bool()
        labels = quantize_gaze(batch['gaze_uv'], valid)
        teacher_features, logits = self.run_trunk(encoded, labels, valid)
        ce = gaze_ce(logits, labels, valid)
        metrics = {'gaze_ce':ce.detach(), 'gaze_coverage':valid.float().mean()}
        if self.training:
            features = teacher_features
        else:
            # Main validation loss uses predicted gaze, never the labels/validity mask.
            features, predicted = self.predicted_features(encoded)
            uv = (predicted.float() + .5) / BINS
            err = (uv - torch.where(valid[:, None], batch['gaze_uv'], uv)).norm(dim=-1)
            metrics['gaze_uv_error'] = err.sum() / valid.sum().clamp_min(1)
        action, _, _ = flow_loss(self.policy.heads['rby1'], features, batch['action'], batch['action_valid'])
        dino = action * 0
        keep = batch['dino_valid'].bool()
        if keep.any():
            dino = self.policy.heads['rby1_dino_wm_future'].compute_loss(features[keep], {'action':batch['dino'][keep]})
        metrics.update(action=action.detach(), aux=dino.detach(), gaze_weighted=(ce*self.gaze_weight).detach())
        return action + dino + (self.gaze_weight*ce if self.training else 0.), metrics

    @torch.no_grad()
    def predict_actions(self, observations, noise=None):
        if self.training: raise RuntimeError('Call eval() before deployment inference')
        features, labels = self.predicted_features(self.encode(observations))
        head = self.policy.heads['rby1']
        if noise is None: noise = torch.randn(len(features), 64, 49, device=features.device, dtype=features.dtype)
        actions = head.inference(noise, features)  # one fixed context throughout integration
        return {'action':actions.float()*self.action_scale+self.action_center,
                'gaze_uv':(labels.float()+.5)/BINS}

    def transfer_policy(self, source_state):
        keys = super().transfer_policy(source_state)
        own = self.state_dict()
        for k in own:
            if k.startswith('gaze.'):
                if k not in source_state or source_state[k].shape != own[k].shape:
                    raise ValueError('Gaze transfer mismatch: '+k)
                own[k] = source_state[k]
                keys.append(k)
        self.load_state_dict(own, strict=True)
        return keys
