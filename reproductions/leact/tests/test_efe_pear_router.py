import pytest

from reproductions.leact.cluster.efe_pear_router import assert_expert_routers_frozen,freeze_expert_routers


def test_optimizer_updates_experts_and_backbone_but_keeps_router_exactly_fixed():
    torch=pytest.importorskip('torch');nn=torch.nn
    torch.manual_seed(7)
    class TinyMoE(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Linear(4,4)
            self.mlp=nn.Module();self.mlp.gate=nn.Linear(4,3,bias=False)
            self.mlp.gate_proj=nn.Linear(4,3);self.mlp.shared_expert_gate=nn.Linear(4,1)
        def forward(self,x):
            x=self.encoder(x)
            return (self.mlp.gate(x).softmax(-1)*self.mlp.gate_proj(x)).sum(-1)+self.mlp.shared_expert_gate(x).squeeze(-1)
    model=TinyMoE();receipt=freeze_expert_routers(model)
    assert receipt['names']==['mlp.gate.weight']
    before={name:p.detach().clone()for name,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    model(torch.randn(8,4)).square().mean().backward();optimizer.step()
    assert model.mlp.gate.weight.grad is None
    assert torch.equal(before['mlp.gate.weight'],model.mlp.gate.weight)
    for name,p in model.named_parameters():
        if name!='mlp.gate.weight':assert not torch.equal(before[name],p)
    assert_expert_routers_frozen(model)
    model.mlp.gate.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='trainable'):assert_expert_routers_frozen(model)
    with pytest.raises(ValueError,match='No expert-selection'):freeze_expert_routers(nn.Linear(4,4))


def test_router_matching_excludes_other_gates_and_keeps_backpropagation_enabled():
    class Parameter:
        def __init__(self):self.requires_grad=True
        def requires_grad_(self,value):self.requires_grad=value
        def numel(self):return 12
    class Model:
        def __init__(self):self.parameters={name:Parameter()for name in [
            'model.language_model.layers.0.mlp.gate.weight','model.layers.0.mlp.gate_proj.weight',
            'model.layers.0.mlp.shared_expert_gate.weight','model.layers.0.linear_attn.in_proj_b.weight']}
        def named_parameters(self):return self.parameters.items()
    model=Model();receipt=freeze_expert_routers(model)
    assert receipt['parameters']==12
    assert not model.parameters[receipt['names'][0]].requires_grad
    assert all(p.requires_grad for name,p in model.named_parameters()if name not in receipt['names'])
    assert_expert_routers_frozen(model)
