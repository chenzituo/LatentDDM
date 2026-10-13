"""Numerical checks for convolutional adapters and the shared field objective."""
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment import ConvLoRA
from codiffuse_Schwarz_Darcy.budget_protocol import shared_field_loss
from codiffuse_Schwarz_Darcy.train_input_correction_latentddm import gradient_mse
from codiffuse_Schwarz_Darcy.train_residual_port_chart import masked_mse


def test_zero_delta_reproduces_convolution_and_input_derivative():
    torch.manual_seed(7)
    for dilation in [1,2,4]:
        base=nn.Conv2d(5,8,3,padding=dilation,dilation=dilation)
        adapter=ConvLoRA(base,3)
        x=torch.randn(2,5,9,11,requires_grad=True)
        expected=base(x);actual=adapter(x)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        g0=torch.autograd.grad(expected.square().sum(),x,retain_graph=True)[0]
        g1=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        torch.testing.assert_close(g1,g0,rtol=0,atol=0)
        actual.square().sum().backward()
        assert adapter.lora_b.grad.abs().sum()>0
        assert base.weight.grad is None and base.bias.grad is None


def test_nonzero_low_rank_kernel_matches_explicit_dense_update():
    torch.manual_seed(8)
    base=nn.Conv2d(4,7,5,padding=2)
    adapter=ConvLoRA(base,2)
    with torch.no_grad():adapter.lora_b.normal_()
    x=torch.randn(2,4,8,8)
    expected=base(x)+torch.nn.functional.conv2d(x,(adapter.lora_b@adapter.lora_a).reshape_as(base.weight),padding=2)
    torch.testing.assert_close(adapter(x),expected,rtol=2e-5,atol=2e-5)
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad)==2*(4*25+7)


def test_scalar_output_caps_rank():
    adapter=ConvLoRA(nn.Conv2d(128,1,3,padding=1),24)
    assert adapter.rank==1
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad)==1153


def test_shared_loss_matches_legacy_composer_output_terms():
    torch.manual_seed(9)
    prediction=torch.randn(2,1,16,16,requires_grad=True);target=torch.randn_like(prediction)
    mask=(torch.rand_like(prediction)>.2).float();sdf=torch.randn_like(prediction)
    loss,_=shared_field_loss(prediction,target,mask,sdf)
    expected=masked_mse(prediction,target,mask)+.2*gradient_mse(prediction,target,mask)+.25*masked_mse(prediction,target,torch.exp(-8*sdf.abs())*mask)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    g1=torch.autograd.grad(loss,prediction,retain_graph=True)[0]
    g2=torch.autograd.grad(expected,prediction)[0]
    torch.testing.assert_close(g1,g2,rtol=2e-6,atol=1e-8)


def test_safe_moments_preserve_features_and_have_finite_local_gradients():
    from types import MethodType
    from experiment import differentiable_encoder_forward
    from codiffuse_Schwarz_Darcy.field_token_composer import DarcyFieldTokenEncoder
    torch.manual_seed(10)
    encoder=DarcyFieldTokenEncoder(1,3,4,8).requires_grad_(False)
    local=torch.randn(4,1,6,6,requires_grad=True)
    physical=torch.randn(4,3,6,6);scalar=torch.randn(4,4);valid=torch.ones(4,1,6,6)
    valid[0]=0;valid[1,:,:,0]=0
    expected=encoder(local,physical,scalar,valid,(2,2))
    encoder.forward=MethodType(differentiable_encoder_forward,encoder)
    actual=encoder(local,physical,scalar,valid,(2,2))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(local.grad).all() and local.grad.abs().sum()>0
    assert local.grad[0].abs().sum()==0
