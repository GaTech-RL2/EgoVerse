"""PEAR's existing portable MoE path with routers frozen before FSDP/optimizer."""
from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels
from efe_pear_router import assert_expert_routers_frozen,freeze_expert_routers


class FrozenRouterTrainingKernels(Qwen3_5TrainingKernels):
    MODEL_TYPES=('qwen3_5_moe',)

    @classmethod
    def install(cls,module,fp32_residual,use_fused_kernels,fused_kernels_backend):
        super().install(module,fp32_residual,use_fused_kernels,fused_kernels_backend)
        cls.router_freeze_receipt=freeze_expert_routers(module)

    @classmethod
    def check_model(cls,module,use_remove_padding,use_fused_kernels,fused_kernels_backend):
        super().check_model(module,use_remove_padding,use_fused_kernels,fused_kernels_backend)
        assert_expert_routers_frozen(module)
