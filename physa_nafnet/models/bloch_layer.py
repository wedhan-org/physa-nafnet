import torch
import torch.nn as nn


class DifferentiableBlochLayer(nn.Module):
    def __init__(
            self,
            tr_t1=646.0,
            te_t1=9.3,
            tr_t2=3204.0,
            te_t2=93.0,
            tr_stir=4000.0,
            te_stir=45.0,
            ti_stir=160.0,
            learnable=True
    ):
        super().__init__()
        # 初始化为临床实际扫描参数 (遵循 Table 2)
        self.log_TR_t1 = nn.Parameter(torch.log(torch.tensor(tr_t1)), requires_grad=learnable)
        self.log_TE_t1 = nn.Parameter(torch.log(torch.tensor(te_t1)), requires_grad=learnable)
        self.log_TR_t2 = nn.Parameter(torch.log(torch.tensor(tr_t2)), requires_grad=learnable)
        self.log_TE_t2 = nn.Parameter(torch.log(torch.tensor(te_t2)), requires_grad=learnable)
        self.log_TR_stir = nn.Parameter(torch.log(torch.tensor(tr_stir)), requires_grad=learnable)
        self.log_TE_stir = nn.Parameter(torch.log(torch.tensor(te_stir)), requires_grad=learnable)
        self.log_TI_stir = nn.Parameter(torch.log(torch.tensor(ti_stir)), requires_grad=learnable)

    def forward(self, pd, t1, t2):
        TR1 = torch.exp(self.log_TR_t1)
        TE1 = torch.exp(self.log_TE_t1)
        TR2 = torch.exp(self.log_TR_t2)
        TE2 = torch.exp(self.log_TE_t2)
        TRs = torch.exp(self.log_TR_stir)
        TEs = torch.exp(self.log_TE_stir)
        TIs = torch.exp(self.log_TI_stir)

        # 将网络输出 [0,1] 映射到物理弛豫时间区间
        t1_v = t1 * 3000.0 + 10.0
        t2_v = t2 * 500.0 + 10.0

        rec_t1 = pd * (1 - torch.exp(-TR1 / t1_v)) * torch.exp(-TE1 / t2_v)
        rec_t2 = pd * (1 - torch.exp(-TR2 / t1_v)) * torch.exp(-TE2 / t2_v)

        mz = 1 - 2 * torch.exp(-TIs / t1_v) + torch.exp(-TRs / t1_v)
        rec_stir = pd * torch.abs(mz) * torch.exp(-TEs / t2_v)

        return rec_t1, rec_t2, rec_stir