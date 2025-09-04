import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DAPPM(nn.Module):
    def __init__(self, branch_planes: int, outplanes: int):
        super(DAPPM, self).__init__()

        self.branch_planes = branch_planes
        self.outplanes = outplanes
        self._initialised: bool = False

    def _make_layers(self, inplanes: int, device: torch.device) -> None:
        self.scale1 = (
            nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=False),
                nn.Conv2d(inplanes, self.branch_planes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.scale2 = (
            nn.Sequential(
                nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=False),
                nn.Conv2d(inplanes, self.branch_planes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.scale3 = (
            nn.Sequential(
                nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=False),
                nn.Conv2d(inplanes, self.branch_planes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.scale4 = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=False),
                nn.Conv2d(inplanes, self.branch_planes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.scale0 = (
            nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=False),
                nn.Conv2d(inplanes, self.branch_planes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.process1 = (
            nn.Sequential(
                nn.BatchNorm2d(self.branch_planes),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.branch_planes, self.branch_planes, kernel_size=3, padding=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.process2 = (
            nn.Sequential(
                nn.BatchNorm2d(self.branch_planes),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.branch_planes, self.branch_planes, kernel_size=3, padding=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.process3 = (
            nn.Sequential(
                nn.BatchNorm2d(self.branch_planes),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.branch_planes, self.branch_planes, kernel_size=3, padding=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.process4 = (
            nn.Sequential(
                nn.BatchNorm2d(self.branch_planes),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.branch_planes, self.branch_planes, kernel_size=3, padding=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.compression = (
            nn.Sequential(
                nn.BatchNorm2d(self.branch_planes * 5),
                nn.ReLU(inplace=False),
                nn.Conv2d(self.branch_planes * 5, self.outplanes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

        self.shortcut = (
            nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=False),
                nn.Conv2d(inplanes, self.outplanes, kernel_size=1, bias=False),
            )
            .to(device=device)
            .eval()
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self._initialised:
            self._initialised = True

            self._make_layers(x.shape[1], x.device)

        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x), size=[height, width], mode="bilinear") + x_list[0])))
        x_list.append(
            (self.process2((F.interpolate(self.scale2(x), size=[height, width], mode="bilinear") + x_list[1])))
        )
        x_list.append(self.process3((F.interpolate(self.scale3(x), size=[height, width], mode="bilinear") + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x), size=[height, width], mode="bilinear") + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)

        return out  # type: ignore
