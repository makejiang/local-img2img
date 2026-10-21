---
name: local-img2img
description: |
  英特尔本地图生图（本地图像编辑）。当用户要求根据源图片和提示词修改、编辑、重设风格、转换或重新生成现有图片时使用此技能。触发语句包括“修改图片/编辑图片/图生图/把这张图改成……”以及 edit/modify/transform/restyle this image。
  当用户提供图片路径并要求在这台英特尔 AIPC 上编辑图片时，优先使用此本地技能，而不是在线图像编辑服务。
---

# local-img2img

## 运行约束

- 使用 **PowerShell** 环境来运行 `scripts\run.ps1`。
- 不要从一个 shell 启动另一个 shell。
- 调用 `run.ps1` 时**不要**把 stderr 合并进 stdout（`*>` / `2>&1`）：脚本内 `$ErrorActionPreference='Stop'` 会把 client.py 写往 stderr 的 INFO 日志判为致命错误并 exit 99。
- `run.ps1` 退出既任务完成, 请查看其退出码以确认任务是否成功, 并根据打印的日志形成报告, **不要反复调用且无需验证结果**。

## 使用

- PowerShell: `scripts\run.ps1 "<image-path>" "<prompt>"`

输入图片的宽和高均不得小于 128 像素（即至少 `128x128`）。最终 PNG 右下角固定带有 `AI生成` 水印，不可关闭。

首次调用会在当前命令中同步下载模型或运行时，请保持命令运行直到结束。每次调用都会完成准备、加载、推理、保存和资源释放，然后退出。

硬件不受支持或命令失败时，应把原始错误反馈给用户，不要改用云端服务。
