# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Single-process image-to-image client."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path


def _normalize_log_path(value: str | None) -> Path:
    if not value:
        value = f"~/.openvino/log/img2img-client-py-{time.strftime('%Y%m%d-%H%M%S')}.log"
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _format_timing(timing: dict) -> str:
    parts = [f"{name}: {float(timing[key]):.3f}秒" for key, name in (
        ("load_s", "加载"), ("infer_s", "推理"), ("save_s", "保存")
    ) if key in timing]
    return f"耗时: {float(timing.get('total_s', timing.get('total', 0.0))):.3f} 秒" + (
        f" ({', '.join(parts)})" if parts else ""
    )


def _print_request_reply(reply: dict) -> int:
    if not reply.get("ok") or not reply.get("success"):
        print(f"❌ 图片生成失败:\n{reply.get('error', reply.get('error_code', '<unknown error>'))}", file=sys.stderr)
        return 1
    print(f"✅ 图片已修改: {reply.get('image_path')}")
    print(f"  原图:   {reply.get('reference_image_path')}")
    print(f"  提示词: {reply.get('prompt')}")
    print(f"  种子:   {reply.get('seed')}")
    print(f"  参数:   {reply.get('width')}x{reply.get('height')}, steps={reply.get('steps')}")
    print(f"  设备:   {reply.get('device')}")
    print(_format_timing(reply.get("timing", {})))
    return 0


def run_request(image_path: str, prompt: str, log_path: Path | None = None) -> int:
    runtime = None
    try:
        import runtime as runtime_module

        runtime = runtime_module.Runtime(log_path=log_path)
        runtime.initialize()
        if runtime.state != runtime_module.STATE_RUNNING:
            print(runtime.init_error or f"runtime initialization failed: {runtime.state}", file=sys.stderr)
            return 1
        return _print_request_reply(runtime.dispatch({
            "op": "generate", "reference_image_path": image_path, "prompt": prompt,
        }))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"❌ 图片生成失败:\n{exc}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.pipeline = None
        gc.collect()


def _cmd_request(reference_image_path: str, prompt: str, log_path: Path | None = None) -> int:
    return run_request(reference_image_path, prompt, log_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="local-img2img single-process client")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("-i", "--input", required=True, help="edit instruction")
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)
    return run_request(args.image_path, args.input, _normalize_log_path(args.log))


if __name__ == "__main__":
    raise SystemExit(main())
