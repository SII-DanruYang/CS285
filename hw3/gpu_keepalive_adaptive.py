import argparse
import subprocess
import time
import torch


def get_gpu_util(gpu_id: int = 0) -> int:
    """Return GPU utilization percentage from nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--burst-ms", type=int, default=150)
    parser.add_argument("--sleep-ms", type=int, default=350)
    parser.add_argument("--max-temp", type=int, default=82)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # fp16 矩阵，显存占用不大，但能制造计算负载
    n = args.matrix_size
    a = torch.randn((n, n), device=device, dtype=torch.float16)
    b = torch.randn((n, n), device=device, dtype=torch.float16)
    c = torch.empty((n, n), device=device, dtype=torch.float16)

    print(
        f"[keepalive] gpu={args.gpu}, target={args.target}%, "
        f"matrix={n}x{n}, burst={args.burst_ms}ms, sleep={args.sleep_ms}ms",
        flush=True,
    )

    while True:
        util = get_gpu_util(args.gpu)

        # 利用率已经够高，就让路
        if util >= args.target:
            print(f"[sleep] util={util}% >= target={args.target}%", flush=True)
            time.sleep(args.sleep_ms / 1000)
            continue

        # 利用率低，打一小段计算 burst
        start = time.time()
        iters = 0
        while (time.time() - start) * 1000 < args.burst_ms:
            torch.matmul(a, b, out=c)
            # 加一点点链式操作，避免被过度优化
            c.mul_(1.0001).add_(0.0001)
            iters += 1

        torch.cuda.synchronize()
        util_after = get_gpu_util(args.gpu)

        print(
            f"[work] util_before={util}% util_after={util_after}% "
            f"iters={iters}",
            flush=True,
        )

        time.sleep(args.sleep_ms / 1000)


if __name__ == "__main__":
    main()
