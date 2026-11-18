"""本地复现页面上的参考数值。

用法:  python tools/reference_run.py
不依赖任何框架，只用 NumPy 手写 2→2→1 全批量 BCEWithLogits + Adam，
跑 float64 与 float32 两次，打印最终参数、预测概率与两者的最大差。
"""
import numpy as np

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([0., 1., 1., 0.]).reshape(4, 1)
SEED, EPOCHS, LR, EPS = 42, 260, 0.06, 1e-8
B1, B2 = 0.9, 0.999


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init(seed, dtype):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, (2, 2)).astype(dtype),
            np.zeros(2, dtype),
            rng.uniform(-1, 1, (2, 1)).astype(dtype),
            np.zeros(1, dtype)]


def train(seed=SEED, dtype=np.float64):
    W, b, V, c = init(seed, dtype)
    params = [W, b, V, c]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    x, y = X.astype(dtype), Y.astype(dtype)
    hist = []

    for t in range(1, EPOCHS + 1):
        W, b, V, c = params
        H = sigmoid(x @ W + b)
        logit = H @ V + c
        pr = sigmoid(logit)
        loss = float(np.mean(np.log1p(np.exp(-np.abs(logit)))
                             + np.maximum(logit, 0) - y * logit))
        hist.append((t, loss, int(np.sum((pr > 0.5) == (y > 0.5)))))

        dlogit = (pr - y) / dtype(4.0)
        dh = (dlogit @ V.T) * H * (1 - H)
        grads = [x.T @ dh, dh.sum(axis=0), H.T @ dlogit, dlogit.sum(axis=0)]
        for i, g in enumerate(grads):
            m[i] = B1 * m[i] + (1 - B1) * g
            v[i] = B2 * v[i] + (1 - B2) * g * g
            mh = m[i] / (1 - B1 ** t)
            vh = v[i] / (1 - B2 ** t)
            params[i] = params[i] - (LR * mh / (np.sqrt(vh) + dtype(EPS))).astype(dtype)

    W, b, V, c = params
    H = sigmoid(x @ W + b)
    pred = sigmoid(H @ V + c).ravel()
    return {
        "W": np.round(W.ravel(), 6), "b": np.round(b, 6),
        "V": np.round(V.ravel(), 6), "c": np.round(c, 6),
        "pred": np.round(pred, 4), "loss": round(hist[-1][1], 6),
        "acc": hist[-1][2], "hist": hist,
    }


def main():
    iw = init(SEED, np.float64)
    print("初始化 (seed=42, uniform(-1,1))")
    print("  W =", np.round(iw[0], 6).tolist(), " b =", iw[1].tolist())
    print("  v =", np.round(iw[2].ravel(), 6).tolist(), " c =", float(iw[3][0]))
    print()

    r64 = train(dtype=np.float64)
    r32 = train(dtype=np.float32)

    names = ["W[0][0]", "W[0][1]", "W[1][0]", "W[1][1]", "b[0]", "b[1]", "v[0]", "v[1]", "c"]
    flat64 = np.concatenate([r64["W"], r64["b"], r64["V"], r64["c"]])
    flat32 = np.concatenate([r32["W"], r32["b"], r32["V"], r32["c"]])
    print(f"{'参数':<8}{'float64':>14}{'float32':>14}")
    for n, a, b_ in zip(names, flat64, flat32):
        print(f"{n:<8}{a:>14.6f}{b_:>14.6f}")
    print(f"\n最大参数差 = {np.max(np.abs(flat64 - flat32)):.2e}")
    print("float64 预测 =", r64["pred"], " loss =", r64["loss"], " 命中 =", r64["acc"], "/4")
    print("float32 预测 =", r32["pred"], " loss =", r32["loss"], " 命中 =", r32["acc"], "/4")

    r11 = train(seed=11, dtype=np.float64)
    print("\n对照 · seed=11 的塌缩解：loss =", r11["loss"], " 命中 =", r11["acc"],
          "/4 预测 =", r11["pred"])


if __name__ == "__main__":
    main()
