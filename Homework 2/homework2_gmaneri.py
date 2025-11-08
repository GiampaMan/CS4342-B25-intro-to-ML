import numpy as np
import matplotlib.pyplot as plt  # to show images
from matplotlib.backends.backend_pdf import PdfPages

# Given an array of faces (N x 48 x 48 or N x 2304), return Xtilde ((2304+1) x N) with final row 1s.
def reshapeAndAppend1s(faces):
    arr = np.asarray(faces)
    if arr.ndim == 3:
        N, H, W = arr.shape
        X = arr.reshape(N, H*W)
    elif arr.ndim == 2:
        N, D = arr.shape
        X = arr
    else:
        raise ValueError(f"Unexpected faces shape {arr.shape}; expected (N,48,48) or (N,2304)")
    X = X.astype(np.float64).T
    ones = np.ones((1, X.shape[1]), dtype=X.dtype)
    return np.vstack([X, ones])

def fMSE(wtilde, Xtilde, y):
    yhat = (wtilde @ Xtilde)
    r = yhat - y.reshape(-1)
    n = r.size
    return 0.5 / n * (r @ r)

def gradfMSE(wtilde, Xtilde, y, alpha=0.0):
    y = y.reshape(-1)
    n = y.size
    residual = (wtilde @ Xtilde) - y
    grad = (Xtilde @ residual) / n
    if alpha != 0.0:
        w_no_bias = wtilde.copy()
        w_no_bias[-1] = 0.0  # don't regularize bias
        grad = grad + (alpha / n) * w_no_bias
    return grad

# (a) Analytical solution via normal equations with pinv fallback
def method1(Xtilde, y):
    A = Xtilde @ Xtilde.T
    b = Xtilde @ y.reshape(-1)
    try:
        wtilde = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        wtilde = np.linalg.pinv(A) @ b
    return wtilde

# Helper for (b) and (c): gradient descent on (regularized) half-MSE
def gradientDescent(Xtilde, y, alpha=0.0):
    EPSILON = 3e-3   # learning rate
    T = 5000         # iterations
    rng = np.random.default_rng(42)
    wtilde = rng.normal(0.0, 0.01, size=(Xtilde.shape[0],))
    for _ in range(T):
        g = gradfMSE(wtilde, Xtilde, y, alpha=alpha)
        wtilde = wtilde - EPSILON * g
    return wtilde

# (b) Unregularized GD
def method2(Xtilde, y):
    return gradientDescent(Xtilde, y, alpha=0.0)

# (c) GD with L2 on w (bias unpenalized)
def method3(Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, alpha=ALPHA)

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy").reshape(-1).astype(np.float64)
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy").reshape(-1).astype(np.float64)

    # Train
    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report half-MSE on train/test
    print("Half-MSE (train/test):")
    for name, w in [("Analytical", w1), ("GD (no reg)", w2), ("GD (L2, alpha=0.1)", w3)]:
        tr = fMSE(w, Xtilde_tr, ytr)
        te = fMSE(w, Xtilde_te, yte)
        print(f"{name:>16} | train: {tr:.6f} | test: {te:.6f}")

    # Part (d): create the required PDF using method (c)
    def weight_image(wtilde): return wtilde[:-1].reshape(48,48)
    yhat_te_c = w3 @ Xtilde_te
    abs_err = np.abs(yhat_te_c - yte)
    top5_idx = np.argsort(abs_err)[-5:][::-1]

    with PdfPages("homework2_errors_gmaneri.pdf") as pdf:
        # simple summary page
        fig = plt.figure(figsize=(8.5, 11)); ax = plt.gca(); ax.axis('off')
        ax.text(0.05, 0.98, "Homework 2 — Linear Regression for Age Estimation", va='top', ha='left', fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # three weight images
        for title, w in [("Weights — Analytical", w1), ("Weights — GD (no reg)", w2), ("Weights — GD (L2, alpha=0.1)", w3)]:
            fig = plt.figure(figsize=(8.5, 11)); ax = plt.gca(); ax.set_title(title)
            ax.imshow(weight_image(w)); ax.axis('off'); pdf.savefig(fig); plt.close(fig)

        # worst five test errors for method (c)
        Xte = np.load("age_regression_Xte.npy")
        for rank, i in enumerate(top5_idx, start=1):
            fig = plt.figure(figsize=(8.5, 11)); ax = plt.gca()
            face = Xte[i].reshape(48,48) if Xte[i].ndim == 1 else Xte[i]
            ax.set_title(f"Worst error #{rank}\nGT={yte[i]:.2f}, Pred={yhat_te_c[i]:.2f}, |err|={abs_err[i]:.2f}")
            ax.imshow(face); ax.axis('off'); pdf.savefig(fig); plt.close(fig)
