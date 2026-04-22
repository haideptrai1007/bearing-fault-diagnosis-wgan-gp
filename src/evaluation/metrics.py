import numpy as np
from scipy import linalg

def compute_fid(feats_real, feats_fake):
    """Fréchet Inception Distance between two feature arrays (N,d)."""
    mu_r, mu_f = feats_real.mean(0), feats_fake.mean(0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_f = np.cov(feats_fake, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    # numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def compute_kid(feats_real, feats_fake, num_subsets=100, subset_size=50):
    """
    Kernel Inception Distance (unbiased, polynomial kernel).
    Uses subsampling to keep memory low.
    """
    n = min(len(feats_real), len(feats_fake))
    subset_size = min(subset_size, n)
    kid_values = []

    for _ in range(num_subsets):
        idx_r = np.random.choice(len(feats_real), subset_size, replace=False)
        idx_f = np.random.choice(len(feats_fake), subset_size, replace=False)
        fr = feats_real[idx_r]
        ff = feats_fake[idx_f]

        # polynomial kernel k(x,y) = (x·y / d + 1)^3
        d = fr.shape[1]
        krr = (fr @ fr.T / d + 1) ** 3
        kff = (ff @ ff.T / d + 1) ** 3
        krf = (fr @ ff.T / d + 1) ** 3

        m = subset_size
        # unbiased MMD^2 estimator
        kid = (krr.sum() - np.trace(krr)) / (m * (m - 1)) \
            + (kff.sum() - np.trace(kff)) / (m * (m - 1)) \
            - 2 * krf.mean()
        kid_values.append(float(kid))

    return float(np.mean(kid_values)), float(np.std(kid_values))