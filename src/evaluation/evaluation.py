import torch
import gc
from src.evaluation.metrics import compute_fid, compute_kid

def evaluate_generator(generator, real_images, extractor, device, gen_batch=32, feat_batch=16):
    """Generate fakes, extract features, compute FID & KID, free VRAM."""
    num_fake = len(real_images)
    generator.eval()

    # --- generate fakes on CPU ---
    fake_list = []
    with torch.no_grad():
        for i in range(0, num_fake, gen_batch):
            bs = min(gen_batch, num_fake - i)
            z = generator.sample_latent(bs, device)
            fake_list.append(generator(z).cpu())
            del z
            torch.cuda.empty_cache()
    fake_images = torch.cat(fake_list, dim=0)
    del fake_list
    gc.collect()
    torch.cuda.empty_cache()

    # --- extract features ---
    feats_real = extractor.extract(real_images, batch_size=feat_batch)
    feats_fake = extractor.extract(fake_images, batch_size=feat_batch)

    del fake_images
    gc.collect()
    torch.cuda.empty_cache()

    # --- compute ---
    fid = compute_fid(feats_real, feats_fake)
    kid_mean, kid_std = compute_kid(feats_real, feats_fake)

    del feats_real, feats_fake
    gc.collect()

    generator.train()
    return fid, kid_mean, kid_std