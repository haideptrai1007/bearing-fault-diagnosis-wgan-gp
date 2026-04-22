import torch
import torch.nn as nn
from src.models.architecture.wgan_gp import FeatureExtractor
from torchvision.utils import make_grid
from src.evaluation.metrics import compute_fid, compute_kid
import numpy as np
import gc
import csv
import imageio

class Trainer():
    def __init__(self, generator, discriminator, g_optim, d_optim,
                 gp_w=10, critic_iters=5, logs=50, cuda=False):
        self.G = generator
        self.D = discriminator
        self.g_opt = g_optim
        self.d_opt = d_optim
        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
        self.metrics = {"epoch": [], "FID": [], "KID_mean": [], "KID_std": []}
        self.num_steps = 0
        self.cuda = cuda
        self.gp_w = gp_w
        self.critic_iters = critic_iters
        self.logs = logs
        self.best_fid = float("inf")

        if self.cuda:
            self.G.cuda()
            self.D.cuda()
            if torch.cuda.device_count() > 1:
                print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
                self.G = nn.DataParallel(self.G)
                self.D = nn.DataParallel(self.D)

    def _g_module(self):
        return self.G.module if isinstance(self.G, nn.DataParallel) else self.G

    def _d_module(self):
        return self.D.module if isinstance(self.D, nn.DataParallel) else self.D

    def __gradient_penalty(self, real, generated):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real)
        if self.cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real.data + (1 - alpha) * generated.data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        if self.cuda:
            interpolated = interpolated.cuda()
        prob_inter = self.D(interpolated)
        gradients = torch.autograd.grad(
            outputs=prob_inter, inputs=interpolated,
            grad_outputs=torch.ones(prob_inter.size()).cuda() if self.cuda
            else torch.ones(prob_inter.size()),
            create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().item())
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        return self.gp_w * ((gradients_norm - 1) ** 2).mean()

    def __critic_train_iter(self, data):
        batch_size = data.size(0)
        device = "cuda" if self.cuda else "cpu"
        generated_data = self.G(self._g_module().sample_latent(batch_size, device))
        data = torch.autograd.Variable(data)
        if self.cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_fake = self.D(generated_data)
        gradient_penalty = self.__gradient_penalty(data, generated_data)
        self.losses["GP"].append(gradient_penalty.item())
        self.d_opt.zero_grad()
        d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.d_opt.step()
        self.losses["D"].append(d_loss.item())

    def __generator_train_iter(self, data):
        batch_size = data.size(0)
        device = "cuda" if self.cuda else "cpu"
        generated_data = self.G(self._g_module().sample_latent(batch_size, device))
        self.g_opt.zero_grad()
        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.g_opt.step()
        self.losses["G"].append(g_loss.item())

    def __train_ep(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self.__critic_train_iter(data)
            if self.num_steps % self.critic_iters == 0:
                self.__generator_train_iter(data)
            if i % self.logs == 0:
                print("Iteration {}".format(i+1))
                print("D: {}".format(self.losses["D"][-1]))
                print("GP: {}".format(self.losses["GP"][-1]))
                print("Gradient Norm: {}".format(self.losses["gradient_norm"][-1]))
                if self.num_steps > self.critic_iters:
                    print("G: {}".format(self.losses["G"][-1]))

    def train(self, data_loader, eps, save=True, eval_every=50, real_data=None):
        device = "cuda" if self.cuda else "cpu"

        extractor = None
        if eval_every > 0 and real_data is not None:
            extractor = FeatureExtractor(device=device)
            print("Pre-extracting real features for evaluation …")
            self._real_feats = extractor.extract(real_data, batch_size=16)
            extractor.model.cpu()
            torch.cuda.empty_cache()
            print("Done.\n")

        if save:
            fixed_latents = torch.autograd.Variable(
                self._g_module().sample_latent(16)
            )
            if self.cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_imgs = []

        for ep in range(1, eps + 1):
            print("\nEpoch {}".format(ep))
            self.__train_ep(data_loader)

            if save:
                with torch.no_grad():
                    img_grid = make_grid(self.G(fixed_latents).cpu().data)
                    img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                    img_grid = (img_grid * 0.5 + 0.5) * 255
                    img_grid = img_grid.clip(0, 255).astype(np.uint8)
                    training_progress_imgs.append(img_grid)

            if eval_every > 0 and extractor is not None and ep % eval_every == 0:
                print(f"\n{'='*50}")
                print(f"  Evaluating at epoch {ep}")
                print(f"{'='*50}")
                fid, kid_mean, kid_std = self._evaluate_with_cached_real(
                    extractor, device, num_fake=len(real_data)
                )
                self.metrics["epoch"].append(ep)
                self.metrics["FID"].append(fid)
                self.metrics["KID_mean"].append(kid_mean)
                self.metrics["KID_std"].append(kid_std)
                print(f"  FID  : {fid:.4f}")
                print(f"  KID  : {kid_mean:.6f} ± {kid_std:.6f}")
                self.save()
                print(f"{'='*50}\n")

        if self.metrics["epoch"]:
            print("\n" + "=" * 60)
            print("  EVALUATION SUMMARY")
            print("=" * 60)
            print(f"  {'Epoch':>6}  {'FID':>12}  {'KID mean':>14}  {'KID std':>12}")
            print("-" * 60)
            for i in range(len(self.metrics["epoch"])):
                print(f"  {self.metrics['epoch'][i]:>6}  "
                      f"{self.metrics['FID'][i]:>12.4f}  "
                      f"{self.metrics['KID_mean'][i]:>14.6f}  "
                      f"{self.metrics['KID_std'][i]:>12.6f}")
            print("=" * 60)
            print(f"  Best FID: {self.best_fid:.4f}")

        if extractor is not None:
            extractor.cleanup()
            del self._real_feats
            gc.collect()
            torch.cuda.empty_cache()

        if self.metrics["epoch"]:
            csv_path = "./eval_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "FID", "KID_mean", "KID_std"])
                for i in range(len(self.metrics["epoch"])):
                    writer.writerow([
                        self.metrics["epoch"][i],
                        self.metrics["FID"][i],
                        self.metrics["KID_mean"][i],
                        self.metrics["KID_std"][i],
                    ])
            print(f"Metrics saved → {csv_path}")

        if save:
            print(f"Saving GIF with {len(training_progress_imgs)} frames")
            imageio.mimsave("./training_{}_epochs.gif".format(eps),
                            training_progress_imgs, duration=0.05)

    def _evaluate_with_cached_real(self, extractor, device, num_fake, gen_batch=32):
        self.G.eval()
        fake_list = []
        with torch.no_grad():
            for i in range(0, num_fake, gen_batch):
                bs = min(gen_batch, num_fake - i)
                z = self._g_module().sample_latent(bs, device)
                fake_list.append(self.G(z).cpu())
                del z
                torch.cuda.empty_cache()
        fake_images = torch.cat(fake_list, dim=0)
        del fake_list
        gc.collect()
        torch.cuda.empty_cache()

        feats_fake = extractor.extract(fake_images, batch_size=16)
        del fake_images
        gc.collect()
        torch.cuda.empty_cache()

        fid = compute_fid(self._real_feats, feats_fake)
        kid_mean, kid_std = compute_kid(self._real_feats, feats_fake)

        del feats_fake
        gc.collect()
        self.G.train()
        return fid, kid_mean, kid_std

    def sample_generator(self, num_samples):
        latent_samples = torch.autograd.Variable(
            self._g_module().sample_latent(num_samples)
        )
        if self.cuda:
            latent_samples = latent_samples.cuda()
        return self.G(latent_samples)

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        return generated_data.data.cpu().numpy()[:, 0, :, :]

    # ── Save unwrapped state dicts ──
    def save(self):
        torch.save(self._g_module().state_dict(), "./gen_model.pt")
        torch.save(self._d_module().state_dict(), "./dis_model.pt")

        if self.metrics["FID"] and self.metrics["FID"][-1] < self.best_fid:
            self.best_fid = self.metrics["FID"][-1]
            checkpoint = {
                "epoch": self.metrics["epoch"][-1],
                "fid": self.best_fid,
                "generator": self._g_module().state_dict(),
                "discriminator": self._d_module().state_dict(),
                "g_optimizer": self.g_opt.state_dict(),
                "d_optimizer": self.d_opt.state_dict(),
            }
            torch.save(checkpoint, "./best_model.pt")
            print(f"New best model saved (FID: {self.best_fid:.4f})")

    # ── Load best model checkpoint ──
    def load_best(self):
        checkpoint = torch.load("./best_model.pt", weights_only=False)
        self._g_module().load_state_dict(checkpoint["generator"])
        self._d_module().load_state_dict(checkpoint["discriminator"])
        self.g_opt.load_state_dict(checkpoint["g_optimizer"])
        self.d_opt.load_state_dict(checkpoint["d_optimizer"])
        self.best_fid = checkpoint["fid"]
        print(f"Loaded best model from epoch {checkpoint['epoch']} "
              f"(FID: {checkpoint['fid']:.4f})")
        return checkpoint["epoch"]