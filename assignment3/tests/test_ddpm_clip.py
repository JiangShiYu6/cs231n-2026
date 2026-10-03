"""Small CPU regression checks; no datasets or pretrained weights required."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cs231n.gaussian_diffusion import GaussianDiffusion
from cs231n.unet import Unet
from cs231n.clip_dino import (
    CLIPImageRetriever, DINOSegmentation,
    clip_zero_shot_classifier, get_similarity_no_loop,
)


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(231)
        torch.set_num_threads(2)

    def test_forward_inverse(self):
        diffusion = GaussianDiffusion(None, image_size=4, timesteps=100).double()
        x = torch.randn(3, 3, 4, 4, dtype=torch.float64)
        noise = torch.randn_like(x)
        t = torch.tensor([0, 30, 99])
        noisy = diffusion.q_sample(x, t, noise)
        torch.testing.assert_close(diffusion.predict_start_from_noise(noisy, t, noise), x)
        torch.testing.assert_close(diffusion.predict_noise_from_start(noisy, t, x), noise)

    def test_last_step_and_both_objectives(self):
        class FixedOutput(nn.Module):
            def forward(self, x, t, model_kwargs):
                return torch.full_like(x, 2.)
        for objective in ('pred_noise', 'pred_x_start'):
            diffusion = GaussianDiffusion(FixedOutput(), image_size=4, timesteps=100, objective=objective)
            x = torch.randn(2, 3, 4, 4)
            result = diffusion.p_sample(x, 0)
            torch.testing.assert_close(result, diffusion.p_sample(x, 0))
            self.assertTrue(torch.isfinite(result).all())
            self.assertLessEqual(float(result.abs().max()), 1.)

    def test_guidance_does_not_mutate_inputs(self):
        model = Unet(4, 4, dim_mults=(2,)).eval()
        x, t = torch.randn(2, 3, 4, 4), torch.tensor([1, 7])
        text = torch.randn(2, 4)
        kwargs = {'text_emb': text, 'cfg_scale': 2.31}
        with torch.no_grad():
            cond = model(x, t, {'text_emb': text})
            uncond = model(x, t, {'text_emb': None})
            guided = model(x, t, kwargs)
            torch.testing.assert_close(guided, 3.31 * cond - 2.31 * uncond)
            torch.testing.assert_close(model(x, t, kwargs), guided)
            self.assertEqual(kwargs['cfg_scale'], 2.31)
            self.assertIs(kwargs['text_emb'], text)
            torch.testing.assert_close(model(x, t, {'text_emb': text, 'cfg_scale': 0}), cond)

    def test_loss_backward(self):
        for objective in ('pred_noise', 'pred_x_start'):
            model = Unet(4, 4, dim_mults=(2,))
            diffusion = GaussianDiffusion(model, image_size=4, timesteps=100, objective=objective)
            loss = diffusion.p_losses(torch.rand(2, 3, 4, 4), {'text_emb': torch.randn(2, 4)})
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


class ClipTests(unittest.TestCase):
    def test_segmentation_uses_training_labels_and_returns_indices(self):
        torch.manual_seed(231)
        segmentation = DINOSegmentation('cpu', 2, inp_dim=2)
        features = torch.tensor([[2., 0.], [3., 0.], [0., 2.], [0., 3.]])
        labels = torch.tensor([0, 0, 1, 1])
        segmentation.train(features, labels, num_iters=100)
        torch.testing.assert_close(segmentation.inference(features), labels)

    def test_similarity_rectangular(self):
        torch.manual_seed(231)
        text, images = torch.randn(5, 8), torch.randn(3, 8)
        expected = torch.nn.functional.cosine_similarity(text[:, None], images[None], dim=-1)
        torch.testing.assert_close(get_similarity_no_loop(text, images), expected)

    def test_classification_and_cached_retrieval(self):
        class Encoder:
            image_calls = 0
            def encode_text(self, tokens):
                return torch.eye(2)[tokens[:, 0]]
            def encode_image(self, images):
                self.image_calls += 1
                return torch.eye(2)
        encoder = Encoder()
        images = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
        preprocess = lambda image: torch.zeros(3, 2, 2)
        tokenize = lambda texts: torch.tensor([[int(text)] for text in texts])
        with patch('cs231n.clip_dino.clip.tokenize', side_effect=tokenize):
            self.assertEqual(clip_zero_shot_classifier(encoder, preprocess, images, ['0', '1'], 'cpu'), ['0', '1'])
            retriever = CLIPImageRetriever(encoder, preprocess, images, 'cpu')
            calls = encoder.image_calls
            self.assertEqual(retriever.retrieve('1', k=1), [1])
            self.assertEqual(retriever.retrieve('0', k=1), [0])
            self.assertEqual(encoder.image_calls, calls)


if __name__ == '__main__':
    unittest.main()
