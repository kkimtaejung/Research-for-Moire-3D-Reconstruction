import os
import torch
from torchvision.utils import save_image


def save_compared_images(opt, generator, encoder, dataloader, device):
    generator.load_state_dict(torch.load("results/generator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    encoder.to(device).eval()

    os.makedirs("results/images_diff", exist_ok=True)

    for i, (img, label) in enumerate(dataloader):
        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        
        def gt_normalize(gt,phase): 	
		max_value = np.max(gt)

		# 최댓값의 위치 찾기
		max_position = np.unravel_index(gt.argmax(), gt.shape)
		
		max_phase = phase[max_position[0],max_position[1]]
		print(max_value, max_phase)
		rate = max_value/max_phase
		gt = gt.astype('float32')
		gt /= rate
	
	return gt
	fake_img = gt_normalize(fake_img,real_img)

        compared_images = torch.empty(real_img.shape[0] * 3,
                                      *real_img.shape[1:])
        compared_images[0::3] = real_img
        compared_images[1::3] = fake_img
        compared_images[2::3] = real_img - fake_img

        save_image(compared_images.data,
                   f"results/images_diff/{opt.n_grid_lines*(i+1):06}.png",
                   nrow=3, normalize=True)

        if opt.n_iters is not None and opt.n_iters == i:
            break
