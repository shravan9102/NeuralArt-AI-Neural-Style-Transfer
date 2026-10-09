
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Automatically detect project directory on Windows
    BASE_DIR = Path(__file__).resolve().parent

    parser.add_argument(
        '--content_dir',
        type=str,
        default=str(BASE_DIR / 'content_data'),
        help='Location of content dataset'
    )

    parser.add_argument(
        '--style_dir',
        type=str,
        default=str(BASE_DIR / 'style_data'),
        help='Location of style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default=str(BASE_DIR / 'vgg_normalised.pth'),
        help='Location of pre-trained VGG'
    )

    parser.add_argument('--experiment', type=str, default='experiment1')
    parser.add_argument('--final_size', type=int, default=256)
    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)

    parser.add_argument(
        '--crop',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Crop image'
    )

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--style_weight', type=float, default=5.0)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=1)

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--decoder_path', type=str, default=None)
    parser.add_argument('--optimizer_path', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # CPU or CUDA
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device: {device}')

    # Check dataset and pretrained weights paths
    for path, description in [
        (args.content_dir, 'Content dataset'),
        (args.style_dir, 'Style dataset'),
        (args.vgg, 'VGG weights')
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f'{description} not found: {path}'
            )

    # Experiment output folder
    save_dir = (
        Path(__file__).resolve().parent
        / 'experiment'
        / args.experiment
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'args.txt', 'w', encoding='utf-8') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

    # Transforms
    content_transform = get_transform(
        args.content_size, args.crop, args.final_size
    )
    style_transform = get_transform(
        args.style_size, args.crop, args.final_size
    )

    # Datasets
    content_dataset = ImageFolderDataset(
        args.content_dir, content_transform
    )
    style_dataset = ImageFolderDataset(
        args.style_dir, style_transform
    )

    if len(content_dataset) == 0:
        raise ValueError(
            f'No images found in {args.content_dir}'
        )

    if len(style_dataset) == 0:
        raise ValueError(
            f'No images found in {args.style_dir}'
        )

    # CPU-friendly dataloaders
    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        num_workers=0
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        num_workers=0
    )

    print('Content images:', len(content_dataset))
    print('Style images:', len(style_dataset))
    print('Content batches:', len(content_dataloader))
    print('Style batches:', len(style_dataloader))

    # Models
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (
            1.0 + args.lr_decay * epoch
        )
    )

    # Resume training
    if args.resume:
        if not args.decoder_path or not args.optimizer_path:
            raise ValueError(
                'Provide --decoder_path and --optimizer_path '
                'when using --resume'
            )

        decoder.load_state_dict(
            torch.load(
                args.decoder_path,
                map_location=device
            )
        )

        optimizer.load_state_dict(
            torch.load(
                args.optimizer_path,
                map_location=device
            )
        )

    # Freeze VGG weights but allow gradients through its operations
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    mse_loss = torch.nn.MSELoss()

    print('Training started...')

    for epoch in range(args.epochs):
        decoder.train()

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        total_batches = 0

        # Process as many batches as the larger dataset contains
        steps = max(
            len(content_dataloader),
            len(style_dataloader)
        )

        content_iter = iter(content_dataloader)
        style_iter = iter(style_dataloader)

        progress_bar = tqdm(
            range(steps),
            desc=f'Epoch {epoch + 1}/{args.epochs}'
        )

        last_content = None
        last_style = None
        last_output = None

        for _ in progress_bar:
            try:
                content_batch = next(content_iter)
            except StopIteration:
                content_iter = iter(content_dataloader)
                content_batch = next(content_iter)

            try:
                style_batch = next(style_iter)
            except StopIteration:
                style_iter = iter(style_dataloader)
                style_batch = next(style_iter)

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Extract fixed content and style features
            with torch.no_grad():
                c_feats = encoder(content_batch)
                s_feats = encoder(style_batch)

            # Adaptive Instance Normalization
            target = adaptive_instance_normalization(
                c_feats[-1], s_feats[-1]
            )

            # Generate stylized image
            generated = decoder(target)

            # Do NOT use no_grad here:
            # gradients must flow back to the decoder
            generated_feats = encoder(generated)

            # Content loss
            loss_c = (
                mse_loss(generated_feats[-1], target)
                * args.content_weight
            )

            # Style loss
            loss_s = 0.0

            for generated_feature, style_feature in zip(
                generated_feats, s_feats
            ):
                generated_mean, generated_std = calc_mean_std(
                    generated_feature
                )
                style_mean, style_std = calc_mean_std(
                    style_feature
                )

                loss_s = loss_s + (
                    mse_loss(generated_mean, style_mean)
                    + mse_loss(generated_std, style_std)
                )

            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            total_batches += 1

            last_content = content_batch.detach()
            last_style = style_batch.detach()
            last_output = generated.detach()

            progress_bar.set_postfix(
                loss=f'{loss.item():.4f}',
                content=f'{loss_c.item():.4f}',
                style=f'{loss_s.item():.4f}'
            )

        scheduler.step()

        running_loss /= total_batches
        running_closs /= total_batches
        running_sloss /= total_batches

        if (epoch + 1) % args.log_interval == 0:
            print(
                f'Epoch {epoch + 1}: '
                f'Loss: {running_loss:.4f}, '
                f'Content Loss: {running_closs:.4f}, '
                f'Style Loss: {running_sloss:.4f}'
            )

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                decoder.state_dict(),
                save_dir / f'decoder_{epoch + 1}.pth'
            )

            torch.save(
                optimizer.state_dict(),
                save_dir / f'optimizer_{epoch + 1}.pth'
            )

            # Save content, style and generated images
            with torch.no_grad():
                output_images = torch.cat(
                    [last_content, last_style, last_output],
                    dim=0
                )

                save_image(
                    output_images.cpu(),
                    save_dir / f'output_{epoch + 1}.png',
                    nrow=args.batch_size
                )

            print(f'Model and image saved to: {save_dir}')

    print('Training completed!')


if __name__ == '__main__':
    main()
