import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def _compute_stft_for_frontend(frontend, wav):
    """
    wav: [B, M, F, T]
    returns:
        stft: [B, M, Freq, Frames] complex
        mag : [B, M, Freq, Frames] real
    """
    device = next(frontend.parameters()).device
    wav = wav.to(device)

    B, M, n_frames_in, T = wav.shape
    wav_reshape = wav.reshape(B * M, n_frames_in * T)

    stft = torch.stft(
        wav_reshape,
        n_fft=frontend.n_fft,
        hop_length=frontend.hop_length,
        win_length=frontend.win_length,
        window=frontend.window.to(device),
        center=True,
        return_complex=True,
    )
    stft = stft.reshape(B, M, stft.shape[1], stft.shape[2])

    if stft.shape[-1] > n_frames_in:
        stft = stft[:, :, :, :n_frames_in]

    mag = torch.abs(stft)
    return stft, mag


def _get_prior_logs(frontend, mode="current"):
    """
    mode:
        - 'base'    : fixed base prior only
        - 'current' : base + current trainable delta 반영
    """
    shared_prior_log = frontend.shared_prior_log.detach().cpu().numpy()

    if mode == "base":
        mel_prior_log = frontend.mel_prior_base.detach().cpu().numpy()
        gcc_prior_log = frontend.gcc_prior_base.detach().cpu().numpy()
    elif mode == "current":
        mel_prior_log_t, gcc_prior_log_t = frontend.get_current_prior_logs()
        mel_prior_log = mel_prior_log_t.detach().cpu().numpy()
        gcc_prior_log = gcc_prior_log_t.detach().cpu().numpy()
    else:
        raise ValueError("mode must be 'base' or 'current'")

    return shared_prior_log, mel_prior_log, gcc_prior_log


def plot_frontend_prior(frontend, sr=24000, save_path=None, mode="current"):
    """
    prior 모양만 시각화.
    mode='base'    : 초기 고정 prior
    mode='current' : 현재 학습된 delta까지 반영된 prior
    """
    was_training = frontend.training
    frontend.eval()

    with torch.no_grad():
        freqs = np.linspace(0, sr / 2, frontend.n_freqs)
        shared_prior_log, mel_prior_log, gcc_prior_log = _get_prior_logs(frontend, mode=mode)

        shared_prior_weight = np.exp(frontend.shared_scale * shared_prior_log)
        mel_prior_weight = np.exp(
            frontend.shared_scale * shared_prior_log
            + frontend.residual_scale * mel_prior_log
        )
        gcc_prior_weight = np.exp(
            frontend.shared_scale * shared_prior_log
            + frontend.residual_scale * gcc_prior_log
        )

    plt.figure(figsize=(9, 5))
    plt.plot(freqs, shared_prior_weight, '--', label='Shared prior')
    plt.plot(freqs, mel_prior_weight, '--', label=f'Mel prior ({mode})')
    plt.plot(freqs, gcc_prior_weight, '--', label=f'GCC prior ({mode})')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Weight')
    plt.title(f'GRU-style frontend priors ({mode})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    if was_training:
        frontend.train()


def plot_frontend_dynamic_weights(frontend, wav, sr=24000, save_path=None, reduction='mean', prior_mode="current"):
    """
    실제 입력 wav에 대해 나온 dynamic mel/gcc weights를 시각화.

    Args:
        frontend: AuditoryFrontend
        wav: [B, M, F, T]
        reduction:
            - 'mean': batch 평균 곡선 1개씩 그림
            - 'none': batch의 각 샘플 곡선을 전부 그림
        prior_mode:
            - 'base'    : 고정 base prior와 비교
            - 'current' : 현재 학습된 prior와 비교
    """
    was_training = frontend.training
    frontend.eval()

    with torch.no_grad():
        stft, mag = _compute_stft_for_frontend(frontend, wav)
        mel_w, gcc_w = frontend.get_weights(stft, mag)

        mel_w = mel_w.detach().cpu().numpy()
        gcc_w = gcc_w.detach().cpu().numpy()

        shared_prior_log, mel_prior_log, gcc_prior_log = _get_prior_logs(frontend, mode=prior_mode)

        mel_prior_weight = np.exp(
            frontend.shared_scale * shared_prior_log
            + frontend.residual_scale * mel_prior_log
        )
        gcc_prior_weight = np.exp(
            frontend.shared_scale * shared_prior_log
            + frontend.residual_scale * gcc_prior_log
        )

    freqs = np.linspace(0, sr / 2, frontend.n_freqs)

    plt.figure(figsize=(10, 5))

    plt.plot(freqs, mel_prior_weight, '--', label=f'Mel prior ({prior_mode})', linewidth=2)
    plt.plot(freqs, gcc_prior_weight, '--', label=f'GCC prior ({prior_mode})', linewidth=2)

    if reduction == 'mean':
        mel_mean = mel_w.mean(axis=0)
        gcc_mean = gcc_w.mean(axis=0)

        mel_std = mel_w.std(axis=0)
        gcc_std = gcc_w.std(axis=0)

        plt.plot(freqs, mel_mean, label='Mel dynamic (mean)', color='tab:green')
        plt.plot(freqs, gcc_mean, label='GCC dynamic (mean)', color='tab:red')

        plt.fill_between(freqs, mel_mean - mel_std, mel_mean + mel_std, alpha=0.2)
        plt.fill_between(freqs, gcc_mean - gcc_std, gcc_mean + gcc_std, alpha=0.2)

    elif reduction == 'none':
        for i in range(mel_w.shape[0]):
            plt.plot(freqs, mel_w[i], alpha=0.5, label='Mel dynamic' if i == 0 else None)
            plt.plot(freqs, gcc_w[i], alpha=0.5, label='GCC dynamic' if i == 0 else None)
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Weight')
    plt.title(f'GRU-style frequency weights: prior vs dynamic ({prior_mode})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != '':
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    if was_training:
        frontend.train()