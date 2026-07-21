#
# demo.py
#
# End-to-end demo: loads the IBL ephys behavioral data and renders a single
# figure with the key behavioral panels used in the paper, arranged in a
# 2x4 grid (eight axes, fully populated).
#
# Each panel faithfully reproduces a snippet of the existing plotting code, but
# drawn onto shared axes (ax.* calls) instead of separate figures:
#
#   [0,0] RT distribution                      <- RT_analysis.plot_RT_stats       L70-L75
#   [0,1] Fast psychometric (per block)        <- RT_analysis.plot_RT_stats       L135-L142 (subplot 1)
#   [0,2] Normal psychometric (per block)      <- RT_analysis.plot_RT_stats       L143-L148 (subplot 2)
#   [0,3] RT distribution (log, slow thresh.)  <- RT_analysis.plot_RT_stats       L83-L91
#   [1,0] RT hist, high contrast (correct/err) <- RT_analysis.plot_RT_stats       L172-L177 (subplot 1)
#   [1,1] RT hist, low contrast (correct/err)  <- RT_analysis.plot_RT_stats       L179-L183 (subplot 2)
#   [1,2] Fast/slow ratio vs session time      <- RT_analysis.plot_RT_stats       L223-L230
#   [1,3] RT-type conditional probability      <- behav_analysis.fast_slow_freq_stats L1008-L1027
#
# Hyperparameters match behav_analysis.py's __main__ block.
#
import os
import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt

from data_loading import load_data, one
from RT_analysis import calc_contrast_idx, calc_block_idx
from behav_analysis import process_data

# Reproducible shuffled (null) curve in the conditional-probability panel.
# The source functions do not seed; remove this line to match their behavior.
np.random.seed(0)

clr2s = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 9 signed-contrast levels (matches RT_analysis.plot_RT_stats L133)
CONTRASTS = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]


# ---------------------------------------------------------------------------
# Panel data computations (each mirrors the corresponding block in the source)
# ---------------------------------------------------------------------------

def compute_rt_total(data):
    """Concatenated reaction times across all sessions (source L51-L62)."""
    RTs = []
    for subject in data.keys():
        for session_data in data[subject]:
            RTs.append(session_data['first_movement_onset_times'] - session_data['stimOn_times'])
    RTs_total = RTs[0].copy()
    for i in range(1, len(RTs)):
        RTs_total = np.concatenate((RTs_total, RTs[i]))
    return RTs_total


def compute_psychometrics(data, fast_threshold, slow_threshold):
    """P(rightward) vs contrast, per prob-left block, for fast and normal RTs (source L94-L120)."""
    normal_psychometric = np.zeros((2, 9)); normal_psychometric_cnt = np.zeros((2, 9))
    fast_psychometric = np.zeros((2, 9));   fast_psychometric_cnt = np.zeros((2, 9))

    for subject in data.keys():
        for session_data in data[subject]:
            for tridx in range(len(session_data['stimOn_times'])):
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
                block_idx = calc_block_idx(session_data['probLeft'][tridx])
                if block_idx == 0 or block_idx == 1:
                    if rttmp < fast_threshold:
                        fast_psychometric[block_idx, contrast_idx] += (session_data['first_movement_directions'][tridx] + 1) / 2
                        fast_psychometric_cnt[block_idx, contrast_idx] += 1
                    elif rttmp < slow_threshold:
                        normal_psychometric[block_idx, contrast_idx] += (session_data['first_movement_directions'][tridx] + 1) / 2
                        normal_psychometric_cnt[block_idx, contrast_idx] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        fast_psychometric = np.divide(fast_psychometric, fast_psychometric_cnt)
        fp_ste = np.sqrt(np.divide(fast_psychometric * (1.0 - fast_psychometric), fast_psychometric_cnt))
        normal_psychometric = np.divide(normal_psychometric, normal_psychometric_cnt)
        np_ste = np.sqrt(np.divide(normal_psychometric * (1.0 - normal_psychometric), normal_psychometric_cnt))
    return fast_psychometric, fp_ste, normal_psychometric, np_ste


def compute_contrast_rt_hists(data):
    """log-RT samples split by high/low contrast and correct/incorrect (source L153-L170)."""
    RTs_HC = []; RTs_HC_incorrect = []
    RTs_LC = []; RTs_LC_incorrect = []
    for subject in data.keys():
        for session_data in data[subject]:
            for tridx in range(len(session_data['stimOn_times'])):
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
                if (contrast_idx <= 1 or 7 <= contrast_idx) and rttmp > 0.05:
                    RTs_HC.append(np.log(rttmp))
                    if session_data['feedbackType'][tridx] < 0.0:
                        RTs_HC_incorrect.append(np.log(rttmp))
                elif (3 <= contrast_idx <= 5) and rttmp > 0.05:
                    RTs_LC.append(np.log(rttmp))
                    if session_data['feedbackType'][tridx] < 0.0:
                        RTs_LC_incorrect.append(np.log(rttmp))
    return RTs_HC, RTs_HC_incorrect, RTs_LC, RTs_LC_incorrect


def compute_relative_session_ratio(data, fast_threshold, slow_threshold):
    """Fast/slow-RT fraction vs relative position within a session (source L197-L221)."""
    dT = 0.01
    relative_session_times = np.arange(0.0, 1.0, dT)
    rstlen = len(relative_session_times)

    fast_rt_counts = np.zeros(rstlen)
    slow_rt_counts = np.zeros(rstlen)
    total_rt_counts = np.zeros(rstlen)

    for subject in data.keys():
        for session_data in data[subject]:
            session_len = len(session_data['stimOn_times'])
            for tridx in range(session_len):
                ridx = int(np.floor((tridx / session_len) / dT))
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                total_rt_counts[ridx] += 1
                if rttmp < fast_threshold:
                    fast_rt_counts[ridx] += 1
                if rttmp >= slow_threshold:
                    slow_rt_counts[ridx] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        fast_rt_ratio = np.divide(fast_rt_counts, total_rt_counts)
        slow_rt_ratio = np.divide(slow_rt_counts, total_rt_counts)
        fast_rt_err = np.sqrt(np.divide((1.0 - fast_rt_ratio) * fast_rt_ratio, total_rt_counts))
        slow_rt_err = np.sqrt(np.divide((1.0 - slow_rt_ratio) * slow_rt_ratio, total_rt_counts))
    return relative_session_times, fast_rt_ratio, slow_rt_ratio, fast_rt_err, slow_rt_err


def _calc_cond_prob(seq, nlag):
    """Numerators/denominators of P(1 at lag k | 1 at 0) (source: nested calc_cond_prob)."""
    full_corr = np.correlate(seq, seq, mode='full')
    n = len(seq)
    nums = full_corr[n:n + nlag]
    dens = np.array([np.sum(seq[:n - k]) if k < n else 0 for k in range(1, nlag + 1)])
    return nums, dens


def compute_cond_probs(data, fast_threshold, slow_threshold, nlag=50, min_trial_count=0):
    """Session-wise conditional probabilities and shuffled null (source L892-L979, fig1b inputs only)."""
    sl_cond_probs = {'fast': [], 'slow': []}
    sl_shuffled_cond_probs = {'fast': [], 'slow': []}
    cond_probs = {'fast': [], 'slow': []}  # animal-wise; only its length is used by the panel

    for subject in data.keys():
        total_nums = {'fast': np.zeros(nlag), 'slow': np.zeros(nlag)}
        total_dens = {'fast': np.zeros(nlag), 'slow': np.zeros(nlag)}

        for session_data in data[subject]:
            if len(session_data['contrast']) > nlag:
                trial_mask = {}
                trial_mask['fast'] = np.where(session_data['reaction_times'] < fast_threshold, 1, 0)
                trial_mask['slow'] = np.where(session_data['reaction_times'] > slow_threshold, 1, 0)

                for trial_type in ['fast', 'slow']:
                    if np.sum(trial_mask[trial_type]) > min_trial_count:
                        seq = trial_mask[trial_type]

                        nums, dens = _calc_cond_prob(seq, nlag)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            session_prob = np.divide(nums, dens, out=np.full(nlag, np.nan), where=dens > 0)
                            sl_cond_probs[trial_type].append(session_prob)
                        total_nums[trial_type][:len(nums)] += nums
                        total_dens[trial_type][:len(dens)] += dens

                        for _ in range(10):
                            np.random.shuffle(seq)
                            nums, dens = _calc_cond_prob(seq, nlag)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                shuffled_session_prob = np.divide(nums, dens, out=np.full(nlag, np.nan), where=dens > 0)
                                sl_shuffled_cond_probs[trial_type].append(shuffled_session_prob)

        for trial_type in ['fast', 'slow']:
            with np.errstate(divide='ignore', invalid='ignore'):
                animal_prob = np.divide(total_nums[trial_type], total_dens[trial_type],
                                        out=np.full(nlag, np.nan), where=total_dens[trial_type] > 0)
                cond_probs[trial_type].append(animal_prob)

    return sl_cond_probs, sl_shuffled_cond_probs, cond_probs, nlag


# ---------------------------------------------------------------------------
# Panel drawing (ax.* versions of the source plotting snippets)
# ---------------------------------------------------------------------------

def draw_rt_distribution(ax, RTs_total, fast_threshold):
    ax.hist(RTs_total, range=(-0.3, 1.6), bins=144, color=clr2s[0])
    ax.axvline(fast_threshold, ls='--', color='r', lw=2.0)
    ax.set_xlim(-0.3, 1.5)
    ax.set_title('RT distribution')
    ax.set_xlabel('RT [s]')


def draw_rt_log_distribution(ax, RTs_log_total, slow_threshold):
    ax.hist(RTs_log_total, bins=144, color=clr2s[0])
    ax.axvline(np.log(slow_threshold), ls='--', color='r', lw=2.0)
    ax.set_xlim(np.log(0.05), np.log(30.0))
    ax.set_xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])
    ax.set_title('RT distribution (log)')
    ax.set_xlabel('RT [s]')


def draw_psychometric(ax, psychometric, ste, title):
    for bidx in range(2):
        ax.fill_between(CONTRASTS, psychometric[bidx] + ste[bidx], psychometric[bidx] - ste[bidx], color=clr2s[bidx], alpha=0.2)
        ax.plot(CONTRASTS, psychometric[bidx], 'o-', color=clr2s[bidx])
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_title(title)
    ax.set_xlabel('contrast')
    ax.set_ylabel('P(rightward)')


def draw_contrast_rt_hist(ax, RTs, RTs_incorrect, title):
    ax.hist(RTs, range=(np.log(0.05), np.log(30.0)), bins=72, alpha=0.5, color=clr2s[0])
    ax.hist(RTs_incorrect, range=(np.log(0.05), np.log(30.0)), bins=72, alpha=0.5, color='k')
    ax.set_xlim(np.log(0.05), np.log(30.0))
    ax.set_xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])
    ax.set_title(title)
    ax.set_xlabel('RT [s]')


def draw_relative_session_ratio(ax, times, fast_ratio, slow_ratio, fast_err, slow_err):
    ax.fill_between(times, fast_ratio + fast_err, fast_ratio - fast_err, alpha=0.25, color='y')
    ax.plot(times, fast_ratio, color='y')
    ax.fill_between(times, slow_ratio + slow_err, slow_ratio - slow_err, alpha=0.25, color='m')
    ax.plot(times, slow_ratio, color='m')
    ax.set_xlim(-0.01, 1.01)
    ax.set_title('Fast/slow ratio vs session time')
    ax.set_xlabel('relative session time')
    ax.set_ylabel('fraction of trials')


def draw_cond_probs(ax, sl_cond_probs, sl_shuffled_cond_probs, cond_probs, nlag):
    lags = np.arange(1, nlag + 1)
    fs_colors = {'fast': 'y', 'slow': 'm'}
    for trial_type in ['slow', 'fast']:
        cond_probs_tot = np.array(sl_cond_probs[trial_type])
        mean_cond_probs = np.nanmean(cond_probs_tot, axis=0)
        sem_cond_probs = np.nanstd(cond_probs_tot, axis=0) / np.sqrt(len(cond_probs[trial_type]))
        ax.fill_between(lags, mean_cond_probs + sem_cond_probs, mean_cond_probs - sem_cond_probs, color=fs_colors[trial_type], alpha=0.2)
        ax.plot(lags, mean_cond_probs, color=fs_colors[trial_type])

        shuffled_cond_probs_tot = np.array(sl_shuffled_cond_probs[trial_type])
        shuffled_mean_cond_probs = np.nanmean(shuffled_cond_probs_tot, axis=0)
        ax.plot(lags, shuffled_mean_cond_probs, color=fs_colors[trial_type], ls='--')
    ax.set_ylim(0.0, 0.35)
    ax.set_xlim(0.0, nlag)
    ax.set_title('RT-type conditional prob.')
    ax.set_xlabel('trial lag')
    ax.set_ylabel('P(same type)')


# ---------------------------------------------------------------------------

def main():
    # Hyperparameters (identical to behav_analysis.py __main__)
    params = {
        'session_type': 'ephys',   # 'ephys' or 'all_biased'
        'min_trials': 400,         # minimum number of trials (NOT inclusive)
        'fast_threshold': 0.08,    # [s] threshold for fast trials
        'slow_threshold': 1.25,    # [s] threshold for slow trials
        's_cutoff': 40,            # remove last 40 trials to minimize the effect of satiation
        'min_sessions': 2,         # minimum number of sessions for individual-level analysis
    }
    fast_threshold = params['fast_threshold']
    slow_threshold = params['slow_threshold']

    print('Loading behavioral data...')
    data, subject_info = load_data(params['session_type'], one)
    # process_data adds per-session 'reaction_times' (needed by the conditional-prob panel)
    data, subj_data = process_data(data, params)

    print('Computing panels...')
    RTs_total = compute_rt_total(data)
    RTs_log_total = [np.log(rt) for rt in RTs_total if rt > 0.05]  # source L81
    fast_psychometric, fp_ste, normal_psychometric, np_ste = compute_psychometrics(data, fast_threshold, slow_threshold)
    RTs_HC, RTs_HC_incorrect, RTs_LC, RTs_LC_incorrect = compute_contrast_rt_hists(data)
    rel_times, fast_rt_ratio, slow_rt_ratio, fast_rt_err, slow_rt_err = compute_relative_session_ratio(data, fast_threshold, slow_threshold)
    sl_cond_probs, sl_shuffled_cond_probs, cond_probs, nlag = compute_cond_probs(data, fast_threshold, slow_threshold)

    print('Rendering figure...')
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    draw_rt_distribution(axes[0, 0], RTs_total, fast_threshold)
    draw_psychometric(axes[0, 1], fast_psychometric, fp_ste, 'Fast psychometric')
    draw_psychometric(axes[0, 2], normal_psychometric, np_ste, 'Normal psychometric')
    draw_rt_log_distribution(axes[0, 3], RTs_log_total, slow_threshold)
    draw_contrast_rt_hist(axes[1, 0], RTs_HC, RTs_HC_incorrect, 'RT (high contrast)')
    draw_contrast_rt_hist(axes[1, 1], RTs_LC, RTs_LC_incorrect, 'RT (low contrast)')
    draw_relative_session_ratio(axes[1, 2], rel_times, fast_rt_ratio, slow_rt_ratio, fast_rt_err, slow_rt_err)
    draw_cond_probs(axes[1, 3], sl_cond_probs, sl_shuffled_cond_probs, cond_probs, nlag)

    fig.tight_layout()

    if not os.path.exists('figs/fig_behav'):
        os.makedirs('figs/fig_behav')
    out_path = 'figs/fig_behav/demo_panels.pdf'
    fig.savefig(out_path)
    print('Saved', out_path)
    plt.show()


if __name__ == '__main__':
    main()
