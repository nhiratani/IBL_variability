#
# demo2.py
#
# End-to-end demo (companion to demo.py): loads the IBL ephys behavioral data
# and renders a single figure with the behavioral-variability panels used in
# Figure 2 of the paper.
#
# Each panel faithfully reproduces a snippet of the existing plotting code, but
# drawn onto shared axes (ax.* calls) instead of separate figures:
#
#   [0,0] Fast vs slow response ratio (colored by impulsivity) <- behav_analysis.plot_impulsivity_stats        L117-L132
#   [0,1] Within-animal impulsivity variability vs shuffled    <- behav_analysis.plot_within_animal_variability L511-L518
#   [0,2] Impulsivity vs reward rate                           <- behav_analysis.plot_impulsivity_stats        L187-L193
#   [0,3] Impulsivity vs trials/session                        <- behav_analysis.plot_impulsivity_stats        L173-L179
#   [1,0] Impulsivity by lab (box plot)                        <- behav_analysis.plot_impulsivity_stats2       L381-L392
#   [1,1] Impulsivity by sex (histogram)                       <- behav_analysis.plot_impulsivity_stats2       L316-L323
#   [1,2] Impulsivity vs median RT (by sex)                    <- behav_analysis.plot_medianRT_stats           L602-L606
#
#
import os
import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from util.data_loading import load_data, one
from RT_analysis import get_lab_list
from behav_analysis import process_data, derive_colors, lab_clrs, clr2s

np.random.seed(0)


# ---------------------------------------------------------------------------
# Panel data computations (each mirrors the corresponding block in the source)
# ---------------------------------------------------------------------------

def compute_impulsivity_ratios(subj_data, min_sessions):
    """Per-subject fast/slow response ratios and impulsivity (source L104-L107)."""
    subjects = list(subj_data.keys())
    ratio_fast = [subj_data[s]['num_fast'] / subj_data[s]['num_trials']
                  for s in subjects if subj_data[s]['num_sessions'] >= min_sessions]
    ratio_slow = [subj_data[s]['num_slow'] / subj_data[s]['num_trials']
                  for s in subjects if subj_data[s]['num_sessions'] >= min_sessions]
    impulsivity = np.array(ratio_fast) - np.array(ratio_slow)
    return np.array(ratio_fast), np.array(ratio_slow), impulsivity


def compute_trials_rewards_labcolors(subj_data, subject_info, min_sessions):
    """Per-subject trials/session, reward rate, and lab colors (source L144, L168-L171, L183)."""
    subjects = list(subj_data.keys())
    num_trials_per_session = [subj_data[s]['num_trials'] / subj_data[s]['num_sessions']
                              for s in subjects if subj_data[s]['num_sessions'] >= min_sessions]
    reward_rates = [subj_data[s]['num_rewarded'] / subj_data[s]['num_trials']
                    for s in subjects if subj_data[s]['num_sessions'] >= min_sessions]
    lab_list = get_lab_list()
    color_subj_lab = [lab_clrs[lab_list.index(subject_info[s]['lab'])]
                      for s in subjects if subj_data[s]['num_sessions'] >= min_sessions]
    return num_trials_per_session, reward_rates, color_subj_lab


def compute_within_animal_variability(data, subj_data, params):
    """Within-animal impulsivity SD and its shuffled null (source L460-L495)."""
    min_trials = params['min_trials']
    min_sessions = params['min_sessions']
    s_cutoff = params['s_cutoff']

    within_animal_variability = []
    for subject in data.keys():
        num_sessions = len(data[subject])
        if subj_data[subject]['num_sessions'] >= min_sessions:
            impulsivity_per_animal_session = []
            for sidx in range(num_sessions):
                # here num_trials is the number of valid (post-cutoff) trials
                if data[subject][sidx]['num_trials'] > min_trials - s_cutoff:
                    itmp = (data[subject][sidx]['num_fast'] - data[subject][sidx]['num_slow']) / data[subject][sidx]['num_trials']
                    impulsivity_per_animal_session.append(itmp)
            within_animal_variability.append(np.std(impulsivity_per_animal_session, ddof=1))

    # Null: randomly re-partition sessions into pseudo-animals of matching size
    impulsivity_array = []
    num_sessions_array = []
    for subject in data.keys():
        num_sessions = len(data[subject])
        if subj_data[subject]['num_sessions'] >= min_sessions:
            nsidx = 0
            for sidx in range(num_sessions):
                if data[subject][sidx]['num_trials'] > min_trials - s_cutoff:
                    impulsivity_array.append((data[subject][sidx]['num_fast'] - data[subject][sidx]['num_slow']) / data[subject][sidx]['num_trials'])
                    nsidx += 1
            num_sessions_array.append(nsidx)

    impulsivity_array = np.array(impulsivity_array)
    num_sessions_array = np.array(num_sessions_array)
    std_unweighted = []
    for _ in range(1000):
        impulsivity_array = np.random.permutation(impulsivity_array)
        sidx = 0
        for j in num_sessions_array:
            std_unweighted.append(np.std(impulsivity_array[sidx:sidx + j], ddof=1))
            sidx = sidx + j

    return within_animal_variability, std_unweighted


def compute_behav_stats(subj_data, subject_info):
    """Per-subject summary (impulsivity, sex, lab) for grouping (source L280-L292)."""
    behav_stats = {}
    for subject in subj_data.keys():
        if subj_data[subject]['num_trials'] > 0:
            ratio_fast = subj_data[subject]['num_fast'] / subj_data[subject]['num_trials']
            ratio_slow = subj_data[subject]['num_slow'] / subj_data[subject]['num_trials']
            behav_stats[subject] = {
                'num_sessions': subj_data[subject]['num_sessions'],
                'impulsivity': ratio_fast - ratio_slow,
                'sex': subject_info[subject]['sex'],
                'lab': subject_info[subject]['lab'],
            }
    return behav_stats


def compute_impulsivity_by_sex(behav_stats, min_sessions):
    """Impulsivity split into [female, male] (source L294-L300)."""
    impulsivity_by_sex = [[], []]
    for subject in behav_stats.keys():
        if behav_stats[subject]['num_sessions'] >= min_sessions:
            if behav_stats[subject]['sex'] == 'F':
                impulsivity_by_sex[0].append(behav_stats[subject]['impulsivity'])
            elif behav_stats[subject]['sex'] == 'M':
                impulsivity_by_sex[1].append(behav_stats[subject]['impulsivity'])
    return impulsivity_by_sex


def compute_impulsivity_by_lab(behav_stats, min_sessions):
    """Impulsivity grouped per lab, ordered by get_lab_list() (source L350-L365)."""
    impulsivity_by_lab = {}
    for subject in behav_stats.keys():
        if behav_stats[subject]['num_sessions'] >= min_sessions and not np.isnan(behav_stats[subject]['impulsivity']):
            lab = behav_stats[subject]['lab']
            impulsivity_by_lab.setdefault(lab, []).append(behav_stats[subject]['impulsivity'])

    lab_list = get_lab_list()
    list_impulsivity_by_lab = [[] for _ in lab_list]
    for lab in impulsivity_by_lab.keys():
        list_impulsivity_by_lab[lab_list.index(lab)] = impulsivity_by_lab[lab]
    return list_impulsivity_by_lab


def compute_medianRT_by_sex(data, subj_data, subject_info, min_sessions):
    """Per-subject median RT and impulsivity, split by sex (source L544-L594)."""
    medianRTs = {}
    for subject in data.keys():
        sbj_RTs = []
        for sidx in range(len(data[subject])):
            RTtmp = data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times']
            sbj_RTs.extend(RTtmp)
        medianRTs[subject] = np.nanmedian(sbj_RTs)

    medianRT_by_sex = [[], []]
    impulsivity_by_sex = [[], []]
    for subject in medianRTs.keys():
        if subj_data[subject]['num_sessions'] >= min_sessions:
            if subject_info[subject]['sex'] == 'F':
                sex_id = 0
            elif subject_info[subject]['sex'] == 'M':
                sex_id = 1
            else:
                continue
            medianRT_by_sex[sex_id].append(medianRTs[subject])
            impulsivity_by_sex[sex_id].append(
                (subj_data[subject]['num_fast'] - subj_data[subject]['num_slow']) / subj_data[subject]['num_trials'])
    return impulsivity_by_sex, medianRT_by_sex


# ---------------------------------------------------------------------------
# Panel drawing (ax.* versions of the source plotting snippets)
# ---------------------------------------------------------------------------

def draw_fast_vs_slow(fig, ax, ratio_fast, ratio_slow, impulsivity):
    """source L117-L132"""
    ax.scatter(ratio_fast, ratio_slow, color=derive_colors(impulsivity), s=50)
    ax.set_xlim(-0.01, 0.45)
    ax.set_ylim(-0.01, 0.45)

    # Colorbar keyed to impulsivity (source builds the identical viridis mapping)
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=Normalize(vmin=np.min(impulsivity), vmax=np.max(impulsivity)))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([-0.3, -0.15, 0.0, 0.15, 0.3])
    cbar.set_label('ATI')

    ax.set_title('Early vs Late response ratio')
    ax.set_xlabel('Early response ratio')
    ax.set_ylabel('Late response ratio')


def draw_within_animal_variability(ax, within_animal_variability, std_unweighted):
    """source L511-L518"""
    ax.hist(std_unweighted, bins=50, density=True, alpha=0.5, color='gray', label='shuffled')
    ax.hist(within_animal_variability, density=True, alpha=0.5, color=clr2s[0], label='within-animal')
    ax.set_xlim(0.0, 0.5)
    ax.legend()
    ax.set_title('Within-animal variability')
    ax.set_xlabel('ATI variability (SD)')
    ax.set_ylabel('Probability density')


def draw_impulsivity_vs_trials(ax, impulsivity, num_trials_per_session, color_subj_lab):
    """source L173-L179 (regression from L145)"""
    slope, intercept, r, p, se = scist.linregress(impulsivity, num_trials_per_session)
    ax.scatter(impulsivity, num_trials_per_session, color=color_subj_lab, s=50)
    xs = np.arange(-0.5, 0.5, 0.01)
    #ax.plot(xs, xs * slope + intercept, color='k', lw=1.0)
    ax.set_xlim(-0.45, 0.45)
    ax.set_title('ATI vs trials/session')
    ax.set_xlabel('Anticipatory tendency')
    ax.set_ylabel('Trials per session')


def draw_impulsivity_vs_reward(ax, impulsivity, reward_rates, color_subj_lab):
    """source L187-L193"""
    slope, intercept, r, p, se = scist.linregress(impulsivity, reward_rates)
    ax.scatter(impulsivity, reward_rates, color=color_subj_lab, s=50)
    xs = np.arange(-0.5, 0.5, 0.01)
    #ax.plot(xs, xs * slope + intercept, color='k', lw=1.0)
    ax.set_xlim(-0.45, 0.45)
    ax.set_title('ATI vs reward rate')
    ax.set_xlabel('Anticipatory tendency')
    ax.set_ylabel('Task performance')


def draw_impulsivity_by_lab(ax, list_impulsivity_by_lab):
    """source L381-L392"""
    lab_clrs2 = []
    for lab_clr in lab_clrs:
        lab_clrs2.extend([lab_clr, lab_clr])
    bp = ax.boxplot(list_impulsivity_by_lab)
    for patch, color in zip(bp['boxes'], lab_clrs):
        plt.setp(patch, color=color)
    for patch, color in zip(bp['whiskers'], lab_clrs2):
        plt.setp(patch, color=color)
    for patch, color in zip(bp['caps'], lab_clrs2):
        plt.setp(patch, color=color)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
    ax.set_title('ATI by lab')
    ax.set_xlabel('Lab id')
    ax.set_ylabel('Anticipatory tendency')


def draw_impulsivity_by_sex(ax, impulsivity_by_sex):
    """source L316-L323"""
    ax.hist(impulsivity_by_sex[1], bins=10, color='C1', alpha=0.5, label='M')
    ax.hist(impulsivity_by_sex[0], bins=10, color='C0', alpha=0.5, label='F')
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.legend()
    ax.set_title('ATI by sex')
    ax.set_xlabel('Anticipatory tendency')
    ax.set_ylabel('Number of animals')


def draw_impulsivity_vs_medianRT(ax, impulsivity_by_sex, medianRT_by_sex):
    """source L602-L606"""
    sex_clrs = ['C0', 'C1']
    sex_labels = ['F', 'M']
    for sex_id in range(2):
        ax.scatter(impulsivity_by_sex[sex_id], medianRT_by_sex[sex_id],
                   color=sex_clrs[sex_id], s=50, label=sex_labels[sex_id])
    ax.set_ylim(0.0, 0.5)
    ax.legend()
    ax.set_title('ATI vs median RT')
    ax.set_xlabel('Anticipatory tendency')
    ax.set_ylabel('Median RT [s]')


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
    min_sessions = params['min_sessions']

    print('Loading behavioral data...')
    data, subject_info = load_data(params['session_type'], one)
    # process_data adds per-session/per-subject fast/slow/trial counts used by every panel
    data, subj_data = process_data(data, params)

    print('Computing panels...')
    ratio_fast, ratio_slow, impulsivity = compute_impulsivity_ratios(subj_data, min_sessions)
    num_trials_per_session, reward_rates, color_subj_lab = compute_trials_rewards_labcolors(subj_data, subject_info, min_sessions)
    within_animal_variability, std_unweighted = compute_within_animal_variability(data, subj_data, params)
    behav_stats = compute_behav_stats(subj_data, subject_info)
    impulsivity_by_sex = compute_impulsivity_by_sex(behav_stats, min_sessions)
    list_impulsivity_by_lab = compute_impulsivity_by_lab(behav_stats, min_sessions)
    mrt_impulsivity_by_sex, medianRT_by_sex = compute_medianRT_by_sex(data, subj_data, subject_info, min_sessions)

    print('Rendering figure...')
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    draw_fast_vs_slow(fig, axes[0, 0], ratio_fast, ratio_slow, impulsivity)
    draw_within_animal_variability(axes[0, 1], within_animal_variability, std_unweighted)
    draw_impulsivity_vs_reward(axes[0, 2], impulsivity, reward_rates, color_subj_lab)
    draw_impulsivity_vs_trials(axes[0, 3], impulsivity, num_trials_per_session, color_subj_lab)
    draw_impulsivity_by_lab(axes[1, 0], list_impulsivity_by_lab)
    draw_impulsivity_by_sex(axes[1, 1], impulsivity_by_sex)
    draw_impulsivity_vs_medianRT(axes[1, 2], mrt_impulsivity_by_sex, medianRT_by_sex)
    axes[1, 3].axis('off')  # only seven panels

    fig.tight_layout()

    if not os.path.exists('figs/fig_behav'):
        os.makedirs('figs/fig_behav')
    out_path = 'figs/fig_behav/demo2_panels.pdf'
    fig.savefig(out_path)
    print('Saved', out_path)
    plt.show()


if __name__ == '__main__':
    main()
