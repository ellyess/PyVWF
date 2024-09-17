import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

bg_colour = '#f0f0f0'
custom_params = {'xtick.bottom': True, 'axes.edgecolor': 'black', 'axes.spines.right': False, 'axes.spines.top': False, 'mathtext.default': 'regular'}
sns.set_theme(style='ticks', rc=custom_params)

def plot_overall_error(run, country, df_metrics, name):
    # fig, axes = plt.subplots(1, 2, figsize=(6.67, 3)) # for normal
    # fig, axes = plt.subplots(1, 2, figsize=(6.8, 3)) # for train with the legend
    fig, axes = plt.subplots(1, 3, figsize=(12, 3)) # for train with the legend
    
    sns.lineplot(
        x='num_clu',
        y='rmse',
        hue ="time_res",
        style="time_res",
        hue_order = ['month', 'bimonth', 'season', 'fixed'],
        style_order= ['month', 'bimonth', 'season', 'fixed'],
        data = df_metrics[(df_metrics['time_res'] != 'uncorrected')],
        ax = axes[0],
        legend = False
    )
    axes[0].set_xscale('log')
    axes[0].set_ylabel('RMSE')
    axes[0].set_xlabel('Number of Clusters ($n_{clu}$)')
    axes[0].set_xticks([1, 10, 100,1000])
    # axes[0].set_yticks([0.0975, 0.1000, 0.1025,0.1050,0.1075,0.1000,0.1100])
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    sns.lineplot(
        x='num_clu',
        y='mae',
        hue ="time_res",
        style= "time_res",
        hue_order = ['month', 'bimonth', 'season', 'fixed'],
        style_order= ['month', 'bimonth', 'season', 'fixed'],
        data = df_metrics[(df_metrics['time_res'] != 'uncorrected')],
        ax = axes[1],
        legend = False
    )
    
    axes[1].set_xscale('log')
    axes[1].set_ylabel('MAE')
    axes[1].set_xlabel('Number of Clusters ($n_{clu}$)')
    axes[1].set_xticks([1, 10, 100,1000])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    sns.lineplot(
        x='num_clu',
        y='mbe',
        hue ="time_res",
        style= "time_res",
        hue_order = ['month', 'bimonth', 'season', 'fixed'],
        style_order= ['month', 'bimonth', 'season', 'fixed'],
        data = df_metrics[(df_metrics['time_res'] != 'uncorrected')],
        ax = axes[2],
        legend = True
    )
    
    axes[2].set_xscale('log')
    axes[2].set_ylabel('MBE')
    axes[2].set_xlabel('Number of Clusters ($n_{clu}$)')
    axes[2].set_xticks([1, 10, 100,1000])
    axes[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    
    # fig.suptitle(name)
    axes[2].get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Monthly', 'Bimonthly',  'Seasonal', 'Fixed'] # renaming labels
    # plt.figlegend(handles, labels, loc = 'center left', bbox_to_anchor=(-0.25, 0.5), ncol=1, title='Temporal Frequency\n            ($t_{freq}$)', frameon=False)
    plt.tight_layout()
    plt.savefig(run+'/plots/'+country+'_'+name+'_error_appendix.png', bbox_inches='tight')
    return print("Saved to "+str(run+'/plots/'+country+'_'+name+'_error.png')+"\n")