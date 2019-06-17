import logging
import operator
import math

from functools import reduce
from itertools import repeat

import numpy as np
import datajoint as dj

import scipy.stats as sc_stats

from . import lab
from . import experiment
from . import ephys
[lab, experiment, ephys]  # NOQA

from . import get_schema_name

schema = dj.schema(get_schema_name('psth'))
log = logging.getLogger(__name__)

# NOW:
# - [X] rename UnitCondition to TrialCondition
# - [X] store actual Selectivity value
# - remove Condition & refactor
#   - provide canned queries
#   - (old? also null filtering funs?)


@schema
class TrialCondition(dj.Manual):
    definition = """
    # manually curated conditions of interest
    condition_id:                               int
    ---
    condition_desc:                             varchar(4096)
    """

    class TaskProtocol(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.TaskProtocol
        """

    class TrialInstruction(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.TrialInstruction
        """

    class EarlyLick(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.EarlyLick
        """

    class Outcome(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.Outcome
        """

    class PhotostimLocation(dj.Part):
        definition = """
        -> master
        ---
        -> experiment.Photostim
        """

    @classmethod
    def expand(cls, condition_id):
        """
        Expand the given condition_id into a dictionary containing the
        fetched sub-parts of the condition.
        """

        self = cls()
        key = {'condition_id': condition_id}

        return {
            'Condition': (self & key).fetch1(),
            'TaskProtocol':
                (TrialCondition.TaskProtocol & key).fetch(as_dict=True),
            'TrialInstruction':
                (TrialCondition.TrialInstruction & key).fetch(as_dict=True),
            'EarlyLick':
                (TrialCondition.EarlyLick & key).fetch(as_dict=True),
            'Outcome':
                (TrialCondition.Outcome & key).fetch(as_dict=True),
            'PhotostimLocation':
                (TrialCondition.PhotostimLocation & key).fetch(as_dict=True)
        }

    @classmethod
    def trials(cls, cond, r={}):
        """
        Get trials for a Condition.

        Accepts either a condition_id as an integer, or the output of
        the 'expand' function, above.

        Each Condition 'part' defined in the Condition is filtered
        to a primary key for the associated child table (pk_map),
        and then restricted through the table defined in 'restrict_map'
        along with experiment.SessionTrial to retrieve the corresponding
        trials for that particular part. In other words, the pseudo-query:

          SessionTrial & restrict_map & pk_map[Condition.Part & cond]

        is performed for each of the trial-parts.

        The intersection of these trial-part results are then combined
        locally to build the result, which is a list of SessionTrial keys.

        The parameter 'r' can be used to add additional query restrictions,
        currently applied to all of the sub-queries.
        """

        self = cls()
        if isinstance(cond, (int, np.integer)):
            cond = self.expand(cond)

        pk_map = {
            'TaskProtocol': experiment.TaskProtocol,
            'TrialInstruction': experiment.TrialInstruction,
            'EarlyLick': experiment.EarlyLick,
            'Outcome': experiment.Outcome,
            'PhotostimLocation': experiment.Photostim
        }
        restrict_map = {
            'TaskProtocol': experiment.BehaviorTrial,
            'TrialInstruction': experiment.BehaviorTrial,
            'EarlyLick': experiment.BehaviorTrial,
            'Outcome': experiment.BehaviorTrial,
            'PhotostimLocation': experiment.PhotostimEvent
        }

        res = []
        for c in cond:
            if c == 'Condition':
                continue

            tup = cond[c]
            tab = restrict_map[c]
            pk = pk_map[c].primary_key

            tup_keys = [{k: t[k] for k in t if k in pk}
                        for t in tup]
            trials = [(experiment.SessionTrial() & (tab() & t & r).proj())
                      for t in tup_keys]

            res.append({tuple(i.values()) for i in
                        reduce(operator.add, (t.proj() for t in trials))})

        return [{'subject_id': t[0], 'session': t[1], 'trial': t[2]}
                for t in sorted(set.intersection(*res))]

    @classmethod
    def populate(cls):
        """
        Table contents for Condition.

        Currently there is no way to initialize a dj.Lookup with parts,
        so we leave contents blank and create a function to explicitly insert.

        This is not run implicitly since it requires database write access.
        """
        self = cls()

        #
        # Condition 0: Audio Delay Task - Contra Hit
        #

        cond_key = {
            'condition_id': 0,
            'condition_desc': 'audio delay contra hit - nostim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 1: Audio Delay Task - Ipsi Hit
        #

        cond_key = {
            'condition_id': 1,
            'condition_desc': 'audio delay ipsi hit - nostim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 2: Audio Delay Task - Contra Error
        #

        cond_key = {
            'condition_id': 2,
            'condition_desc': 'audio delay contra error - nostim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 3: Audio Delay Task - Ipsi Error
        #

        cond_key = {
            'condition_id': 3,
            'condition_desc': 'audio delay ipsi error - nostim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 4: Audio Delay Task - Contra Hit
        #

        cond_key = {
            'condition_id': 4,
            'condition_desc': 'audio delay contra hit - onlystim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 5: Audio Delay Task - Ipsi Hit
        #

        cond_key = {
            'condition_id': 5,
            'condition_desc': 'audio delay ipsi hit - onlystim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='hit'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 6: Audio Delay Task - Contra Error
        #

        cond_key = {
            'condition_id': 6,
            'condition_desc': 'audio delay contra error - onlystim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='right'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)

        #
        # Condition 7: Audio Delay Task - Ipsi Error
        #

        cond_key = {
            'condition_id': 7,
            'condition_desc': 'audio delay ipsi error - onlystim'
        }
        self.insert1(cond_key, skip_duplicates=True)

        TrialCondition.TaskProtocol.insert1(
            dict(cond_key, task='audio delay', task_protocol=1),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.TrialInstruction.insert1(
            dict(cond_key, trial_instruction='left'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.EarlyLick.insert1(
            dict(cond_key, early_lick='no early'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.Outcome.insert1(
            dict(cond_key, outcome='miss'),
            skip_duplicates=True, ignore_extra_fields=True)

        TrialCondition.PhotostimLocation.insert(
            [dict(cond_key, **ploc) for ploc
             in experiment.Photostim & {'brainloc_id': 0}],
            skip_duplicates=True, ignore_extra_fields=True)


@schema
class UnitPsth(dj.Computed):
    definition = """
    -> TrialCondition
    -> ephys.Unit
    ---
    unit_psth=NULL:                             longblob
    """
    psth_params = {'xmin': -3, 'xmax': 3, 'binsize': 0.04}

    def make(self, key):
        log.info('UnitPsth.make(): key: {}'.format(key))

        unit = {k: v for k, v in key.items() if k in ephys.Unit.primary_key}

        # Expand Condition -
        # could conditionalize e.g.
        # if key['condition_id'] in [1,2,3]: self.make_thiskind(key), etc.
        # for now, we assume one method of processing.

        cond = TrialCondition.expand(key['condition_id'])

        all_trials = TrialCondition.trials({
            'TaskProtocol': cond['TaskProtocol'],
            'TrialInstruction': cond['TrialInstruction'],
            'EarlyLick': cond['EarlyLick'],
            'Outcome': cond['Outcome']})

        photo_trials = TrialCondition.trials({
            'PhotostimLocation': cond['PhotostimLocation']})

        # HACK special case stim condition logic -
        #   ... should be fixed by expanding Condition support logic.
        if 'onlystim' in cond['Condition']['condition_desc']:
            tgt_trials = [t for t in all_trials if t in photo_trials]
        elif 'nostim' in cond['Condition']['condition_desc']:
            tgt_trials = [t for t in all_trials if t not in photo_trials]
        else:
            tgt_trials = all_trials

        q = (ephys.TrialSpikes() & unit & tgt_trials)
        spikes = q.fetch('spike_times')

        if len(spikes) == 0:
            log.warning('no spikes found for key {} - null psth'.format(key))
            self.insert1(key)
            return

        spikes = np.concatenate(spikes)
        xmin, xmax, bins = self.psth_params.values()
        # XXX: xmin, xmax+bins (149 here vs 150 in matlab)..
        #   See also [:1] slice in plots..
        psth = list(np.histogram(spikes, bins=np.arange(xmin, xmax, bins)))
        psth[0] = psth[0] / len(tgt_trials) / bins

        self.insert1({**key, 'unit_psth': np.array(psth)})

    @classmethod
    def get(cls, condition_key, unit_key,
            incl_conds=['TaskProtocol', 'TrialInstruction', 'EarlyLick',
                        'Outcome'],
            excl_conds=['PhotostimLocation']):
        """
        Retrieve / build data needed for a Unit PSTH Plot based on the given
        unit condition and included / excluded condition (sub-)variables.

        Returns a dictionary of the form:

          {
             'trials': ephys.TrialSpikes.trials,
             'spikes': ephys.TrialSpikes.spikes,
             'psth': UnitPsth.unit_psth,
             'raster': Spike * Trial raster [np.array, np.array]
          }

        """

        condition = TrialCondition.expand(condition_key['condition_id'])
        session_key = {k: unit_key[k] for k in experiment.Session.primary_key}

        psth_q = (UnitPsth & {**condition_key, **unit_key})
        psth = psth_q.fetch1()['unit_psth']

        i_trials = TrialCondition.trials({k: condition[k] for k in incl_conds},
                                         session_key)

        x_trials = TrialCondition.trials({k: condition[k] for k in excl_conds},
                                         session_key)

        st_q = ((ephys.TrialSpikes & i_trials & unit_key) -
                (experiment.SessionTrial & x_trials & unit_key))

        spikes, trials = st_q.fetch('spike_times', 'trial',
                                    order_by='trial asc')

        raster = [np.concatenate(spikes),
                  np.concatenate([[t] * len(s)
                                  for s, t in zip(spikes, trials)])]

        return dict(trials=trials, spikes=spikes, psth=psth, raster=raster)



@schema
class UnitSelectivityChris(dj.Computed):
    """
    Compute unit selectivity for a unit in a particular time period.
    Calculation:
    2 tail t significance of unit firing rate for trial type: CorrectLeft vs. CorrectRight (no stim, no early lick)
    frequency = nspikes(period)/len(period)
    """
    definition = """
    -> ephys.Unit
    ---
    sample_selectivity=Null:    float         # sample period selectivity
    delay_selectivity=Null:     float         # delay period selectivity
    go_selectivity=Null:        float         # go period selectivity
    global_selectivity=Null:    float         # global selectivity
    min_selectivity=Null:       float         # (sample|delay|go) selectivity
    sample_preference=Null:     boolean       # sample period pref. (i|c)
    delay_preference=Null:      boolean       # delay period pref. (i|c)
    go_preference=Null:         boolean       # go period pref. (i|c)
    global_preference=Null:     boolean       # global non-period pref. (i|c)
    any_preference=Null:        boolean       # any period pref. (i|c)
    """

    alpha = 0.05  # default alpha value

    @property
    def selective(self):
        return 'min_selectivity<{}'.format(self.alpha)

    ipsi_preferring = 'global_preference=1'
    contra_preferring = 'global_preference=0'

    key_source = ephys.Unit & 'unit_quality = "good"'

    def make(self, key):
        log.debug('Selectivity.make(): key: {}'.format(key))

        # Verify insertion location is present,
        egpos = None
        try:
            egpos = (ephys.ProbeInsertion.InsertionLocation
                     * experiment.BrainLocation & key).fetch1()
        except dj.DataJointError as e:
            if 'exactly one tuple' in repr(e):
                log.error('... Insertion Location missing. skipping')
                return

        # retrieving the spikes of interest,
        spikes_q = ((ephys.TrialSpikes & key)
                    & (experiment.BehaviorTrial()
                       & {'task': 'audio delay'}
                       & {'early_lick': 'no early'}
                       & {'outcome': 'hit'}) - experiment.PhotostimEvent)

        # and their corresponding behavior,
        lr = ['left', 'right']
        behav = (experiment.BehaviorTrial & spikes_q.proj()).fetch(
            order_by='trial asc')
        behav_lr = {k: np.where(behav['trial_instruction'] == k) for k in lr}

        if egpos['hemisphere'] == 'left':
            behav_i = behav_lr['left']
            behav_c = behav_lr['right']
        else:
            behav_i = behav_lr['right']
            behav_c = behav_lr['left']

        # constructing a square, nan-padded trial x spike array
        spikes = spikes_q.fetch(order_by='trial asc')
        ydim = max(len(i['spike_times']) for i in spikes)
        square = np.array(
            np.array([np.concatenate([st, pad])[:ydim]
                      for st, pad in zip(spikes['spike_times'],
                                         repeat([math.nan]*ydim))]))

        criteria = {}  # with which to calculate the selctivity criteria.

        ranges = self.ranges
        periods = list(ranges.keys())

        for period in periods:
            bounds = ranges[period]
            name = period + '_selectivity'
            pref = period + '_preference'

            lower_mask = np.ma.masked_greater_equal(square, bounds[0])
            upper_mask = np.ma.masked_less_equal(square, bounds[1])
            inrng_mask = np.logical_and(lower_mask.mask, upper_mask.mask)

            rsum = np.sum(inrng_mask, axis=1)
            dur = bounds[1] - bounds[0]
            freq = rsum / dur

            freq_i = freq[behav_i]
            freq_c = freq[behav_c]
            t_stat, pval = sc_stats.ttest_ind(freq_i, freq_c, equal_var=False)

            criteria[name] = pval

            if period != 'global':
                criteria[pref] = (1 if np.average(freq_i)
                                  > np.average(freq_c) else 0)
            else:
                min_sel = min([v for k, v in criteria.items()
                               if 'selectivity' in k])

                any_pref = any([v for k, v in criteria.items()
                                if 'preference' in k])

                criteria['min_selectivity'] = min_sel
                criteria['any_preference'] = any_pref

                # XXX: hacky.. best would be to have another value
                gbl_pref = (1 if ((np.average(freq_i) > np.average(freq_c))
                                  and (min_sel <= self.alpha)) else 0)

                criteria[pref] = gbl_pref

        self.insert1({**key, **criteria})


@schema
class Selectivity(dj.Lookup):
    definition = """
    selectivity: varchar(24)
    """

    contents = zip(['contra-selective', 'ipsi-selective', 'non-selective'])


@schema
class UnitSelectivity(dj.Computed):
    """
    Compute unit selectivity for a unit in a particular time period.
    Calculation:
    2 tail t significance of unit firing rate for trial type: CorrectLeft vs. CorrectRight (no stim, no early lick)
    frequency = nspikes(period)/len(period)
    """
    definition = """
    -> ephys.Unit
    ---
    -> Selectivity.proj(unit_selectivity='selectivity')
    """

    class PeriodSelectivity(dj.Part):
        definition = """
        -> master
        -> experiment.Period
        ---
        -> Selectivity.proj(period_selectivity='selectivity')
        contra_firing_rate: float  # mean firing rate of all contra-trials
        ipsi_firing_rate: float  # mean firing rate of all ipsi-trials
        p_value: float  # p-value of the t-test of spike-rate of all trials
        """

    key_source = ephys.Unit & 'unit_quality = "good"'

    def make(self, key):
        log.debug('Selectivity.make(): key: {}'.format(key))

        trial_restrictor = {'task': 'audio delay', 'task_protocol': 1,
                            'outcome': 'hit', 'early_lick': 'no early'}
        correct_right = {**trial_restrictor, 'trial_instruction': 'right'}
        correct_left = {**trial_restrictor, 'trial_instruction': 'left'}

        # get trial spike times
        right_trialspikes = (ephys.TrialSpikes * experiment.BehaviorTrial
                             - experiment.PhotostimTrial & key & correct_right).fetch('spike_times', order_by='trial')
        left_trialspikes = (ephys.TrialSpikes * experiment.BehaviorTrial
                            - experiment.PhotostimTrial & key & correct_left).fetch('spike_times', order_by='trial')

        unit_hemi = (ephys.ProbeInsertion.InsertionLocation * experiment.BrainLocation & key).fetch1('hemisphere')

        if unit_hemi not in ('left', 'right'):
            raise Exception('Hemisphere Error! Unit not belonging to either left or right hemisphere')

        contra_trialspikes = right_trialspikes if unit_hemi == 'left' else left_trialspikes
        ipsi_trialspikes = left_trialspikes if unit_hemi == 'left' else right_trialspikes

        period_selectivity = []
        for period in experiment.Period.fetch(as_dict=True):
            period_dur = period['period_end'] - period['period_start']
            contra_trial_spk_rate = [(np.logical_and(t >= period['period_start'],
                                                      t < period['period_end'])).astype(int).sum() / period_dur
                             for t in contra_trialspikes]
            ipsi_trial_spk_rate = [(np.logical_and(t >= period['period_start'],
                                                    t < period['period_end'])).astype(int).sum() / period_dur
                           for t in ipsi_trialspikes]

            contra_frate = np.mean(contra_trial_spk_rate)
            ipsi_frate = np.mean(ipsi_trial_spk_rate)

            # do t-test on the spike-count per trial for all contra trials vs. ipsi trials
            t_stat, pval = sc_stats.ttest_ind(contra_trial_spk_rate, ipsi_trial_spk_rate)

            if pval > 0.05:
                pref = 'non-selective'
            else:
                pref = 'ipsi-selective' if ipsi_frate > contra_frate else 'contra-selective'

            period_selectivity.append(dict(key, **period, period_selectivity=pref, p_value=pval,
                                            contra_firing_rate=contra_frate, ipsi_firing_rate=ipsi_frate))

        unit_selective = not (np.array([p['period_selectivity'] for p in period_selectivity]) == 'non-selective').all()
        if not unit_selective:
            unit_pref = 'non-selective'
        else:
            ave_ipsi_frate = np.array([p['ipsi_firing_rate'] for p in period_selectivity]).mean()
            ave_contra_frate = np.array([p['contra_firing_rate'] for p in period_selectivity]).mean()
            unit_pref = 'ipsi-selective' if ave_ipsi_frate > ave_contra_frate else 'contra-selective'

        self.insert1(dict(**key, unit_selectivity=unit_pref))
        self.PeriodSelectivity.insert(period_selectivity, ignore_extra_fields=True)


# ---------- HELPER FUNCTIONS --------------

def compute_unit_psth(unit_key, trial_keys):
    q = (ephys.TrialSpikes() & unit_key & trial_keys)
    spikes = q.fetch('spike_times')
    spikes = np.concatenate(spikes)

    xmin, xmax, bins = UnitPsth.psth_params.values()
    psth = list(np.histogram(spikes, bins = np.arange(xmin, xmax, bins)))
    psth[0] = psth[0] / len(trial_keys) / bins

    return np.array(psth)
