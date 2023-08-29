import datajoint as dj
import pathlib
import warnings
import numpy as np
import json
from datetime import datetime
from dateutil.tz import tzlocal
from datajoint.errors import DataJointError
import pynwb
from pynwb import NWBFile, NWBHDF5IO

from pipeline import lab, experiment, tracking, ephys, histology, psth, ccf
from pipeline.experiment import get_wr_sessdatetime
from pipeline.ingest import ephys as ephys_ingest
from pipeline.ingest import tracking as tracking_ingest
from pipeline.ingest.utils.paths import get_ephys_paths


# Helper functions for raw ephys data import
def get_electrodes_mapping(electrodes):
    """
    Create a mapping from the probe and electrode id to the row number of the electrodes
    table. This is used in the construction of the DynamicTableRegion that indicates what rows of the electrodes
    table correspond to the data in an ElectricalSeries.
    Parameters
    ----------
    electrodes: hdmf.common.table.DynamicTable
    Returns
    -------
    dict
    """
    return {
        (
            electrodes["group"][idx].device.name,
            electrodes["electrode"][idx],
        ): idx
        for idx in range(len(electrodes))
    }


def gains_helper(gains):
    """
    This handles three different cases for gains:
    1. gains are all 1. In this case, return conversion=1e-6, which applies to all
    channels and converts from microvolts to volts.
    2. Gains are all equal, but not 1. In this case, multiply this by 1e-6 to apply this
    gain to all channels and convert units to volts.
    3. Gains are different for different channels. In this case use the
    `channel_conversion` field in addition to the `conversion` field so that each
    channel can be converted to volts using its own individual gain.
    Parameters
    ----------
    gains: np.ndarray
    Returns
    -------
    dict
        conversion : float
        channel_conversion : np.ndarray
    """
    if all(x == 1 for x in gains):
        return dict(conversion=1e-6, channel_conversion=None)
    if all(x == gains[0] for x in gains):
        return dict(conversion=1e-6 * gains[0], channel_conversion=None)
    return dict(conversion=1e-6, channel_conversion=gains)


# Some constants to work with
zero_time = datetime.strptime(
    "00:00:00", "%H:%M:%S"
).time()  # no precise time available

DEV_POS_FOLDER_MAPPING = {
    "side": "Face",
    "bottom": "Bottom",
    "body": "Body"
}

# NWB exports

def datajoint_to_nwb(session_key):
    """
    Generate one NWBFile object representing all data
     coming from the specified "session_key" (representing one session)
    """
    _, sess_datetime = get_wr_sessdatetime(session_key)

    session_identifier = _get_session_identifier(session_key)

    experiment_description = (
        experiment.TaskProtocol & (experiment.BehaviorTrial & session_key)
    ).fetch1("task_protocol_description")

    try:
        session_descr = (experiment.SessionComment & session_key).fetch1(
            "session_comment"
        )
    except DataJointError:
        session_descr = ""

    nwbfile = NWBFile(identifier=session_identifier,
                      session_description=session_descr,
                      session_start_time=datetime.strptime(sess_datetime, '%Y%m%d_%H%M%S'),
                      file_create_date=datetime.now(tzlocal()),
                      experimenter=list((experiment.Session & session_key).fetch('username')),
                      data_collection='',
                      institution='Janelia Research Campus',
                      experiment_description=experiment_description,
                      related_publications='',
                      keywords=['electrophysiology'])

    # ==================================== SUBJECT ==================================
    subject = (
        lab.Subject * lab.WaterRestriction.proj("water_restriction_number")
        & session_key
    ).fetch1()
    nwbfile.subject = pynwb.file.Subject(
        subject_id=str(subject['subject_id']),
        date_of_birth=datetime.combine(subject['date_of_birth'], zero_time) if subject['date_of_birth'] else None,
        description=subject['water_restriction_number'],
        sex=subject['sex'],
        species='Mus musculus')

    # ==================================== EPHYS ==================================
    # add additional columns to the electrodes table
    electrodes_query = lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode
    for additional_attribute in ['electrode', 'shank', 'shank_col', 'shank_row']:
        nwbfile.add_electrode_column(
            name=electrodes_query.heading.attributes[additional_attribute].name,
            description=electrodes_query.heading.attributes[
                additional_attribute
            ].comment,
        )

    # add additional columns to the units table
    if dj.__version__ >= '0.13.0':
        units_query = (ephys.ProbeInsertion.RecordingSystemSetup
                       * ephys.Unit & session_key).proj(
            ..., '-spike_times', '-spike_sites', '-spike_depths').join(
            ephys.UnitStat, left=True).join(
            ephys.MAPClusterMetric.DriftMetric, left=True).join(
            ephys.ClusterMetric, left=True).join(
            ephys.WaveformMetric, left=True).join(
            ephys.SingleUnitClassification.UnitClassification, left=True)
    else:
        units_query = (ephys.ProbeInsertion.RecordingSystemSetup
                       * ephys.Unit & session_key).proj(
            ..., '-spike_times', '-spike_sites', '-spike_depths').aggr(
            ephys.UnitStat, ..., **{n: n for n in ephys.UnitStat.heading.names if n not in ephys.UnitStat.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.MAPClusterMetric.DriftMetric, ..., **{n: n for n in ephys.MAPClusterMetric.DriftMetric.heading.names if n not in ephys.MAPClusterMetric.DriftMetric.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.ClusterMetric, ..., **{n: n for n in ephys.ClusterMetric.heading.names if n not in ephys.ClusterMetric.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.WaveformMetric, ..., **{n: n for n in ephys.WaveformMetric.heading.names if n not in ephys.WaveformMetric.heading.primary_key},
            keep_all_rows=True).aggr(
            ephys.SingleUnitClassification.UnitClassification, ..., **{n: n for n in ephys.SingleUnitClassification.UnitClassification.heading.names if n not in ephys.SingleUnitClassification.UnitClassification.heading.primary_key},
            keep_all_rows=True)

    units_omitted_attributes = ['subject_id', 'session', 'insertion_number',
                                'clustering_method', 'unit_uid', 'probe_type',
                                'epoch_name_quality_metrics', 'epoch_name_waveform_metrics',
                                'electrode_config_name', 'electrode_group',
                                'electrode', 'waveform']

    for attr in units_query.heading.names:
        if attr in units_omitted_attributes:
            continue
        nwbfile.add_unit_column(
            name=units_query.heading.attributes[attr].name,
            description=units_query.heading.attributes[attr].comment,
        )

    # add a column for a set of manually annotated "good trial" for a particular probe insertion
    nwbfile.add_unit_column(
        name="is_good_trials",
        description="boolean array specifying for each trial if it is manually annotated 'good' trials for a particular probe insertion",
    )

    # iterate through curated clusterings and export units data
    for insert_key in (ephys.ProbeInsertion & session_key).fetch("KEY"):
        # ---- Probe Insertion Location ----
        if ephys.ProbeInsertion.InsertionLocation & insert_key:
            insert_location = {
                k: str(v) for k, v in (
                        (ephys.ProbeInsertion.proj() & insert_key).aggr(
                            ephys.ProbeInsertion.RecordableBrainRegion.proj(
                                ..., brain_region='CONCAT(hemisphere, " ", brain_area)'),
                            ..., brain_regions='GROUP_CONCAT(brain_region SEPARATOR ", ")')
                        * ephys.ProbeInsertion.InsertionLocation).fetch1().items()
                if k not in ephys.ProbeInsertion.primary_key}
            insert_location = json.dumps(insert_location)
        else:
            insert_location = "N/A"

        # ---- Electrode Configuration ----
        electrode_config = (lab.Probe * lab.ProbeType * lab.ElectrodeConfig
                            * ephys.ProbeInsertion & insert_key).fetch1()
        ephys_device_name = f'{electrode_config["probe"]} ({electrode_config["probe_type"]})'
        ephys_device = (nwbfile.get_device(ephys_device_name)
                        if ephys_device_name in nwbfile.devices
                        else nwbfile.create_device(name=ephys_device_name,
                                                   description=electrode_config["probe_type"]))

        electrode_group = nwbfile.create_electrode_group(
            name=f'{electrode_config["probe"]} {electrode_config["electrode_config_name"]}',
            description=json.dumps(electrode_config, default=str),
            device=ephys_device,
            location=insert_location,
        )

        electrode_query = (
            lab.ProbeType.Electrode * lab.ElectrodeConfig.Electrode & electrode_config
        )
        electrode_ccf = {
            e: {"x": float(x), "y": float(y), "z": float(z)}
            for e, x, y, z in zip(
                *(
                    histology.ElectrodeCCFPosition.ElectrodePosition & electrode_config
                ).fetch("electrode", "ccf_x", "ccf_y", "ccf_z")
            )
        }

        for electrode in electrode_query.fetch(as_dict=True):
            nwbfile.add_electrode(
                electrode=electrode['electrode'], group=electrode_group,
                filtering='', imp=-1.,
                **electrode_ccf.get(electrode['electrode'], {'x': np.nan, 'y': np.nan, 'z': np.nan}),
                rel_x=electrode['x_coord'], rel_y=electrode['y_coord'], rel_z=np.nan,
                shank=electrode['shank'], shank_col=electrode['shank_col'], shank_row=electrode['shank_row'],
                location=electrode_group.location)

        # add one more electrode for the SYNC channel
        nwbfile.add_electrode(
            electrode=max(electrode_query.fetch('electrode')) + 1,
            group=electrode_group,
            filtering='', imp=-1.,
            x=np.nan, y=np.nan, z=np.nan,
            rel_x=np.nan, rel_y=np.nan, rel_z=np.nan,
            shank=-1, shank_col=-1, shank_row=-1,
            location="N/A - SYNC")

        electrode_df = nwbfile.electrodes.to_dataframe()

        # ---- Units ----
        trials, go_cue_times, trial_starts, trial_stops = (
                experiment.TrialEvent * experiment.SessionTrial
                & (ephys.Unit.TrialSpikes & insert_key)
                & {'trial_event_type': 'go'}).fetch(
            'trial', 'trial_event_time', 'start_time', 'stop_time', order_by='trial')

        insertion_good_trial_q = ephys.ProbeInsertionQuality.GoodTrial & insert_key
        if insertion_good_trial_q:
            insertion_good_trials = insertion_good_trial_q.fetch('trial')
            is_good_trials = [t in insertion_good_trials for t in trials]
        else:
            is_good_trials = np.full_like(trials, True).astype(bool).tolist()

        unit_query = units_query & insert_key
        for unit in unit_query.fetch(as_dict=True):
            unit['id'] = max(nwbfile.units.id.data) + 1 if nwbfile.units.id.data else 0
            aligned_spikes = (ephys.Unit.TrialSpikes & unit).fetch(
                'spike_times', order_by='trial')
            raw_spike_times = []
            for aligned_spks, go_cue_time, trial_start, trial_stop in zip(
                    aligned_spikes, go_cue_times, trial_starts, trial_stops):
                unaligned_spikes = aligned_spks + float(trial_start) + float(go_cue_time)
                raw_spike_times.append(unaligned_spikes[np.logical_and(
                    unaligned_spikes >= trial_start, unaligned_spikes <= trial_stop)])
            spikes = np.concatenate(raw_spike_times).ravel()
            observed_times = np.array([trial_starts, trial_stops]).T.astype('float')
            unit['spike_times'] = spikes
            unit['obs_intervals'] = observed_times
            unit['electrodes'] = electrode_df.query(
                f'group_name == "{electrode_group.name}" & electrode == {unit.pop("electrode")}'
            ).index.values
            unit['waveform_mean'] = unit.pop('waveform')
            unit['waveform_sd'] = np.full(1, np.nan)
            unit['is_good_trials'] = is_good_trials

            for attr in list(unit.keys()):
                if attr in units_omitted_attributes:
                    unit.pop(attr)
                elif unit[attr] is None:
                    unit[attr] = np.nan

            unit['electrode_group'] = electrode_group

            nwbfile.add_unit(**unit)

    # =============================== PHOTO-STIMULATION ===============================
    stim_sites = {}
    photostim_query = experiment.Photostim & (experiment.PhotostimTrial & session_key)
    if photostim_query:
        for photostim_key in photostim_query.fetch("KEY"):
            photostim = (
                experiment.Photostim * lab.PhotostimDevice.proj("excitation_wavelength")
                & photostim_key
            ).fetch1()
            stim_device = (
                nwbfile.get_device(photostim["photostim_device"])
                if photostim["photostim_device"] in nwbfile.devices
                else nwbfile.create_device(name=photostim["photostim_device"])
            )

            stim_site = pynwb.ogen.OptogeneticStimulusSite(
                name=f'{photostim["photostim_device"]}_{photostim["photo_stim"]}',
                device=stim_device,
                excitation_lambda=float(photostim["excitation_wavelength"]),
                location=json.dumps(
                    [
                        {
                            k: v
                            for k, v in stim_locs.items()
                            if k not in experiment.Photostim.primary_key
                        }
                        for stim_locs in (
                            experiment.Photostim.PhotostimLocation & photostim_key
                        ).fetch(as_dict=True)
                    ],
                    default=str,
                ),
                description=f'excitation_duration: {photostim["duration"]}',
            )
            nwbfile.add_ogen_site(stim_site)
            stim_sites[photostim["photo_stim"]] = stim_site

    # =============================== TRACKING ===============================
    if tracking.Tracking & session_key:
        behav_acq = pynwb.behavior.BehavioralTimeSeries(name="BehavioralTimeSeries")
        nwbfile.add_acquisition(behav_acq)

        tracking_devices = (
            tracking.TrackingDevice & (tracking.Tracking & session_key)
        ).fetch(as_dict=True)

        for trk_device in tracking_devices:
            trk_device_name = (
                trk_device["tracking_device"].replace(" ", "")
                + "_"
                + trk_device["tracking_position"]
            )
            trk_fs = float(trk_device["sampling_rate"])
            for feature, feature_tbl in tracking.Tracking().tracking_features.items():
                ft_attrs = [
                    n
                    for n in feature_tbl.heading.names
                    if n not in feature_tbl.primary_key
                ]
                if feature_tbl & trk_device & session_key:
                    if feature == "WhiskerTracking":
                        additional_conditions = [
                            {"whisker_name": n}
                            for n in set(
                                (feature_tbl & trk_device & session_key).fetch(
                                    "whisker_name"
                                )
                            )
                        ]
                    else:
                        additional_conditions = [{}]
                    for r in additional_conditions:
                        samples, start_time, *position_data = (
                            experiment.SessionTrial * tracking.Tracking * feature_tbl
                            & session_key
                            & r
                        ).fetch(
                            "tracking_samples",
                            "start_time",
                            *ft_attrs,
                            order_by="trial",
                        )

                        tracking_timestamps = np.hstack([np.arange(nsample) / trk_fs + float(trial_start_time)
                                                         for nsample, trial_start_time in zip(samples, start_time)])
                        position_data = np.vstack([np.hstack(d) for d in position_data]).T

                        behav_ts_name = f"{trk_device_name}_{feature}" + (
                            f'_{r["whisker_name"]}' if r else ""
                        )

                        behav_acq.create_timeseries(name=behav_ts_name,
                                                    data=position_data,
                                                    timestamps=tracking_timestamps[:position_data.shape[0]],
                                                    description=f'Time series for {feature} position: {tuple(ft_attrs)}',
                                                    unit='a.u.',
                                                    conversion=1.0)

    # =============================== BEHAVIOR TRIALS ===============================
    # ---- TrialSet ----
    q_photostim = (experiment.PhotostimEvent * experiment.Photostim & session_key).proj(
        "photostim_event_time", "power", "duration"
    )
    q_trial = experiment.SessionTrial * experiment.BehaviorTrial & session_key
    if dj.__version__ >= '0.13.0':
        q_trial = q_trial.proj().aggr(
            q_photostim, ...,
            photostim_onset='IFNULL(GROUP_CONCAT(photostim_event_time SEPARATOR ", "), "N/A")',
            photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
            photostim_duration='IFNULL(GROUP_CONCAT(duration SEPARATOR ", "), "N/A")',
            keep_all_rows=True) * q_trial
    else:
        q_trial = q_trial.aggr(
            q_photostim, ...,
            photostim_onset='IFNULL(GROUP_CONCAT(photostim_event_time SEPARATOR ", "), "N/A")',
            photostim_power='IFNULL(GROUP_CONCAT(power SEPARATOR ", "), "N/A")',
            photostim_duration='IFNULL(GROUP_CONCAT(duration SEPARATOR ", "), "N/A")',
            keep_all_rows=True)

    skip_adding_columns = experiment.Session.primary_key

    if q_trial:
        # Get trial descriptors from TrialSet.Trial and TrialStimInfo
        trial_columns = {
            tag: {"name": tag, "description": q_trial.heading.attributes[tag].comment}
            for tag in q_trial.heading.names
            if tag not in skip_adding_columns + ["start_time", "stop_time"]
        }

        # Add new table columns to nwb trial-table
        for column in trial_columns.values():
            nwbfile.add_trial_column(**column)

        # Add entries to the trial-table
        for trial in q_trial.fetch(as_dict=True):
            trial["start_time"], trial["stop_time"] = float(trial["start_time"]), float(
                trial["stop_time"]
            )
            nwbfile.add_trial(
                **{k: v for k, v in trial.items() if k not in skip_adding_columns}
            )

    # =============================== BEHAVIOR TRIALS' EVENTS ===============================

    behavioral_event = pynwb.behavior.BehavioralEvents(name="BehavioralEvents")
    nwbfile.add_acquisition(behavioral_event)

    # ---- behavior events

    q_trial_event = (
        experiment.TrialEvent * experiment.SessionTrial & session_key
    ).proj(
        "trial_event_type",
        event_start="trial_event_time + start_time",
        event_stop="trial_event_time + start_time + duration",
    )

    for trial_event_type in (experiment.TrialEventType & q_trial_event).fetch('trial_event_type'):
        trial, event_starts, event_stops = (q_trial_event
                                            & {'trial_event_type': trial_event_type}).fetch(
            'trial', 'event_start', 'event_stop', order_by='trial, event_start')

        behavioral_event.create_timeseries(
            name=trial_event_type + "_start_times",
            unit="a.u.",
            conversion=1.0,
            data=np.full_like(event_starts.astype(float), 1),
            timestamps=event_starts.astype(float),
            description=f'Timestamps for event type: {trial_event_type} - Start Time')

        behavioral_event.create_timeseries(
            name=trial_event_type + "_stop_times",
            unit="a.u.",
            conversion=1.0,
            data=np.full_like(event_stops.astype(float), 1),
            timestamps=event_stops.astype(float),
            description=f'Timestamps for event type: {trial_event_type} - Stop Time')

    # ---- action events

    q_action_event = (
        experiment.ActionEvent * experiment.SessionTrial & session_key
    ).proj("action_event_type", event_time="action_event_time + start_time")

    for action_event_type in (experiment.ActionEventType & q_action_event).fetch(
        "action_event_type"
    ):
        trial, event_starts = (
            q_action_event & {"action_event_type": action_event_type}
        ).fetch("trial", "event_time", order_by="trial")

        behavioral_event.create_timeseries(
            name=action_event_type.replace(" ", "_") + "_times",
            unit="a.u.",
            conversion=1.0,
            data=np.full_like(event_starts.astype(float), 1),
            timestamps=event_starts.astype(float),
            description=f'Timestamps for event type: {action_event_type}')

    # ---- photostim events ----

    q_photostim_event = (
        experiment.PhotostimEvent
        * experiment.Photostim.proj("duration")
        * experiment.SessionTrial
        & session_key
    ).proj(
        "trial",
        "power",
        "photostim_event_time",
        event_start="photostim_event_time + start_time",
        event_stop="photostim_event_time + start_time + duration",
    )

    trials, event_starts, event_stops, powers, photo_stim = q_photostim_event.fetch(
        "trial", "event_start", "event_stop", "power", "photo_stim", order_by="trial"
    )

    behavioral_event.create_timeseries(
        name="photostim_start_times",
        unit="mW",
        conversion=1.0,
        description="Timestamps of the photo-stimulation and the corresponding powers (in mW) being applied",
        data=powers.astype(float),
        timestamps=event_starts.astype(float),
        control=photo_stim.astype("uint8"),
        control_description=stim_sites,
    )
    behavioral_event.create_timeseries(
        name="photostim_stop_times",
        unit="mW",
        conversion=1.0,
        description="Timestamps of the photo-stimulation being switched off",
        data=np.full_like(event_starts.astype(float), 0),
        timestamps=event_stops.astype(float),
        control=photo_stim.astype("uint8"),
        control_description=stim_sites,
    )

    return nwbfile


def _to_raw_ephys_nwb(session_key,
                      linked_nwb_file,
                      overwrite=False):
    if isinstance(linked_nwb_file, pynwb.NWBFile):
        raw_nwbfile = linked_nwb_file
        external_link = False
    else:
        linked_nwb_file = pathlib.Path(linked_nwb_file)
        raw_ephys_nwb_file = linked_nwb_file.parent / (linked_nwb_file.stem + "_raw_ephys.nwb")

        if raw_ephys_nwb_file.exists() and not overwrite:
            return

        assert pathlib.Path(linked_nwb_file).exists()

        io = NWBHDF5IO(linked_nwb_file, "r")
        linked_nwbfile = io.read()
        raw_nwbfile = linked_nwbfile.copy()
        external_link = True

    # ---- Raw Ephys Data ---
    import spikeinterface.extractors as se
    from nwb_conversion_tools.tools.spikeinterface.spikeinterfacerecordingdatachunkiterator import (
        SpikeInterfaceRecordingDataChunkIterator
    )
    ephys_root_data_dir = pathlib.Path(get_ephys_paths()[0])

    for insert_key in (ephys.ProbeInsertion & session_key).fetch("KEY"):
        ks_dir_relpath = (ephys_ingest.EphysIngest.EphysFile.proj(
            ..., insertion_number='probe_insertion_number')
                          & insert_key).fetch1('ephys_file')
        ks_dir = ephys_root_data_dir / ks_dir_relpath
        npx_dir = ks_dir.parent

        try:
            next(npx_dir.glob("*imec*.ap.bin"))
        except StopIteration:
            warnings.warn(f"No raw ephys file found at {npx_dir}")
            continue
        # except StopIteration:
        #     raise FileNotFoundError(
        #         f"No raw ephys file (.ap.bin) found at {npx_dir}"
        #     )
        electrode_config = (lab.Probe * lab.ProbeType * lab.ElectrodeConfig
                            * ephys.ProbeInsertion & insert_key).fetch1()
        ephys_device_name = f'{electrode_config["probe"]} ({electrode_config["probe_type"]})'
        ephys_device = raw_nwbfile.get_device(ephys_device_name)

        sampling_rate = (
                ephys.ProbeInsertion.RecordingSystemSetup & insert_key
        ).fetch1("sampling_rate")

        mapping = get_electrodes_mapping(raw_nwbfile.electrodes)

        extractor = se.read_spikeglx(npx_dir, load_sync_channel=True)

        conversion_kwargs = gains_helper(extractor.get_channel_gains())

        recording_channels_by_id = (
                lab.ElectrodeConfig.Electrode * ephys.ProbeInsertion & insert_key
        ).fetch("electrode")
        # add sync channel
        recording_channels_by_id = np.concatenate([recording_channels_by_id, [max(recording_channels_by_id) + 1]])

        insert_location = raw_nwbfile.electrode_groups[
            f'{electrode_config["probe"]} {electrode_config["electrode_config_name"]}'].location

        raw_nwbfile.add_acquisition(
            pynwb.ecephys.ElectricalSeries(
                name=f"ElectricalSeries-{insert_key['insertion_number']}",
                description=f"Ephys recording from probe '{ephys_device.name}', at location: {insert_location}",
                data=SpikeInterfaceRecordingDataChunkIterator(extractor),
                rate=float(sampling_rate),
                electrodes=raw_nwbfile.create_electrode_table_region(
                    region=[
                        mapping[(ephys_device.name, x)] for x in recording_channels_by_id
                    ],
                    name="electrodes",
                    description="recorded electrodes",
                ),
                **conversion_kwargs,
            )
        )

    if external_link:
        try:
            with NWBHDF5IO(raw_ephys_nwb_file.as_posix(), mode='w', manager=io.manager) as raw_io:
                raw_io.write(raw_nwbfile)
                print(f'\t\tWrite NWB 2.0 file: {raw_ephys_nwb_file.stem}')
        finally:
            io.close()
    else:
        return raw_nwbfile


def _to_raw_video_nwb(session_key,
                      linked_nwb_file,
                      overwrite=False):
    if isinstance(linked_nwb_file, pynwb.NWBFile):
        raw_nwbfile = linked_nwb_file
        external_link = False
    else:
        linked_nwb_file = pathlib.Path(linked_nwb_file)
        raw_video_nwb_file = linked_nwb_file.parent / (linked_nwb_file.stem + "_raw_video.nwb")

        if raw_video_nwb_file.exists() and not overwrite:
            return

        assert pathlib.Path(linked_nwb_file).exists()

        io = NWBHDF5IO(linked_nwb_file, "r")
        linked_nwbfile = io.read()
        raw_nwbfile = linked_nwbfile.copy()
        external_link = True

    # ----- Raw Video Files -----
    from nwb_conversion_tools.datainterfaces.behavior.movie.moviedatainterface import MovieInterface

    tracking_root_data_dir = pathlib.Path(tracking_ingest.get_tracking_paths()[0])

    tracking_files_info = (
            tracking_ingest.TrackingIngest.TrackingFile
            * tracking.TrackingDevice.proj('tracking_position')
            & session_key).fetch(
        as_dict=True, order_by='tracking_device, trial')
    for tracking_file_info in tracking_files_info:
        trk_file = tracking_file_info.pop("tracking_file")
        trk_file = pathlib.Path(str(trk_file).replace("\\", "/"))

        video_path = pathlib.Path(
            tracking_root_data_dir
            / DEV_POS_FOLDER_MAPPING[tracking_file_info.pop("tracking_position")]
            / trk_file
        ).with_suffix(".mp4")
        video_metadata = dict(
            Behavior=dict(
                Movies=[
                    dict(
                        name=video_path.stem,
                        description=video_path.as_posix(),
                        unit="n.a.",
                        format="external",
                        starting_frame=[0, 0, 0],
                        comments=str(tracking_file_info),
                    )
                ]
            )
        )
        try:
            MovieInterface([video_path]).run_conversion(
                nwbfile=raw_nwbfile, metadata=video_metadata, external_mode=False
            )
        except FileNotFoundError:
            warnings.warn(f"No raw video file found at {video_path}.")
            continue

    if external_link:
        try:
            with NWBHDF5IO(raw_video_nwb_file.as_posix(), mode='w', manager=io.manager) as raw_io:
                raw_io.write(raw_nwbfile)
                print(f'\t\tWrite NWB 2.0 file: {raw_video_nwb_file.stem}')
        finally:
            io.close()
    else:
        return raw_nwbfile


def _get_session_identifier(session_key):
    water_res_num, sess_datetime = get_wr_sessdatetime(session_key)
    return f'{water_res_num}_{sess_datetime}_s{session_key["session"]}'


def export_recording(session_keys, output_dir='./',
                     overwrite=False, validate=False,
                     raw_ephys=False, raw_video=False):
    if not isinstance(session_keys, list):
        session_keys = [session_keys]

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_fullpaths = []
    for session_key in session_keys:
        session_identifier = _get_session_identifier(session_key)
        session_identifier += "_raw_ephys" if raw_ephys else ""
        session_identifier += "_raw_video" if raw_video else ""
        output_fp = (output_dir / f"{session_identifier}.nwb").absolute()
        # Write to .nwb
        if overwrite or not output_fp.exists():
            nwbfile = datajoint_to_nwb(session_key)

            if raw_ephys:
                nwbfile = _to_raw_ephys_nwb(session_key,
                                            linked_nwb_file=nwbfile, overwrite=False)
            if raw_video:
                nwbfile = _to_raw_video_nwb(session_key,
                                            linked_nwb_file=nwbfile, overwrite=False)

            with NWBHDF5IO(output_fp.as_posix(), mode='w') as io:
                io.write(nwbfile)
            print(f'\tWrite NWB 2.0 file: {output_fp.stem}')

        if validate:
            import nwbinspector
            with NWBHDF5IO(output_fp.as_posix(), mode='r') as io:
                validation_status = pynwb.validate(io=io)
            print(validation_status)
            for inspection_message in nwbinspector.inspect_all(path=output_fp):
                print(inspection_message)

        output_fullpaths.append(output_fp)

    return output_fullpaths
