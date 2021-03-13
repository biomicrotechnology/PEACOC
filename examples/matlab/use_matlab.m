%% ACCESSING DATA
%
% We are in the directory where
%
% * analysis results (id__blipSpy.h5)
% * raw data (id__raw500.h5)
% * the readout script (READEA.m)
%
% are contained

cd '/MATLAB Drive/EA_readout'
dir

%% Retrieving all analyses results Option 1
% Option 1: extract all at once

id ='KA114_HC1_05'; %  'EP11_test'; % id of recording analyzed
ext = '__blipSpy';
fname = [id,ext,'.h5'];


Rec = READEA.read_results(fname,{},'configDisplay.json');

%% Retrieving analysis results Option 2
% Option 2: extract only specific results
Rec = {};

Rec.fileinfo = h5info(fname);

% extracting spikes
Rec = READEA.read_spikes(fname,Rec);


% extracting bursts
Rec = READEA.read_bursts(fname,Rec,'configDisplay.json');%display part is optional

% extracting states
Rec = READEA.read_states(fname,Rec);

% artifacts and durAnalyzed
Rec = READEA.read_artifacts(fname,Rec);


%% Retrieving raw data

ext = '__raw500';
rawfile = [id,ext,'.h5'];
[Rec.raw_data,Rec.sr] = READEA.read_raw(rawfile);


%% Using analysis results to calculate diagnostics

% number of discharges per second (Hz)
spikerate = length(Rec.spikes)/Rec.durAnalyzed

% rate of weak in inter-ictal phases (Hz)

% fraction of time spent EA-free

% fraction of time spent in ictal phase

% fraction of time spent in inter-ictal phases

% fraction of time spent in severe bursts

% median severity of severe events

%% Retrieving Diagnostics automatically



%% Visualizaton of results

% spikes

% bursts

% phases

% artifacts

% raw data


%% TO DO
%
% * calculate missing diagnostic measures indicated above
% * make READEA.diagonstics(Rec) function, that gathers these
% * visualizing spikes and bursts
%




