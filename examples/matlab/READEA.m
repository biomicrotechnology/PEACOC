classdef READEA
    methods(Static)

        function [raw_data,sr] =read_raw(filepath)
            raw_data = h5read(filepath,'/data/trace');
            sr = h5readatt(filepath,'/data','sr');

        end




        function container = read_spikes(filepath,container)

            info = h5info(filepath);
            C = extractfield(info.Groups,'Name');
            if ~any(strcmp(C,'/dischargedict_cleaned'))
                disp(['No entries for dischargedict_cleaned in ',filepath]);
                return
            end


            container.spikes = h5read(filepath,'/dischargedict_cleaned/data/spikes');
            varnames = {'t_offset','t_total','polarity'};
            for ind = 1:length(varnames)
              container.(varnames{ind}) =h5readatt(filepath,'/dischargedict_cleaned/data',varnames{ind});
            end
        end





        function container = read_bursts(filepath,container,displaypath)
            %extracting bursts
            info = h5info(filepath);
            C = extractfield(info.Groups,'Name');
            if ~any(strcmp(C,'/burstclasses'))
                disp(['No entries for burstclasses in ',filepath]);
                return
            end

            %extracting clusternames if displaypath is present
            if exist('displaypath','var')
                fid = fopen(displaypath);
                conffile = fread(fid,inf);
                datastr = char(conffile');
                confs = jsondecode(datastr);
                C = confs.name_ranks;
                fields = fieldnames(C);
                numvec = cell2mat(struct2cell(C(1,1:end)));
                clustid_replacer = 'cname';
            else
                clustid_replacer = 'cid';
            end

            data = h5read(filepath,'/burstclasses/data/values');
            [nparams, nbursts] = size(data);

            %extract parameter values
            params = '';
            for idx = 0:nparams-1
                attr = ['i',num2str(idx)];
                param =  h5readatt(filepath,'/burstclasses/data/params',attr);
                if strcmp(param,'seizidx')
                    param='si';
                elseif strcmp(param,'clustid')
                    param= clustid_replacer;
                end
                params = char(params, param);
            end

            params = params(2:end,:);
            %fill in the values
            for idx=data(1,:)
                for varidx = 2:length(params)
                    param = strtrim(params(varidx,:));
                    if strcmp(param,'cname')
                        cidx = find(numvec==data(varidx,idx));
                        if isempty(cidx)
                            cidx = length(fields);
                        end
                        datain = char(fields(cidx));
                    else
                        datain = data(varidx,idx);
                    end
                    container.bursts(idx).(param) = datain;
                end


            end
        end




        function container = read_states(filepath,container)

            %finding out where state data is stored
            info = h5info(filepath);
            C = extractfield(info.Groups,'Name');
            if ~any(strcmp(C,'/states'))
                disp(['No entries for states in ',filepath]);
                return
            end
            C = extractfield(info.Groups,'Name');
            stateloc = info.Groups(strcmp(C, '/states'));
            C = extractfield(stateloc.Groups,'Name');
            dataloc = stateloc.Groups(strcmp(C, '/states/data'));

            entries = {'state' 'start' 'stop' 'begins' 'ends'};

            for ii=1:length(dataloc.Groups)
                group = dataloc.Groups(ii);
                %disp(group.Name)
                temp = strsplit(group.Name,'/');
                container.states(ii).id = char(temp(1,end));
                valfields = extractfield(group.Attributes,'Value');
                for var=entries
                    param = char(var);
                    index = find(strcmp(extractfield(group.Attributes,'Name'),param));
                    if ~isempty(index)

                        container.states(ii).(param) = valfields{index};
                    end
                end
            end

        end


        function container = read_artifacts(filepath,container)
            %reads artifacts and the actual time analyzed (ttotal-toffset-arttimes)

            %artifacts
            arttimes = h5read(filepath,'/dischargedict_cleaned/data/mask_startStop_sec');
            container.artifactTimes = arttimes;

            %1) artifacts
            if sum(arttimes(2))==0
                container.artifacts = NaN;
            else
                for ii=1:length(arttimes)
                    container.artifacts(ii).start= arttimes(ii,1);
                    container.artifacts(ii).stop= arttimes(ii,2);
                end
            end

            %2) durAnalyzed
            toffset = double(h5readatt(filepath,'/dischargedict_raw/data','t_offset'));
            ttotal = double(h5readatt(filepath,'/dischargedict_raw/data','t_total'));

            if sum(arttimes(2))==0
                artdur = 0;
            else
                temp = arttimes;
                temp(temp<toffset) = toffset;
                artdur = sum(temp(:,2)-temp(:,1));
            end
            container.durAnalyzed = ttotal-artdur-toffset;

        end

        function container = read_results(filepath,container,displaypath)

            if ~exist('container','var')
                container = {};
            end

            container.fileinfo = h5info(filepath);

            %extracting spikes
            container = READEA.read_spikes(filepath,container);


            %extracting bursts
            if exist('displaypath','var') %display part is optional
                container = READEA.read_bursts(filepath,container,displaypath);
            else
                container = READEA.read_bursts(filepath,container);
            end

            %readout states
            container = READEA.read_states(filepath,container);

            %artifacts and durAnalyzed
            container = READEA.read_artifacts(filepath,container);


        end
    end
end



