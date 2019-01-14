%%%%%%%%%%%%%% 5 word keyword classifier %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Without using the built-in mfcc function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% extracts 39 features for 50 files of partner 1 and partner 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir('**/*.wav');
audio = cell(1,length(files));
rows= zeros(1,length(files));

for r = 1:length(files)
    [audio{r},fs] = audioread(files(r).name);
    [rows(r), col] = size(audio{r});
end
min_value = min(rows);

final_feature_matrix = zeros(length(files),39* 62);

for p = 1:length(files)
    display(p)
    [audio{p},fs] = audioread(files(p).name);
    y1=audio{p}(:,1); % considering one channel of the audio file
    y=y1(1:min_value);
    
    % 2. Pass the signal through pre-empahsis 
%     preemph = [1 -0.97];
%     y = filter(1,preemph,y);

    % 3. check if the signal can be  divided into even frames
    no_of_frames = frames_formed(y,fs,0.025,0.010);

    % 5. divide the audio signal into frames of 25 ms with 10ms overlap
    frames = create_frames(y,fs,0.025,0.010); % calling the function

    % calculate the energy of the signal
    energy_feature =zeros(no_of_frames,1);
    for i=1:no_of_frames
        selected_frame= frames(i,:);
        energy_feature(i) = sum(selected_frame.^2);
    end

    % to calculate delta energy
    first_delta_energy = delta_calc(energy_feature);

    % to calculate double delta energy
    second_delta_energy = delta_calc(first_delta_energy);

    % 6. compute dft for all the frames formed
    N =512;
    dft_vec = dft(N, no_of_frames, frames);
    % keep first 257 dft coeff
    reqd_dft_coeff = dft_coeff(257, dft_vec);

    % 8. Mel -filter bank
    lower_freq = 0;
    mel_lower_freq = Hz_to_mel(lower_freq);
    upper_freq = (fs/2);
    mel_upper_freq = Hz_to_mel(upper_freq);

    % need 26 filters so consider 28 points in mel freuencies
    no_filters= 26;
    mel_frequencies = linspace(mel_lower_freq,mel_upper_freq,28);
    hz_freq = mel_to_Hz(mel_frequencies);
    fft_bins = zeros(length(hz_freq),1);

    % convert frequencies to fft bins
    for i =1: length(fft_bins)
        fft_bins(i) = floor((N+1)*hz_freq(i)/fs);
    end


    [mel_filter_bank, k] = filter_bank(hz_freq,257, fs);

    % multiply dct of each frame with all filter banks
    [rows_dft, col_dft] = size(reqd_dft_coeff);
    mel_filter = mel_filter_bank.';
    [rows_melfil, col_melfil] = size(mel_filter);
    out_fil_dft = zeros(rows_dft, rows_melfil);
    for l = 1: rows_dft
         for n =1: rows_melfil
         out_fil_dft(l,n) = dot(reqd_dft_coeff(l,:),mel_filter(n,:));
         end
    end

    % taking log 
    log_output = log((abs(out_fil_dft).^2));

    % taking dct
    ifft_output = zeros(size(log_output));
    [rows_dct, col_dct] = size(ifft_output);
    for o =1: rows_dct
        ifft_output(o,:) = (dct(log_output(o,:)));
    end

    mfcc_coeff = ifft_output(:, 1:12);

    % to calculate first derivative
    first_delta_mfcc = delta_calc(mfcc_coeff);

    % to calculate second derivative
    second_delta_mfcc = delta_calc(first_delta_mfcc);

    % contenating the feature matrix
    feature_matrix = zeros(no_of_frames,39);
    feature_matrix(:,1:12)= mfcc_coeff; % 12 -mfcc features
    feature_matrix(:,13) =  energy_feature; % 1- energy feature
    feature_matrix(:,14:25) = first_delta_mfcc; % 12- delta mfcc
    feature_matrix(:,26:37) = second_delta_mfcc; % 12- double delta mfcc
    feature_matrix(:,38)= first_delta_energy; % 1-delta energy feature
    feature_matrix(:,39) = second_delta_energy; % 1- double-delta energy feature
    B = reshape(feature_matrix,1,39*no_of_frames);
    final_feature_matrix(p,:) = B;
    
end
save('final_feature_matrix.mat','final_feature_matrix');

% [sig, fs] = audioread('Car_A1.wav');
% final_feature_matrix_test = features_calculation(sig,fs,1,min_value);

 %function to convert freq in Hz to mel frequecny
    function mel_freq = Hz_to_mel (freq)
        mel_freq = 1125 * log(1+(freq/700));
    end

    % function to convert mel frequencies to Hz
    function hz_freq = mel_to_Hz(mel_frequencies)
        hz_freq= zeros(length(mel_frequencies),1);
        for i=1:length(mel_frequencies)
            hz_freq(i) = 700*(exp(mel_frequencies(i)/1125)-1);
        end
    end

    % function to dft vectors for each frame
    function dft_vec = dft (N,no_of_frames,frames)
        com_dft_vec = zeros(no_of_frames,N);
        dft_vec = zeros(no_of_frames,N);
        for i=1: no_of_frames
           x = frames(i,:); 
           com_dft_vec(i,:) = fft(x,N);
           dft_vec(i,:) = abs( com_dft_vec(i,:));    
        end
    end


    % function to keep the first 257 coefficients of dft
    function reqd_dft_coeff = dft_coeff(no_of_coeff,dft_vec)
        [rows, col] = size(dft_vec);
        reqd_dft_coeff = zeros(rows,no_of_coeff);
        for i =1: rows
            reqd_dft_coeff(i,:) = dft_vec(i,1:no_of_coeff);
        end   

    end

    % function to calculate number of frames
    function no_of_frames = frames_formed(y,fs,frame_length,overlap)
        samples_signal = length(y);
        number_samples_1_frame = floor(fs*frame_length);
        no_overlap_samples = floor(fs*overlap);
        no_of_frames=floor((samples_signal - number_samples_1_frame)/no_overlap_samples +1);
    end

    
    % function to create frames
    function frames = create_frames (y,fs,frame_length,overlap)
        samples_signal = length(y);% number of samples in the signal
        number_samples_1_frame = floor(fs*frame_length);
        no_overlap_samples = floor(fs*overlap);
        strt_index =1;
        end_index=number_samples_1_frame;
        no_of_frames=floor((samples_signal - number_samples_1_frame)/no_overlap_samples +1);
        frames = zeros(no_of_frames,number_samples_1_frame);
        for i=1:no_of_frames
            frames(i,:)= y(strt_index:end_index);
            strt_index=strt_index+no_overlap_samples;
            end_index=strt_index+number_samples_1_frame-1;
        end 
    end


    function [values,k] = filter_bank(f,no_of_coeff,fs)
        M = length(f);
        k = linspace(0,fs/2,no_of_coeff);
        values = zeros(no_of_coeff,M-2);
        for m=2:M-1
            I = k >= f(m-1) & k <= f(m);
            J = k >= f(m) & k <= f(m+1);
            values(I,m-1) = (k(I) - f(m-1))./(f(m) - f(m-1));
            values(J,m-1) = (f(m+1) - k(J))./(f(m+1) - f(m));
        end
    end

    % function to cal delta and double delta

    function delta_value = delta_calc(input)
        delta_matrix = zeros(size(input));
        [ rows_delta_matrix , col_delta_matrix] = size(delta_matrix);
        matrix_to_calc_delta = zeros(rows_delta_matrix+4,col_delta_matrix);
        for i =3:rows_delta_matrix+2
            matrix_to_calc_delta(i,:) = input(i-2,:);
        end
        matrix_to_calc_delta(1,:) = input(1,:);
        matrix_to_calc_delta(2,:) = input(1,:);
        matrix_to_calc_delta(end-1,:)= input(end,:);
        matrix_to_calc_delta(end,:)= input(end,:);
        [row_matrix_to_calc_delta, col_matrix_to_calc_delta] = size(matrix_to_calc_delta);

        for i = 3: row_matrix_to_calc_delta-2
            delta_matrix(i,:) = (matrix_to_calc_delta(i+1,:) - matrix_to_calc_delta(i-1,:)+2*(matrix_to_calc_delta(i+2,:) - matrix_to_calc_delta(i-2,:) ))/10;
        end

        delta_value = delta_matrix(3:rows_delta_matrix+2,:);
    end
       
 %%%%%%%%%%%%%%%%%%%%%% Program to evaluate the algorithm -on 50 more files from partner 1 and partner 2 ; 125 files from other individuals  %%%%%%%%%%%%%%%%%%%%%% more files from 
  
a = load('final_feature_matrix.mat');
all_test_files= dir('**/*.wav');

train =["car","car","car","car","car","car""car","car","car","car","chair","chair","chair","chair","chair","chair","chair","chair","chair","chair","desk","desk","desk","desk","desk","desk","desk","desk","desk","desk","food","food","food","food","food","food","food","food","food","food","phone","phone","phone","phone","phone","phone","phone","phone","phone","phone"];

count=0;
for k = 1:length(all_test_files)
    [audio{k},fs] = audioread(all_test_files(k).name);
    y3=audio{k};
    y3=y3(:,1);
    [rows(k), col] = size(y3);
    y=y3(1:28160);
    final_feature_matrix_1= features_calculation(y,fs,1,28160);
    for j=1:50
        euc_dist(j) =norm(final_feature_matrix_1- a(1).final_feature_matrix(j,:));
    end
    [min_euc_dist,I]= min(euc_dist);
    find_file = all_test_files(I).name;
    for i=1:strlength(train)
        if(contains(find_file,train(i)))==1
            count=count+1;
            break;
            
        end
    end
    
end

accuracy =(count/k)*100;

function final_feature_matrix = features_calculation(y,fs,len,min_value)
       final_feature_matrix = zeros(len,39* 62);
      
    % 2. Pass the signal through pre-empahsis 
%         preemph = [1 -0.93];
%         y= filter(1,pr,y);

        % 3. check if the signal can be  divided into even frames
        no_of_frames = frames_formed(y,fs,0.025,0.010);

        % 5. divide the audio signal into frames of 25 ms with 10ms overlap
        frames = create_frames(y,fs,0.025,0.010); % calling the function

        % calculate the energy of the signal
        energy_feature =zeros(no_of_frames,1);
        for i=1:no_of_frames
            selected_frame= frames(i,:);
            energy_feature(i) = sum(selected_frame.^2);
        end

        % to calculate delta energy
        first_delta_energy = delta_calc(energy_feature);

        % to calculate double delta energy
        second_delta_energy = delta_calc(first_delta_energy);

        % 6. compute dft for all the frames formed
        N =512;
        dft_vec = dft(N, no_of_frames, frames);
        % keep first 257 dft coeff
        reqd_dft_coeff = dft_coeff(257, dft_vec);

        % 8. Mel -filter bank
        lower_freq = 0;
        mel_lower_freq = Hz_to_mel(lower_freq);
        upper_freq = (fs/2);
        mel_upper_freq = Hz_to_mel(upper_freq);

        % need 26 filters so consider 28 points in mel freuencies
        no_filters= 26;
        mel_frequencies = linspace(mel_lower_freq,mel_upper_freq,28);
        hz_freq = mel_to_Hz(mel_frequencies);
        fft_bins = zeros(length(hz_freq),1);

        % convert frequencies to fft bins
        for i =1: length(fft_bins)
            fft_bins(i) = floor((N+1)*hz_freq(i)/fs);
        end


        [mel_filter_bank, k] = filter_bank(hz_freq,257, fs);

        % multiply dct of each frame with all filter banks
        [rows_dft, col_dft] = size(reqd_dft_coeff);
        mel_filter = mel_filter_bank.';
        [rows_melfil, col_melfil] = size(mel_filter);
        out_fil_dft = zeros(rows_dft, rows_melfil);
        for l = 1: rows_dft
             for n =1: rows_melfil
             out_fil_dft(l,n) = dot(reqd_dft_coeff(l,:),mel_filter(n,:));
             end
        end

        % taking log 
        log_output = log((abs(out_fil_dft).^2));

        % taking dct
        ifft_output = zeros(size(log_output));
        [rows_dct, col_dct] = size(ifft_output);
        for o =1: rows_dct
            ifft_output(o,:) = (dct(log_output(o,:)));
        end

        mfcc_coeff = ifft_output(:, 1:12);

        % to calculate first derivative
        first_delta_mfcc = delta_calc(mfcc_coeff);

        % to calculate second derivative
        second_delta_mfcc = delta_calc(first_delta_mfcc);

        % contenating the feature matrix
        feature_matrix = zeros(no_of_frames,39);
        feature_matrix(:,1:12)= mfcc_coeff; % 12 -mfcc features
        feature_matrix(:,13) =  energy_feature; % 1- energy feature
        feature_matrix(:,14:25) = first_delta_mfcc; % 12- delta mfcc
        feature_matrix(:,26:37) = second_delta_mfcc; % 12- double delta mfcc
        feature_matrix(:,38)= first_delta_energy; % 1-delta energy feature
        feature_matrix(:,39) = second_delta_energy; % 1- double-delta energy feature
        B = reshape(feature_matrix,1,39*no_of_frames);
        for p=1:len
            final_feature_matrix(p,:) = B;
        end
end
 %function to convert freq in Hz to mel frequecny
    function mel_freq = Hz_to_mel (freq)
        mel_freq = 1125 * log(1+(freq/700));
    end

    % function to convert mel frequencies to Hz
    function hz_freq = mel_to_Hz(mel_frequencies)
        hz_freq= zeros(length(mel_frequencies),1);
        for i=1:length(mel_frequencies)
            hz_freq(i) = 700*(exp(mel_frequencies(i)/1125)-1);
        end
    end

    % function to dft vectors for each frame
    function dft_vec = dft (N,no_of_frames,frames)
        com_dft_vec = zeros(no_of_frames,N);
        dft_vec = zeros(no_of_frames,N);
        for i=1: no_of_frames
           x = frames(i,:); 
           com_dft_vec(i,:) = fft(x,N);
           dft_vec(i,:) = abs( com_dft_vec(i,:));    
        end
    end


    % function to keep the first 257 coefficients of dft
    function reqd_dft_coeff = dft_coeff(no_of_coeff,dft_vec)
        [rows, col] = size(dft_vec);
        reqd_dft_coeff = zeros(rows,no_of_coeff);
        for i =1: rows
            reqd_dft_coeff(i,:) = dft_vec(i,1:no_of_coeff);
        end   

    end

    % function to calculate number of frames
    function no_of_frames = frames_formed(y,fs,frame_length,overlap)
        samples_signal = length(y);
        number_samples_1_frame = floor(fs*frame_length);
        no_overlap_samples = floor(fs*overlap);
        no_of_frames=floor((samples_signal - number_samples_1_frame)/no_overlap_samples +1);
    end

    
    % function to create frames
    function frames = create_frames (y,fs,frame_length,overlap)
        samples_signal = length(y);% number of samples in the signal
        number_samples_1_frame = floor(fs*frame_length);
        no_overlap_samples = floor(fs*overlap);
        strt_index =1;
        end_index=number_samples_1_frame;
        no_of_frames=floor((samples_signal - number_samples_1_frame)/no_overlap_samples +1);
        frames = zeros(no_of_frames,number_samples_1_frame);
        for i=1:no_of_frames
            frames(i,:)= y(strt_index:end_index);
            strt_index=strt_index+no_overlap_samples;
            end_index=strt_index+number_samples_1_frame-1;
        end 
    end


    function [values,k] = filter_bank(f,no_of_coeff,fs)
        M = length(f);
        k = linspace(0,fs/2,no_of_coeff);
        values = zeros(no_of_coeff,M-2);
        for m=2:M-1
            I = k >= f(m-1) & k <= f(m);
            J = k >= f(m) & k <= f(m+1);
            values(I,m-1) = (k(I) - f(m-1))./(f(m) - f(m-1));
            values(J,m-1) = (f(m+1) - k(J))./(f(m+1) - f(m));
        end
    end

    % function to cal delta and double delta

    function delta_value = delta_calc(input)
        delta_matrix = zeros(size(input));
        [ rows_delta_matrix , col_delta_matrix] = size(delta_matrix);
        matrix_to_calc_delta = zeros(rows_delta_matrix+4,col_delta_matrix);
        for i =3:rows_delta_matrix+2
            matrix_to_calc_delta(i,:) = input(i-2,:);
        end
        matrix_to_calc_delta(1,:) = input(1,:);
        matrix_to_calc_delta(2,:) = input(1,:);
        matrix_to_calc_delta(end-1,:)= input(end,:);
        matrix_to_calc_delta(end,:)= input(end,:);
        [row_matrix_to_calc_delta, col_matrix_to_calc_delta] = size(matrix_to_calc_delta);

        for i = 3: row_matrix_to_calc_delta-2
            delta_matrix(i,:) = (matrix_to_calc_delta(i+1,:) - matrix_to_calc_delta(i-1,:)+2*(matrix_to_calc_delta(i+2,:) - matrix_to_calc_delta(i-2,:) ))/10;
        end

        delta_value = delta_matrix(3:rows_delta_matrix+2,:);
    end



