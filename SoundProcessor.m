classdef SoundProcessor < handle
    % Implement several methods
    % to calculate MFCC
    properties (Access = private)
        %signal samples
        y = [];
        %sampling rate
        fs = 0;
        %number of frames
        frames = 0;
        %samples within each frame
        N = 0;
        %number of samples separate consecutive frames
        M = 0;
        wavFile = '';
    end
   
    methods
        %intialize file
        % read it
         function self = SoundProcessor(wavPath)
            [y, fs] = audioread(wavPath);
            self.y = y;
            self.fs = fs;
            metaFile = strsplit(wavPath,'.');
            wavFile = metaFile(1,1);
         end
         
         % do pre-emphasis on signal
         % parameter :
         % alfa --> e.g : 0.97
         function preEmphasis = doPreEmphasis(self, alfa)
            % calculate the length
            % remember self.y is M x 1 (column vector)
            % length() always return the largest
            totalSamples = length(self.y);
            
            %Create column vector result
            % zeros(size())
            % will initialize all elements to zero
            preEmphasis = zeros(size(self.y));
            % Pre-emphasis formula
            % https://dsp.stackexchange.com/questions/29914/pre-emphasis-filter-design
            for col = 2 : totalSamples
                preEmphasis(col,1) = self.y(col,1) - alfa * self.y((col-1),1);
            end
            
         end
         
         % do frame blocking on signals
         % parameter 
         % signal
         % N --> Number of samples in each frame
         % M --> Number of samples separate consecutive frame
         % reference : https://www.researchgate.net/publication/277090348_Development_an_Automatic_Speech_to_Facial_Animation_Conversion_for_Improve_Deaf_Lives?enrichId=rgreq-2dea3b70b9355741fd78b623d461b569-XXX&enrichSource=Y292ZXJQYWdlOzI3NzA5MDM0ODtBUzozNDg2NTIwNDExOTU1MjFAMTQ2MDEzNjUzMjg1OQ%3D%3D&el=1_x_2&_esc=publicationCoverPdf
         function frames = doFrameBlocking(self, signal, N, M)
            self.N = N;
            self.M = M;
            %length of signal
            len = length(signal);
            %number of overlap samples
            overlap = N - M;
            %begin from the first index
            startIndex = 1;
            %iterate until length of signal
            frames = 1;
            %clear all previous frame files
            self.clearFrameFiles();
            while startIndex <= len
                endIndex = (startIndex - 1) + N;
                if endIndex > len
                    endIndex = len;
                end
                subVector = signal(startIndex:endIndex, 1);
                %rowVector = subVector.';
                % heavy operation
                % we should leverage 
                % to file
                %frames = [frames,rowVector];
                
                %write to file
                fileName = sprintf('%s_frame_#%d.txt',self.wavFile,frames);
                self.signalToFile(fileName,subVector);
                
                startIndex = endIndex - overlap + 1;
                
                frames = frames + 1;
            end
            %minus one
            %because it will be real + 1
            frames = frames - 1;
            self.frames = frames;
         end
         
         function doHammingWindows(self)
            %clear previous file
            self.clearWindowedFrameFiles();
            for frameIndex = 1 : self.frames
                %read again filename of frame
                fileName = sprintf('%s_frame_#%d.txt',self.wavFile,frameIndex);
                %save it in tmp var
                frame = self.reloadFrame(fileName);
                %window read frame using hamming
                windowed = self.doHammingWindow(frame);
                %determine file name for windowed frame
                windowedFile = sprintf('%s_windowed_frame_#%d.txt',self.wavFile,frameIndex);
                %write it to file
                self.signalToFile(windowedFile,windowed);
            end
         end
         
         %compute dft on each windowed frames
         function doDFTs(self)
            %clean up dft files frame
            self.deleteFiles('dft_frame_#');
            for frameIndex = 1 : self.frames
                %read again filename of frame
                fileName = sprintf('%s_windowed_frame_#%d.txt',self.wavFile,frameIndex);
                %save it in tmp var
                frame = self.reloadFrame(fileName);
                %fourier transform
                %fourierTransformed = self.doDFT(frame);
                fourierTransformed = fft(frame);
                %determine file name for fourier transformed frame
                fourierFile = sprintf('%s_dft_frame_#%d.txt',self.wavFile,frameIndex);
                %write it to file
                self.signalToFile(fourierFile,fourierTransformed);
            end
         end
         
         function melFilterBank = doMelFilterBank(self, k, nFilt, nfft)
            lowFreq = 0;
            highFreq = (self.fs / 2.0);
            
            %convert to mel 
            lowMel = self.convertToMel(k,lowFreq);
            highMel = self.convertToMel(k,highFreq);
            
            % generate points --> nFilt + 2
            % row vector 1 x N
            melPoints = linspace(lowMel,highMel,(nFilt+2));
            
            %melFrequencies
            freqPoints = zeros(size(melPoints));
            for i = 1 : length(melPoints)
                freqPoints(1,i) = self.convertMelToF(k,melPoints(1,i));
            end
            
            %constructs new fMelBins
            bin = zeros(size(melPoints));
            
            %iterate over melPoints
            %convert each fft to bin
            lenMel = length(melPoints);
            
            for melIndex = 1 : lenMel
                bin(1,melIndex) = self.convertFftToBin(freqPoints(1,melIndex),nfft);
            end
            
            %iterate according to eq(2)
           
         end
         
         function mel = convertToMel(self,k,f)
             a = 1;
             b = (f/700);
             c = a+b;
            mel = k * log10(c);
         end
         
         function f = convertMelToF(self,k,m)
            a = (m*1.0) / (k*1.0);
            b = 10^(a);
            c = b - 1.0;
            f = 700*c;
         end
         
         function fftBin = convertFftToBin(self,f,nfft)
            a = (nfft+1) * f;
            fftBin = (a*1.0) / (self.fs);
         end
         
         %function to compute dft
         %refence https://www.bogotobogo.com/Matlab/Matlab_Tutorial_DFT_Discrete_Fourier_Transform.php
         function dftFrame = doDFT(self,signal)
            %create the result
            dftFrame =  zeros(size(signal));
            %calculate length of signal
            len = length(signal);
            %compute dft
            for k = 0:len-1
                for n = 0:len-1
                    dftFrame(k+1) = dftFrame(k+1) + signal(n+1)*exp(-j*pi/2*n*k);
                end
            end
         end
        
         %generate hamming window
         % based on N (number of samples on each frame
         % https://www.researchgate.net/publication/277090348_Development_an_Automatic_Speech_to_Facial_Animation_Conversion_for_Improve_Deaf_Lives?enrichId=rgreq-2dea3b70b9355741fd78b623d461b569-XXX&enrichSource=Y292ZXJQYWdlOzI3NzA5MDM0ODtBUzozNDg2NTIwNDExOTU1MjFAMTQ2MDEzNjUzMjg1OQ%3D%3D&el=1_x_2&_esc=publicationCoverPdf
         % refence https://dsp.stackexchange.com/questions/42661/how-is-a-signal-multiplied-with-the-window-functions
         function hammingWindow = doHammingWindow(self,signal)
             samples = length(signal);
             window = hamming(samples);
             hammingWindow = signal.*window;
         end
         
         
         %getter
         %function
         %get fs audio
         function fs = getFs(self)
            fs = self.fs;
         end
         %getter function
         % get audio samples
         function y = getY(self)
            y = self.y;
         end
        
         %plot against time
         %reference on mirlab books
         %http://mirlab.org/jang/books/audiosignalprocessing/matlab4waveRead.asp?title=4-2%20Reading%20Wave%20Files
         function plotSound(self,sound)
            %if no sound provided
            %assume it plots
            %our internal sound
            if(isempty(sound))
                sound = self.y;
            end
            
            len = length(sound);
            time = (1:len)/self.fs;
            plot(time,self.y)
         end
         
         %append file
         function appendToFile(self,fileLocation,val)
            outFile = fopen(fileLocation,'a+');
            fprintf(outFile,'%.15f\n',val);
            fclose(outFile);
         end
         
         %write signal to file
         function signalToFile(self,fileLocation,signal)
            len = length(signal);
            for col = 1 : len
                val = signal(col,1);
                self.appendToFile(fileLocation,val);
            end
         end
         
         %global function
         %to delete files with pattern
         function deletedFiles = deleteFiles(self,filePattern)
            %list all files in current directory
            files = dir;
            % number of files
            deletedFiles = length(files);
            %counter
            deletedFrames = 0;
            for fileIndex = 1 : deletedFiles
                %get first file
                file = files(fileIndex);
                %get filename
                fileName = file.name;
                %match filename with pattern
                findIndex = strfind(fileName,filePattern);
                %if pattern found
                %delete it
                if(isempty(findIndex) ~= true)
                    delete(fileName);
                    deletedFrames = deletedFrames + 1;
                end
            end
         end
         
         %clear all frame files inside
         %current directory
         function deletedFrames = clearFrameFiles(self)
            %list all files in current directory
            files = dir;
            % number of files
            numFiles = length(files);
            %counter
            deletedFrames = 0;
            for fileIndex = 1 : numFiles
                %get first file
                file = files(fileIndex);
                %get filename
                fileName = file.name;
                %match filename with pattern (frame_#)
                patt = sprintf('%s_frame_#', self.wavFile)
                findIndex = strfind(fileName,patt);
                %if pattern found
                %delete it
                if(isempty(findIndex) ~= true)
                    delete(fileName);
                    deletedFrames = deletedFrames + 1;
                end
            end
         end
         
         %duplication
         %not good, but for sake of
         %practice, we should do it
         function deletedWindowedFrames = clearWindowedFrameFiles(self)
            %list all files in current directory
            files = dir;
            % number of files
            numFiles = length(files);
            %counter
            deletedWindowedFrames = 0;
            for fileIndex = 1 : numFiles
                %get first file
                file = files(fileIndex);
                %get filename
                fileName = file.name;
                %match filename with pattern (frame_#)
                patt = sprintf('%s_frame_#', self.wavFile)
                findIndex = strfind(fileName,patt);
                %if pattern found
                %delete it
                if(isempty(findIndex) ~= true)
                    delete(fileName);
                    deletedWindowedFrames = deletedWindowedFrames + 1;
                end
            end
         end
         
         
         %re-read frame file 
         %into col vector
         %using dlmread (nice)
         function reloadedFrame = reloadFrame(self,fileName)
            reloadedFrame = dlmread(fileName);
         end
         %same
         function reloadedWindowedFrame = reloadWindowedFrame(self,fileName)
             reloadedWindowedFrame = dlmread(fileName);
         end
        
    end
    
end

