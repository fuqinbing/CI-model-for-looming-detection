function CI_LMD = run_CI_LMD(videoFile)
% RUN_CI_LMD  Compute CI-LMD response from an input video
%
% INPUT:
%   videoFile - path to the input video file
%
% OUTPUT:
%   CI_LMD - column vector containing CI-LMD response for each frame

%% Read video
video = VideoReader(videoFile);

% Initialize frame storage
L = [];
frameNum = 0;

% Gets the frame rate of the video
frameRate = video.FrameRate;

% Read video frame by frame and convert to grayscale
while hasFrame(video)
    frame = readFrame(video);
    grayFrame = im2gray(frame);
    L = cat(3, L, grayFrame);
    frameNum = frameNum + 1;
end

%% Initialization
L = double(L);                 % Convert pixel values to double
P = zeros(size(L));            % Temporal difference layer
LA = double(L);                % DoG output

ON = zeros(size(L));           % ON channel
OFF = zeros(size(L));          % OFF channel

E_ON = zeros(size(L));         % Excitation (ON)
I_ON = zeros(size(L));         % Inhibition (ON)
E_OFF = zeros(size(L));        % Excitation (OFF)
I_OFF = zeros(size(L));        % Inhibition (OFF)

ON_COVER = zeros(size(L));     % Cross inhibition (ON)
OFF_COVER = zeros(size(L));    % Cross inhibition (OFF)

S_ON = zeros(size(L));         % ON response
S_OFF = zeros(size(L));        % OFF response
S = zeros(size(L));            % Combined response
G = zeros(size(L));            % Grouping layer

CI_LMD = zeros(frameNum,1);    % Final output
CI_LMD_1 = zeros(frameNum,1);  % Intermediate sum

%% Parameters
sigma_1 = 4;                  
sigma_2 = 2;                  
Beta_1 = 0.25; 
Beta_2 = 0.3;
w1 = 0.4;  
w2 = 0.6;
Theta_1 = 1;
Theta_2 = 0.9;
Theta_3 = 0.5;
c = 1/9 * ones(3,3);          
Tg = 10;                      
R = 0.9;                      

%% Main loop
for t = 2:frameNum
    
    % Temporal difference (P-layer)
    P(:,:,t) = L(:,:,t) - L(:,:,t-1);
    [M,N] = size(P(:,:,t));
    
    % Difference of Gaussian (DoG filtering)
    LA(:,:,t) = computeDoG(P(:,:,t), sigma_1, sigma_2);

    % ON and OFF pathways
    ON(:,:,t)  = max(0, LA(:,:,t)) + 0.1 * ON(:,:,t-1);
    OFF(:,:,t) = -min(0, LA(:,:,t)) + 0.1 * OFF(:,:,t-1);

    % Lateral inhibition kernel
    W = [1/8,1/4,1/8;
         1/4,0,1/4;
         1/8,1/4,1/8];

    % ON pathway processing
    E_ON(:,:,t) = ON(:,:,t);
    I_ON(:,:,t) = imfilter(ON(:,:,t-1), W, 'replicate');

    % OFF pathway processing
    E_OFF(:,:,t) = OFF(:,:,t);
    I_OFF(:,:,t) = imfilter(OFF(:,:,t-1), W, 'replicate');

    % Cross inhibition between ON and OFF channels
    ON_COVER(:,:,t)  =  E_OFF(:,:,t);
    OFF_COVER(:,:,t) =  E_ON(:,:,t);

    % ON and OFF responses
    S_ON(:,:,t)  = E_ON(:,:,t)  - w1 * I_ON(:,:,t)  + -Beta_1 * ON_COVER(:,:,t);
    S_OFF(:,:,t) = E_OFF(:,:,t) - w2 * I_OFF(:,:,t) + -Beta_2 * OFF_COVER(:,:,t);

    % Integration layer (nonlinear combination)
    S(:,:,t) = Theta_1 * S_ON(:,:,t) + ...
               Theta_2 * S_OFF(:,:,t) + ...
               Theta_3 * S_ON(:,:,t) .* S_OFF(:,:,t);

    % Grouping layer with spatial filtering
    G(:,:,t) = imfilter(S(:,:,t), c, 'replicate');
    for m = 1 : M
       for n = 1 : N
           if G(m,n,t) >= Tg
               G(m,n,t) = G(m,n,t);
           else
               G(m,n,t) = 0;
           end
       end    
   end

    % CI-LMD output computation
    CI_LMD_1(t) = sum(G(:,:,t), 'all');
    CI_LMD(t) = 1 / (1 + exp(-CI_LMD_1(t) / (M * N * R)));

    % Lower bound constraint
    if CI_LMD(t-1,1) < 0.5
        CI_LMD(t-1,1) = 0.5;
    else
        CI_LMD(t-1,1) = CI_LMD(t-1,1);
    end
end

end

function [DoG_response] = computeDoG(P, sigma_e, sigma_i)
% excitatory Gaussian kernel（5×5）
[u_e, v_e] = meshgrid(-2:2, -2:2);
G_e = (1/(2*pi*sigma_e^2)) * exp(-(u_e.^2 + v_e.^2) / (2*sigma_e^2));

% inhibitory Gaussian kernel（9×9）
[u_i, v_i] = meshgrid(-4:4, -4:4);
G_i = (1/(2*pi*sigma_i^2)) * exp(-(u_i.^2 + v_i.^2) / (2*sigma_i^2));

%  Gaussian filtering 
P_e = imfilter(P, G_e, 'corr', 'replicate');
P_i = imfilter(P, G_i, 'corr', 'replicate');

DoG_response = zeros(size(P));
[R,C] = size(P(:,:));
% compute DoG_response
  for m = 1:R
       for n = 1:C
           if P_e(m,n) >= 0 && P_i(m,n) >= 0
               DoG_response(m,n) = abs(P_e(m,n) - P_i(m,n));
           elseif P_e(m,n) < 0 && P_i(m,n) < 0
               DoG_response(m,n) = -abs(P_e(m,n) - P_i(m,n));
           else
               DoG_response(m,n) = 0;
           end
       end     
  end
end