% 
% April 2014
% Implementation of Grip-Force model from Ulloa, Bullock, and Rhodes 2003.
% "Adaptive force generation for precision-grip lifting by a spectral timing
% model of the cerebellum". Neural Networks 16 (2003), p.521-528

% DESCRIPTION & NOTES (click on plus sign to read)
%{
% This code implements the model of precision gripping as detailed in the
% Ulloa et al paper listed above.  The main result to notice is the change
% in timing of the net adjustment to grip force S between trial 1 and trial
% 60. 

% As the model runs through practice trials, the synaptic weights (z) between the
% Parallel Fibers and Purkinje Cells and the synaptic weights (w) between the Mossy 
% Fibers and Nuclear Cells change.  These changes allow the model to learn to time 
% and scale a learned anticipatory GF (grip force) adjustment rather than simply 
% waiting for a slip-error to be sensed and then reacted to. The growing 
% anticipatory action leads, over a few tens of practice trials, to the near 
%elimination of slips.  

% The anticipation is necessary to reach high performance (low slip of the object 
% gripped) because there is a long delay (modeled as 50 msecs) between the onset of 
% feedback signals from slip-sensing mechanoreceptors and onset of muscle activation.

% The internal decision to lift the object is represented as a context signal C,
% which is sent as a "leading indicator" to the cerebellum 300 msecs before the 
% arm's lifting action is initiated via the GO signal.

% NOTES: 
% ERRATA:
% lambda_f = 0.5 (as in the precursor 1994 paper), not 5.0 as mis-printed
% in the 2003 paper.

% J_neg = the sum of f-sub-k for k=1:40 (as in the precursor 1994 paper); the 2003 
% paper includes weights on the f-sub-k terms, but they were not adapted, and chosen to
% ensure that the sum J_neg had a constant value of 1.0 throughout this
% simulation. In the simulation we omit the weights and make the sum J_neg = 1.0.
% In a more accurate simulation of the cerebellum, these weights, which
% would represent parallel fiber synapses onto the inhibitory stellate
% interneurons, would be adapted by a learning law in which pf-cf coincidence
% leads to LTP rahter than LTD.

% Because this model is of a system's reaction to slip induced by trying to 
$ lift an already grasped object, the initial hand aperture position is set to the 
% target aperture position value, consistent with both sides of the hand being in 
% light contact with the object at the very start of the lifting episode.  What 
% changes is the amount of force necessary, and when, for the object not to slip.

% Purkinje activation p and Nuclear activation n are set to their initial steady state 
% values of 1 and -1 respectively.

% New variables k1 and k2 were introduced since the slip signals were
% defined as proportions in the paper.  These constants were set to 1.

% alpha_gran was defined as an array of 40 granule cell activations with
% values evenly spaced from the interval [1.3 to 12].  These activations
% may be defined differently however: example in the 1994 precursor paper.

% For completeness, equation (21) of the change in Purkinje Cell synaptic
% weights (z-dot) was modified to include a separate learning rate for LTP
% and LTD (beta_zLTP and beta_zLTD) rather than just the single learning
% rate in the paper, although the single learning rate listed in the paper
% does the same thing.  In any case, beta_zLTP is set to 1 here.

% Also for completeness, a parameter kappa_n was introduced into equation
% (26) of the change in deep nuclear activity, to scale the cell's decay in
% activity.  It was set to 1.

% Two functions must be included for this code to work.  posRect(val) takes
% an array of values and returns the array positively rectified, so that
% all negative values are set to zero, but others left as is.  h(val, val_dot) 
% stipulates that a value of a variable will be returned when that variable's 
% derivative is positive and set to zero when its derivative is negative or zero.

%}

%% TIME DEFINITIONS

t_start     = 0;
t_on        = 1;                    % Time the GO signal is turned on
t_Con       = 0.7;                  % Time the Context signal Ci is turned on
t_end       = 3;
dt          = 0.001;
time        = t_start:dt:t_end;

%% SCALAR PARAMETER DEFINITIONS

% ARM TRANSPORT AND GRIP APERTURE COMPONENT PARAMETERS
alpha_G     = 30;                   % EQ. 11
alpha_D     = 30;                   % EQ. 1, EQ. 6
alpha_V     = 300;                  % EQ. 2, EQ. 7
alpha_U     = 40;                   % EQ. 5
alpha_S     = 40;                   % EQ. 10
beta_S      = 3;                    % EQ. 10
alpha_eA    = 50;                   % EQ. 13
gamma_eA    = 0.08;                 % EQ. 13
alpha_eT    = 50;                   % EQ. 15
gamma_eT    = 0.25;                 % EQ. 15
k1          = 1;                    % EQ. 16
k2          = 1;                    % EQ. 14

    %50 msec delay in error terms
delay       = ceil(0.05/dt);        % EQ. 13, EQ. 15

% CEREBELLUM COMPONENT PARAMETERS
alpha_m     = 0.2;                  % EQ. 17
beta_m      = 10;                   % EQ. 17
alpha_gran = linspace(1.3,12,40);   % EQ. 18
beta_gran   = 4;                    % EQ. 18
gamma_gran  = 0.1;                  % EQ. 18
alpha_el    = 0.1;                  % EQ. 19
beta_el     = 5;                    % EQ. 19
gamma_el    = 0.02;                 % EQ. 19
lambda_f    = 0.5;                  % EQ. 20
b           = 12;                   % EQ. 20
c           = 4;                    % EQ. 20
beta_zLTD   = 10;                   % EQ. 21
beta_zLTP   = 1.0;                  % EQ. 21
alpha_n     = 100;                  % EQ. 26
kappa_n     = 1;                    % EQ. 26
alpha_w     = -0.001;               % EQ. 27
beta_w      = 10;                   % EQ. 27

numtrials  = 60;            % The number of trials over which learning occurs


%% VARIABLE ARRAY INITIALIZATIONS

% ARM TRANSPORT ARRAY INITIALIZATIONS
DT       = zeros(size(time));
DT_dot   = zeros(size(time));
TT       = zeros(size(time));
PT       = zeros(size(time));
PT_dot   = zeros(size(time));
VT_dot   = zeros(size(time));
VT       = zeros(size(time));
OT       = zeros(size(time));
U        = zeros(size(time));
U_dot    = zeros(size(time));
G        = zeros(size(time));
G_dot    = zeros(size(time));
g0       = zeros(size(time));
eT       = zeros(size(time));
eT_dot   = zeros(size(time));
epsilonT = zeros(size(time));
LFs      = zeros(size(time));

TT(time>t_on)    = 5;                    % The Target Arm Position changes at t_on
g0(time>t_on)    = 1;                    % step input from a decision center of the brain
Go               = g0.*(time.^1.4);      % EQ. 12
LFs(time>t_on)   = 4;                    % arm load force to lift object of weight u

% GRIP APERTURE ARRAY INITIALIZATIONS
DA      = zeros(size(time));
DA_dot  = zeros(size(time));
TA      = 2*ones(size(time));     % Target Aperture is set arbitrarily to 2
PA      = 2*ones(size(time));     % Grip Aperture is at correct position prior to lifting
PA_dot  = zeros(size(time));
VA      = zeros(size(time));
VA_dot  = zeros(size(time));
OA      = zeros(size(time));
S       = zeros(size(time));
S_dot   = zeros(size(time));
eA      = zeros(size(time));
eA_dot  = zeros(size(time));
epsilonA  = zeros(size(time));
GFs     = zeros(size(time));

IO     = zeros(size(time));       % array to store values of climbing fiber signals h(eA)

TA(time>t_on)    = 2;        % Grip stays at target aperture position throughout trial
GFs(time>t_on)   = 10;       % Force needed to prevent slip of object of weight u and texture v

% CEREBELLAR ARRAY INITIALIZATIONS
C           = zeros(size(time));
m           = zeros(size(time));
m_dot       = zeros(size(time));
g           = zeros(40,length(time));
g_dot       = zeros(40,length(time));
el          = zeros(40,length(time));
el_dot      = zeros(40,length(time));
f           = zeros(40,length(time));
z           = ones(40,length(time));
z_dot       = zeros(40,length(time));
J_pos       = zeros(size(time));
J_neg       = zeros(size(time));
b_pj        = zeros(size(time));
p           = ones(size(time));      % p is set to the steady state it reaches with no adapted pf input
n           = -1*ones(size(time));   % n is set to the steady state it reaches with only Purkinje input
n_dot       = zeros(size(time));
w           = zeros(size(time));
w_dot       = zeros(size(time));
w_storage   = zeros(numtrials, length(time));
eA_storage  = zeros(numtrials, length(time));
z_storage   = zeros(numtrials, 40, length(time));

C(time>=t_Con & time<=(t_Con+.1))    = 1;


%% TRIALS LOOP

for trial=1:numtrials
    
        % Update initial conds synaptic weights w and z from previous trial
    if trial>1
        w(:) = w_storage(trial-1,end);
        for k=1:40
            z(k,:) = z_storage(trial-1,k,end);
        end
    end

    for i=2+delay:length(time)      % BEGIN TIME LOOP OF A SINGLE TRIAL
        
        %% GO SIGNAL
    
        G_dot(i) = alpha_G*(-G(i-1)+Go(i-1));                              % EQ. 11
        
        %% ARM TRANSPORT COMPONENT
        
        DT_dot(i)  = alpha_D*(-DT(i-1)+TT(i-1)-PT(i-1));                   % EQ. 1
        VT_dot(i)  = alpha_V*(-VT(i-1)+G(i-1)*posRect(DT(i-1)));           % EQ. 2
       %PT_dot(i)  = VT(i)  --> This equation is found below               EQ. 3
        OT(i)      = PT(i-1)+U(i-1);                                       % EQ. 4
        U_dot(i)   = alpha_U*eT(i-1);                                      % EQ. 5
        
        % LOAD ERROR
        eT_dot(i) = alpha_eT*(-eT(i-1)+gamma_eT*posRect(epsilonT(i-delay)));                            % EQ. 15
        epsilonT(i) = k1*(LFs(i-1)-U(i-1));                                                             % EQ. 16
        
        %% GRIP APERTURE COMPONENT
        
        DA_dot(i)  = alpha_D*(-DA(i-1)+TA(i-1)-PA(i-1));                                                % EQ. 6
        VA_dot(i)  = alpha_V*(-VA(i-1)+G(i-1)*DA(i-1));                                                 % EQ. 7
       %PA_dot(i)  = VA(i)  --------------------> This equation is found below                            EQ. 8
        OA(i)      = PA(i-1)-S(i-1);                                                                    % EQ. 9
        
        S_dot(i)   = alpha_S*(eA(i-1)+posRect(n(i-1))-beta_S*posRect(VA(i-1))*S(i-1));                  % EQ. 10
        
        % SLIP ERROR
        eA_dot(i) = alpha_eA*(-eA(i-1)+gamma_eA*posRect(epsilonA(i-delay)));                            % EQ. 13
        epsilonA(i) = k2*(GFs(i-1)-S(i-1));                                                             % EQ. 14
        
        %% CEREBELLAR COMPONENT
        
        m_dot(i) = -alpha_m*m(i-1) + beta_m*(1-m(i-1))*(C(i-1)+m(i-1));                                 % EQ. 17
        
        g_dot(:,i) = alpha_gran(:).*((1-g(:,i-1)).*m(i-1) - beta_gran*(g(:,i-1)+gamma_gran).*el(:,i-1));% EQ. 18
        
        el_dot(:,i) = -alpha_el*el(:,i-1) + beta_el*(1-el(:,i-1)).*(gamma_el*m(i-1)+f(:,i-1));          % EQ. 19
        
        
        
        %% EULER METHOD VARIABLE UPDATES
        
        %GO SIGNAL
        G(i)    = G(i-1)  + G_dot(i)*dt;
        
        %ARM TRANSPORT
        DT(i)   = DT(i-1) + DT_dot(i)*dt;
        VT(i)   = VT(i-1) + VT_dot(i)*dt;
        PT_dot(i)  = VT(i);                                                                             % EQ. 3
        PT(i)   = PT(i-1) + PT_dot(i)*dt;
        U(i)    = U(i-1)  + U_dot(i)*dt;
        eT(i)   = eT(i-1) + eT_dot(i)*dt;
        
        %GRIP APERTURE
        DA(i)   = DA(i-1) + DA_dot(i)*dt;
        VA(i)   = VA(i-1) + VA_dot(i)*dt;
        PA_dot(i)  = VA(i);                                                                             % EQ. 8
        PA(i)   = PA(i-1) + PA_dot(i)*dt;
        S(i)    = S(i-1)  + S_dot(i)*dt;
        eA(i)   = eA(i-1) + eA_dot(i)*dt;
        
        IO(i)   = h(eA(i),eA_dot(i));
        
        if trial==1 && i == length(time)
            storeS1  = S;
        end
        
        %CEREBELLUM
        m(i)    = m(i-1)    + m_dot(i)*dt;
        g(:,i)  = g(:,i-1)  + g_dot(:,i).*dt;

        f(:,i) = (b*(posRect(g(:,i)-lambda_f)).^2) ./ (1+c*(posRect(g(:,i)-lambda_f)).^2);              % EQ. 20
        z_dot(:,i) = f(:,i).*(beta_zLTP*(1-z(:,i-1)) - beta_zLTD*h(eA(i),eA_dot(i))*z(:,i-1));          % EQ. 21
        z(:,i)  = z(:,i-1)  + z_dot(:,i).*dt;       
        % EQ. 22 is represented by the function h.m        
        J_pos(i) = sum(f(:,i).*z(:,i));                                                                 % EQ. 24
        J_neg(i) = sum(f(:,i));                                                                         % EQ. 25
        b_pj(i) = J_pos(i)-J_neg(i);
        p(i) = 1 + ( (1.5*sign(b_pj(i))*b_pj(i)^2) / (1+b_pj(i)^2) );                                   % EQ. 23        
        n_dot(i) = alpha_n*(kappa_n*(-n(i-1))+m(i)*w(i-1)-p(i));                                        % EQ. 26        
        w_dot(i) = m(i)*(alpha_w*w(i-1)+beta_w*(1-w(i-1))*h(eA(i),eA_dot(i)));                          % EQ. 27

        el(:,i) = el(:,i-1) + el_dot(:,i).*dt;
        n(i)    = n(i-1)    + n_dot(i)*dt;
        w(i)    = w(i-1)    + w_dot(i)*dt;

        
        
        
    end
    
        % synaptic weights w and z1 through z40 are stored in "memory" for the
        % next trial, so that the system can learn from one trial to the next
    w_storage(trial,:)   = w;
    z_storage(trial,:,:) = z;

        % eA for each trial is stored in an array so it may be graphed and 
        % the change over time visualized
    eA_storage(trial,:)  = eA;
    
    
    
    
end

%% PLOTTING

figure;
plot(time,eA_storage);

figure
plot(time,f')
ylim([0 1])

figure
hold on
plot(time,storeS1,'b')
plot(time,S,'r')
plot(time,GFs,'k')
legend('Trial 1','Trial 60','GFs','Location','NorthWest')